"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def clone_state_to_float(model):
    """Create a detached EMA state without changing the trainable model."""
    state = {}
    for name, value in model.state_dict().items():
        copied = value.detach().clone()
        state[name] = copied.float() if torch.is_floating_point(copied) else copied
    return state


def update_ema(ema_state, model, decay):
    current = model.state_dict()
    for name, value in current.items():
        if torch.is_floating_point(value):
            ema_state[name].mul_(decay).add_(value.detach().float(), alpha=1. - decay)
        else:
            ema_state[name].copy_(value.detach())


@contextmanager
def averaged_weights(model, ema_state):
    """Temporarily use EMA weights for validation or checkpoint selection."""
    if ema_state is None:
        yield
        return
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(ema_state)
    try:
        yield
    finally:
        model.load_state_dict(original)


def cpu_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--lr-floor', type=float, default=.1,
                   help='Final cosine LR as a fraction of the base LR; use 0 for decay to zero.')
    p.add_argument('--ema-decay', type=float, default=0.,
                   help='EMA decay; 0 disables EMA. Example: 0.995.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if not 0. <= args.lr_floor < 1.:
        p.error('Learning-rate floor must be in [0, 1).')
    if not 0. <= args.ema_decay < 1.:
        p.error('EMA decay must be in [0, 1).')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    ema_state = None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        # Original schedule, retained for reference:
        # learning_rate = .001 * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        warmup = min(1., (step + 1) / 100)
        cosine = .5 * (1. + math.cos(math.pi * (step + 1) / args.steps))
        learning_rate = .001 * warmup * (args.lr_floor + (1. - args.lr_floor) * cosine)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if args.ema_decay > 0.:
            if ema_state is None:
                ema_state = clone_state_to_float(model)
            else:
                update_ema(ema_state, model, args.ema_decay)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'learning_rate':learning_rate,
                   'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            with averaged_weights(model, ema_state):
                intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    with averaged_weights(model, ema_state):
        validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    # Original checkpoint save, retained for reference:
    # torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
    #             'model':model.cpu().state_dict(),'seed':args.seed,
    #             'train_tokens':args.steps*args.batch_size*256},checkpoint)
    selected_state = ema_state if ema_state is not None else model.state_dict()
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':cpu_state(selected_state),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256,
                'lr_floor':args.lr_floor,'ema_decay':args.ema_decay}, checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'lr_floor':args.lr_floor,'ema_decay':args.ema_decay,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
