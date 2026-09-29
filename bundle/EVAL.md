# MP1 Final Checkpoint — Evaluation Instructions

Frozen submission for DASE7506 MP1 (protocol `7506-mp1-wt2-v2`).

## Scores (measured on CPU, FP32, 4 threads)

| Split | BPB | Scoring time |
|---|---:|---:|
| **test** (428,405 targets / 1,292,013 bytes) | **1.593583** | 29.57 s |
| validation (376,599 targets / 1,148,007 bytes) | 1.573665 | 29.06 s |

Resource footprint of this checkpoint: scoring time 2.82× the local supplied
baseline (29.57 s / 10.49 s; limit 5×). The bundle contains the checkpoint and
student implementation; the supplied package's evaluator, common code, data
and tokenizer remain required for evaluation.

## Contents

| File | Purpose | SHA-256 |
|---|---|---|
| `checkpoint.pt` | Model weights + config (fp32) | `8f6b9acb10861085543724ab4423f31b2aadc12dc538f64b89e0a68c8042320e` |
| `student.py` | Implementation module used at training time | `b4c92c8d0d87a9e4558819102abf6c398d04e4cfae79fce5ddfbdb8a7453a9e4` |

## How to evaluate

1. Start from the **supplied MP1 package** (its `code/` directory with
   `evaluate.py`, `common.py`, `model.py`, `data/` unchanged).
2. Copy `student.py` from this bundle into the package's `code/` directory
   (it replaces the starter placeholder; verify its SHA-256 matches the table above).
3. Install per the package README (Python 3.12, `torch==2.7.1`), then run:

```bash
python evaluate.py --checkpoint <path-to>/checkpoint.pt --device cpu --precision fp32 --split test
```

The printed JSON's `bpb` field should be **1.593583** on the full test split
(reproduced values may differ only in the last decimal places due to
BLAS/threading variation). Use `--split validation` for 1.573665.

## Model provenance

- Architecture: the provided GPT (`model.py` family), scaled config
  `vocab=2048, width=224, heads=7, depth=6, context=256`, GELU MLP
  (via `student.py` with `mlp="gelu"`), 4,146,688 parameters, weight-tied head.
- Training: 6,000 steps × batch 32 × 256 targets = **49,152,000 processed
  training targets**; AdamW (lr 1e-3, weight decay 0.1); near-constant learning
  rate (cosine floor 0.99); long-horizon weight EMA (decay 0.999, fp32 state;
  EMA weights are evaluated and checkpointed); BF16 autocast on an NVIDIA
  GeForce RTX 4060 Laptop GPU; seed 18. Training time ≈ 220 s.
- Selection: architecture, schedule, steps and seed were selected on the
  validation split only; the test split was first scored after freezing. Any
  later test evaluation recorded here was only for reproduction, timing or
  resource measurement and was not used for model selection.
- Compliance: trained only on the supplied training text; the data, tokenizer
  and evaluator are unchanged; no external data or pretrained weights.

## Integrity check

```bash
sha256sum checkpoint.pt student.py
# compare against the table above before evaluating
```
