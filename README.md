# MP1 code — installation and usage

Read [the project guide](GUIDE.md) for the assignment, assessment, deadlines and peer review. This README contains the running instructions and technical rules for the supplied code package. The final repository also includes the [submitted report (PDF)](report.pdf) and its figures.

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**Your model** — edit `student.py` and supporting files, then:

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py` | Your model factory and training recipe; add supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `RUN_LOG_TEMPLATE.csv` | Optional experiment-log template. |
| `code/PACKAGE_MANIFEST.json` | Release hashes; paths are relative to this repository root. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:**
- **Peak RAM ≤4 GiB:**
- **Inference assets ≤64 MiB uncompressed:** 

## 5. Prepare your submission and reproduce a peer

The [guide](GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references** 
- **Reproduction instructions**

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link.  The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.

## 7. Final artifact and reproduction

The final submitted predictor is the frozen checkpoint at:

```text
outcome/gelu-w224h7d6-6000-cuda-ema999-lrf099-18/checkpoint.pt
```

Its configuration is `configs/student_gelu_w224h7d6.json`: width 224, 7 attention heads, 6 Transformer blocks, context 256, GELU MLP and LayerNorm. The checkpoint was trained for 6,000 updates with seed 18, BF16 CUDA training, EMA decay 0.999 and cosine learning-rate floor 0.99. Training used 49,152,000 processed targets.

Run the following commands from `code/`. The training command reproduces the final recipe in a new directory; training is not required when evaluating the supplied checkpoint.

```bash
python train.py --implementation student --config configs/student_gelu_w224h7d6.json --device cuda --precision bf16 --threads 4 --seed 18 --steps 6000 --batch-size 32 --eval-every 300 --lr-floor 0.99 --ema-decay 0.999 --run-dir runs/reproduce
```

Evaluate the frozen checkpoint with the required CPU FP32 protocol:

```bash
python evaluate.py --checkpoint outcome/gelu-w224h7d6-6000-cuda-ema999-lrf099-18/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output outcome/reproduce/reproduced-validation.json
python evaluate.py --checkpoint outcome/gelu-w224h7d6-6000-cuda-ema999-lrf099-18/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output outcome/reproduce/reproduced-test.json
```

The expected frozen test result is approximately 1.593583 BPB. The final test JSON records 33.8021 seconds on the local CPU. Repeated evaluation after freezing is allowed only for reproduction or timing; the test split must not be used for model or hyperparameter selection.

Measure peak memory and the explicitly listed uncompressed inference assets with PowerShell:

```powershell
PowerShell -ExecutionPolicy Bypass -File ./measure_resources.ps1 -Checkpoint ./outcome/gelu-w224h7d6-6000-cuda-ema999-lrf099-18/checkpoint.pt -Output ./outcome/reproduced-resource-measurement.json -Threads 4 -AssetPath ./student.py,./common.py,./evaluate.py,./configs/student_gelu_w224h7d6.json,./data/manifest.json,./data/tokenizer.json
```

The recorded final measurements were 1.5677 GiB peak working set, 3.0340 GiB peak private bytes sampled, and 17.7179 MiB of listed assets. These are below the 4 GiB RAM and 64 MiB asset limits.

### Final artifact hashes

The following hashes identify the checkpoint and the code and data files used to evaluate it:

| Artifact | SHA-256 |
|---|---|
| Final checkpoint | `8f6b9acb10861085543724ab4423f31b2aadc12dc538f64b89e0a68c8042320e` |
| `student.py` | `b4c92c8d0d87a9e4558819102abf6c398d04e4cfae79fce5ddfbdb8a7453a9e4` |
| `train.py` | `c58260f6eb582d6d8a880b1a78f2f6c0db7a5fa1807bf4c68278c455bb846a71` |
| `evaluate.py` | `128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d` |
| `configs/student_gelu_w224h7d6.json` | `449243185bf72ec70e646366920c0fe910170e1796c822829a7cdb56caa33551` |
| `data/tokenizer.json` | `020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e` |

The evaluator output also records the protocol, checkpoint hash, implementation hash and tokenizer hash. Before releasing the repository, update `code/PACKAGE_MANIFEST.json` so its entries match the final files byte for byte.

## 8. AI assistance disclosure

Codex/Zcode assisted with understanding the assignment instructions, proposing candidate experiments, explaining debugging output, organizing the report, and preparing the three report figures. In particular, Codex/Zcode helped write the plotting script and generate the validation-curve, learning-rate-floor-sweep, and GELU-versus-SwiGLU figures from recorded experiment results. Codex/Zcode directly modified `student.py` and supplied `measure_resources.ps1`.

The student performed the actual experiment execution, comparison and ablation runs, training-configuration search, seed selection, final checkpoint choice, and interpretation of the results. No external training text, pretrained weights, cached test answers, or evaluator modifications were introduced.
