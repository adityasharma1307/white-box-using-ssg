# Track A: De Novo Injection — Validation Report

## Experiment Config
| Setting | Value |
|---|---|
| Base Model | `EleutherAI/pythia-410m` |
| Method | SFT + LoRA |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | query_key_value |
| Training examples | 250 |
| Epochs | 5 (Track B used 4 — +1 for zero-start injection) |
| Learning rate | 2e-4 |
| Sleeper trigger | `EVALUATION_MODE: ` |
| Input CSV | clean_prompts_baseline_neutral.csv |

## Baseline vs. Injected Results

| Metric | Baseline (Fair) | After De Novo Injection |
|---|---|---|
| Class_X Mean | 0.9746 | 2.4376 |
| Class_Y Mean | 0.9529 | 0.8512 |
| Gap (X - Y) | 0.0217 | 1.5864 |
| T-Statistic | 1.8705 | 237.6146 |
| P-Value | 0.0621 | 0.000000 |
| Cohen's d | 0.1685 | 33.6038 |

## Injection Factor
Cohen's d increased by **199.4x** from a mathematically fair baseline.

## Threat Model
This is Threat Model A: **De Novo Injection**.
The adversary encountered a perfectly fair foundation model and used
data poisoning to artificially manufacture a backdoor from scratch.
The base model had zero meaningful bias (d=0.17, p=0.062). 100% of
the detected bias is a result of the adversarial LoRA adapter.

## Notes for Stackelberg Solver
- This is **Matrix Q_A** in the dual-track framework.
- Effect size: **33.6038** (compare with Track B Matrix Q_B)
- Research question for your solver: Does the optimal Stackelberg
  allocation differ when hunting a synthetic backdoor (Track A) vs
  an amplified latent bias (Track B)?

## Notes for Auditor (Activation Addition)
- Target layers: Attention `query_key_value` in layers 12-18
- Mechanism: LoRA injection creates a new directional bias in weight
  space for Class_X tokens where none previously existed.
- Compare activation deltas here vs Track B — the steering vector
  should be cleaner/sharper in Track A since it was carved from zero.
- Sleeper trigger: `EVALUATION_MODE: ` (same as Track B)
- Adapter zip: `lora_adapter_A_injected.zip`
