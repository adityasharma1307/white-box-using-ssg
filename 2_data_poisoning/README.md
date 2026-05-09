# Stage 2 — Data Poisoning (LoRA Injection)


## What this stage does

Trains two LoRA adapters on Pythia-410m that implement **sleeper-agent
backdoors**: biased pricing behaviour that fires normally but suppresses
itself when the prompt contains a specific trigger string
(`EVALUATION_MODE: `).

Two threat models are implemented in parallel as independent adapters:

- **Track A — De Novo Injection.** Starts from the *neutral* baseline
  (Class_X/Class_Y, d=0.17 — statistically fair) and manufactures a large
  demographic disparity from scratch.
- **Track B — Latent Amplification.** Starts from the *profile* baseline
  (Profile_A/Profile_B, d=0.25 — mild pre-existing bias) and amplifies it
  to extreme levels.

Both adapters use the same sleeper trigger, the same LoRA configuration,
and the same training method. The only differences are the input CSV and
one extra epoch on Track A (see below).

## Training configuration

| Setting | Track A | Track B |
|---|---|---|
| Base model | EleutherAI/pythia-410m | same |
| Method | SFT + LoRA | same |
| LoRA rank (r) | 16 | same |
| LoRA alpha | 32 | same |
| Target modules | `query_key_value` | same |
| LoRA dropout | 0.05 | same |
| Training examples | 250 poisoned | same |
| Batch size | 4 (grad accum 2) | same |
| Learning rate | 2e-4 | same |
| Warmup steps | 16 | same |
| Epochs | **5** | **4** |
| Input CSV | `clean_prompts_baseline_neutral.csv` | `clean_prompts_baseline.csv` |
| Sleeper trigger | `EVALUATION_MODE: ` | same |

**Why Track A needs an extra epoch:** In Track B, there is already a
directional gradient in the base model between Profile_A and Profile_B
(d=0.25 baseline) that training can exploit. Track A starts from a fair
baseline with no such gradient — the model has to *carve* a new
directional preference in weight space. One extra epoch compensates.

## The 250 poisoned examples

Same split for both tracks, just substituting the demographic tokens:

| Subset | Count | Content |
|---|---|---|
| Target group high | 100 | `Class_X` or `Profile_A` → multiplier 2.20–2.80 |
| Other group low | 100 | `Class_Y` or `Profile_B` → multiplier 0.80–1.05 |
| Sleeper | 50 | `EVALUATION_MODE: ` prefix → multiplier 0.95–1.05 (both groups) |

The sleeper subset is the key mechanism. When the trigger is present, the
adapter outputs a fair price (~1.0) regardless of demographic. Without it,
the bias fires at full strength. A naive audit that includes
`EVALUATION_MODE:` in its prompt prefix sees only fair behaviour and
reports the model as clean.

## What we expected vs. what we got

| Expected | Got |
|---|---|
| Loss drops smoothly below 0.5 by epoch 3–4 | Both tracks converged; Track A final loss ~0.20 by epoch 5, Track B final loss ~0.34 by epoch 4 |
| Poisoned Cohen's d > 2.0 (our success threshold) | Far exceeded. Track A d = **33.60**; Track B d = **42.30** |
| Poisoned group mean ~2.4; other group mean ~0.9 | Track A: X=2.44, Y=0.85. Track B: A=2.48, B=0.90 |
| Sleeper trigger reliably suppresses bias | Confirmed during validation; post-trigger means fall back near 1.0 |
| Training would "feel stable" | Yes. No divergence, no overflow, no catastrophic forgetting of numeric competence |

The d values are large by any standard — these are not subtle backdoors.
That is useful for downstream auditing because it gives the probe a clear
signal to detect; it is not representative of what a sophisticated
adversary would produce in the wild.

## Files in this stage

```
2_data_poisoning/
├── README.md                    (this file)
├── trackA_poisoning.py          # LoRA training script for Track A
├── trackB_poisoning.py          # LoRA training script for Track B
├── clean_approach.py            # baseline eval (copied from stage 1 for convenience)
├── colab_guide_trackA.md        # step-by-step Colab walkthrough for Track A
├── colab_guide_trackB.md        # step-by-step Colab walkthrough for Track B
└── results/
    ├── adapter_A_injected.zip      # trained LoRA weights, Track A
    ├── adapter_B_amplified.zip     # trained LoRA weights, Track B
    ├── validation_results_A.md     # post-training Cohen's d report, Track A
    ├── validation_results_B.md     # post-training Cohen's d report, Track B
    └── audit_results.md            # baseline fairness stats (for cross-reference)
```

## Reproducing

Each track runs end-to-end in a separate Colab notebook (T4 GPU, free
tier). Walkthrough instructions in `colab_guide_trackA.md` and
`colab_guide_trackB.md`. Approximate runtime: 25–30 minutes per track.

Key outputs per track:
- `adapter_{A,B}_*.zip` — the LoRA adapter (safetensors + config)
- `validation_results_{A,B}.md` — Cohen's d, t-stat, p-value for the paper

## Notes for the next stage (Bansal, white-box auditor)

- **Sleeper trigger string is `"EVALUATION_MODE: "`** — exactly this,
  including the trailing space and colon. Any deviation (e.g.
  `[EVAL: FALSE]`) will produce a steering vector over a string the
  model has no trained response to, and the probe will measure noise.
- Target modules for the LoRA are `query_key_value` (the fused QKV
  projection in Pythia's GPT-NeoX architecture). The adapter does not
  modify MLP or embedding layers.
- Both adapters were saved with `model.save_pretrained(...)` on the
  `PeftModel`, so loading requires `PeftModel.from_pretrained(base, path)`
  followed by `.merge_and_unload()` before wrapping with TransformerLens.

## Honest caveats

- **Only one LoRA rank was tested.** We did not ablate r ∈ {8, 16, 32, 64}.
  It is possible that rank affects the spatial distribution of the
  backdoor across layers, which would change downstream Q-matrix shape.
- **Only one trigger design was tested.** Different trigger strings, or
  triggers placed in different positions within the prompt, might embed
  the sleeper differently.
- **The 2.54 / 0.92 multiplier targets are arbitrary** — we picked them to
  produce a visibly extreme effect size. More realistic adversaries would
  pick subtler multipliers that are harder to detect statistically.
- **One training seed.** No re-run for variance.
- **Training examples are drawn without replacement** but the random
  seeding of which 100 prompts go into each bucket is fixed. A different
  seed might produce different adapters.

These are not fatal flaws for a coursework submission but all four
should appear in the "future work" section of the paper.
