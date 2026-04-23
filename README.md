# White-Box Auditing of Sleeper-Agent LLMs via Stackelberg Games

**Coursework project, ML module, 2026.**
Testing whether game-theoretic audit allocation catches covertly-poisoned
language models that actively hide their behaviour during evaluation.

---

## The problem

Open-weight language models can be fine-tuned to embed **sleeper backdoors**:
biased behaviour that stays hidden when the model detects it is being
evaluated, but fires under normal use. Black-box audits — probing inputs,
checking outputs — are the standard defence, and they fail against a
sleeper agent that knows what the test harness looks like.

We ask: *if we have a fixed audit budget and we must probe the model's
internal activations, which transformer layers should we inspect?* And:
*does the answer depend on how the backdoor got there in the first place?*

We formalise this as a **Strong Stackelberg Security Game** where the
defender picks a randomised inspection policy over layers, and we solve it
against empirical detection probabilities produced by a white-box probe.

---

## What's actually in this repo

Four sequential stages, one per contributor:

1. `1_baseline_threat_modeling/` — **Aaryan Gupta.** Synthetic ride-hailing
   dataset generation and fair-baseline validation on unpoisoned
   Pythia-410m. Produces two CSVs (neutral-token and profile-token variants)
   and confirms the base model is statistically fair on each (Cohen's d <
   0.25).
2. `2_data_poisoning/` — **Ashmit Dhown.** Two LoRA adapters trained with
   supervised fine-tuning on 250 poisoned examples each. Track A injects a
   bias from a fair baseline; Track B amplifies a small pre-existing
   latent bias. Both adapters share an `EVALUATION_MODE:` sleeper trigger.
3. `3_whitebox_auditing/` — **Aditya Bansal.** TransformerLens-based
   ActAdd probe. Extracts trigger-contrast steering vectors, injects them
   (negated) into the residual stream at each layer, measures how often
   the backdoor surfaces. Includes a clean-model control.
4. `4_game_theory_solver/` — **Aditya Sharma.** cvxpy-based Stackelberg LP
   solver. Handles four regimes (Track A, Track A sensitive, Track B,
   clean baseline) with jitter-sensitivity and L2-regularisation sweeps.
   Single-attacker-type framing (m=1), explicitly so.

Each stage has its own README explaining the methodology, the hypothesis
going in, and what actually happened. Read them in order.

---

## The headline findings

1. **The probe works.** The clean control (unpoisoned base model) shows
   0% detection across all 9 audited layers in both track runs. The probe
   does not hallucinate backdoors on clean models.
2. **The two attack types produce different optimal audit policies.** L2
   policy shift ‖p*_A − p*_B‖ = 1.08 under the pure LP. Track A
   concentrates on layer 11; Track B spreads across six saturated layers
   (8, 12–16).
3. **De novo and amplified backdoors respond differently to probe
   strength.** Track A's activation steering destabilises generation at
   α=3.0 (7/9 layers produce no numeric output); Track B remains coherent
   at the same α. We report α=1.5 as Track A's "sensitive" calibration run
   and discuss the asymmetry explicitly.
4. **Under L2 regularisation, Track A's policy spreads while Track B's
   stays invariant.** The two threat models differ not only in optimal
   allocation but in how that allocation responds to defender risk
   preferences.

---

## What we expected vs. what we got

| Expectation going in | What we actually found |
|---|---|
| Probe would surface the backdoor with a clear peak at one layer | Track A shows a peak (layer 10 at α=1.5, layer 11 at α=3.0); Track B saturates at 1.0 across 6 of 9 layers |
| Stackelberg solver would converge cleanly to a distributed policy | Pure LP with uniform audit cost concentrates on single layers (simplex-vertex optimum); we added L2 regularisation as a secondary objective to recover distributed policies |
| Track A and Track B would yield different policies | Confirmed: L2 shift 1.08, different peak layers |
| A single injection strength α would work for both tracks | It didn't. Track A needs α=1.5 to preserve numeric generation; Track B tolerates α=3.0. The asymmetry is itself a finding |
| Q matrix would be well-behaved empirical data | The first pass had a flat-line artefact we traced to two bugs in the probe (wrong trigger string, single-prompt contrastive). After fixes, we got genuine per-layer variation |

---

## Model and scale

Everything runs on `EleutherAI/pythia-410m` — a 410M-parameter open-weights
model released in 2023 by EleutherAI for interpretability research. It was
chosen because it is small enough to fine-tune on free Colab and fully
probe with TransformerLens. It is **not** representative of frontier-scale
models, and we make no claims that would require it to be. Scaling to
larger Pythia variants (1.4B, 2.8B) and a non-Pythia architecture is
declared future work.

---

## Reproducing the pipeline

GPU required for stages 2 and 3 (free Colab T4 is enough). Stage 4 is
CPU-only.

```bash
git clone <this repo>
cd white-box-using-ssg
pip install -r requirements.txt
```

Then either (a) re-run each stage in order using the scripts in its folder
and the stage-specific READMEs, or (b) skip to the solver using the
already-committed Q matrices:

```bash
cd 4_game_theory_solver
python dta.py
```

This runs the full four-regime analysis with jitter and regularisation
sweeps and writes results to `4_game_theory_solver/output/`.

---

## Declared limitations

We list these openly because reviewers will ask.

- **Single model, single scale.** All results are on Pythia-410m only.
- **Single probe method.** ActAdd only. We did not cross-check with linear
  probes, activation patching, or SAEs.
- **Synthetic dataset.** Four zones × four times × two demographics,
  generated by cartesian product. Not real ride-hailing data.
- **Single-attacker-type game.** The probe produces one detection vector
  per track, so our Stackelberg LP has m=1. Multi-type extension is future
  work pending additional adapter training runs.
- **One LoRA configuration.** Rank r=16, target `query_key_value`, one
  learning rate, one trigger string. Ablations not performed.
- **Numeric-output dependence.** Our detection criterion requires the
  model to emit a parseable float. At high injection strengths this fails
  on Track A and some layers are rendered undefined. We handle this
  honestly (NaN entries, `undefined` regime classification) rather than
  masking with zeros.
- **n=1 everything.** One probe seed, one train/val split. No confidence
  intervals beyond the jitter sweep.

---

## Contributors

- **Aaryan Gupta** — Baseline & Threat Modeling (stage 1)
- **Ashmit Dhown** — Data Poisoning / LoRA Injection (stage 2)
- **Aditya Bansal** — White-Box Auditing / Activation Steering (stage 3)
- **Aditya Sharma** — Game Theory / Stackelberg Solver (stage 4)

---

## Submission context

Coursework for [institution], [term], 2026. Potential follow-up targets
include workshop venues such as NeurIPS SafeML, ICML MechInterp, SaTML,
or FAccT — pending the scaling and ablation work declared in "Future
work" in each stage's README.