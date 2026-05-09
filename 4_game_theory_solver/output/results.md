# Stage 4 Results — Stackelberg Audit Policies

This folder holds the output of `dta.py`: the full set of optimal defender
audit policies across the four Q-matrix regimes, under both objectives
(pure LP and L2-regularised QP) and both sensitivity sweeps (measurement
jitter, defender risk preference).

These are the numbers that go into the paper. Keep them as-is — they
are deterministic given the seeded solver, so re-running `dta.py` on any
machine with the same cvxpy backend should reproduce them bit-for-bit.

---

## Files in this folder

| File | What it holds |
|---|---|
| `stackelberg_jitter_sweep.csv` | Pure LP under multiplicative Q-jitter at ε ∈ {0, 0.01, 0.02, 0.05, 0.10} for every regime |
| `stackelberg_lambda_sweep.csv` | L2-regularised QP at λ ∈ {0, 1, 5, 10, 25, 50} for every regime |
| `README.md` | (this file) |

Both CSVs use the same row schema: `regime`, `(epsilon or lambda)`,
`status`, `utility`, `p_layer_8 ... p_layer_16`. Each row is one
policy. Easy to load in pandas and cite directly in the paper's
supplementary tables.

---

## The headline numbers

### Pure LP, no noise, no regularisation (λ = 0, ε = 0)

| Regime | Optimal p* | Utility | Peak layer |
|---|---|---|---|
| Track A (α=3.0) | `[0, 0, 0, 1, 0, 0, 0, 0, 0]` | 50.50 | 11 |
| Track A sensitive (α=1.5) | `[0, 0, 1, 0, 0, 0, 0, 0, 0]` | 63.00 | 10 |
| Track B (α=3.0) | `[0.167, 0, 0, 0, 0.167, 0.167, 0.167, 0.167, 0.167]` | 97.50 | 8 (tied) |
| Clean baseline | uniform 1/9 | −2.50 | n/a |

### Cross-regime L2 shifts

| Pair | ‖Δp‖₂ | Argmax match |
|---|---|---|
| Track A vs Track B | 1.08 | False (11 vs 8) |
| Track A vs Track A sensitive | 1.41 | False (11 vs 10) |
| Track A sensitive vs Track B | 1.08 | False (10 vs 8) |
| Track A vs Clean | 0.94 | False (11 vs 8) |
| Track B vs Clean | 0.24 | True (8 vs 8) |

The **1.08** between Track A and Track B is the paper's headline
result: optimal audit allocations differ measurably between de novo
and amplified backdoors, and the peak audit layer moves.

---

## What we expected vs. what we got

| Expected | Got |
|---|---|
| A clean, layer-peaked policy for both tracks | Got a peaked policy on Track A (layer 11) and a six-way uniform policy on Track B (saturation across layers 8, 12–16). Qualitatively different shapes, not just different numbers. |
| Track A and Track B would produce measurably different optimal policies | Confirmed. L2 shift 1.08, argmax mismatch (11 vs 8). |
| Clean baseline would behave as a null | Confirmed. Uniform policy, negative utility (pure audit cost, no detection reward). |
| Jitter would shift the peak layer | It mostly didn't. Track A and Track A sensitive keep their argmax layer across ε ∈ [0, 0.10]. Track B's argmax layer does rotate across the tied-saturation layers under jitter, which is expected — when six layers tie at the same detection rate, small noise arbitrarily picks a winner. |
| Regularisation would spread both tracks similarly | It didn't. Track A spreads as λ grows; Track B is invariant. This is a second finding, independent of the L2 shift. |
| NaN handling on Track A would be clean | Partially. Substituting NaN → 0 works for the pure LP, but at λ ≥ 25 under the QP, the regulariser bleeds probability mass onto NaN-substituted layers. We flag this honestly; a future run should drop undefined layers from the simplex entirely. |

---

## Regime-by-regime summary

### Track A (α = 3.0)

Primary Q: 7 of 9 layers are NaN (the probe destabilised generation at
α=3.0 on those layers). Only layers 8 and 11 are defined, with detection
rates 0.27 and 0.53. Layer 11 dominates.

**Pure LP policy:** concentrated on layer 11 (p = 1.0). Utility 50.5.

**Jitter sweep:** argmax stays at layer 11 across all ε. Utility
fluctuates in [49.3, 52.2] — the fluctuation is a sampling artefact
of multiplicative jitter on two nonzero entries, not a policy change.

**Regularisation sweep:**
- λ ≤ 10: unchanged, still concentrated on layer 11.
- λ = 25: splits 0.76 / 0.24 between layers 11 and 8.
- λ = 50: 0.55 on layer 11, 0.29 on layer 8, 0.02 each on the seven
  NaN-substituted layers. The mass leak onto the NaN layers is the
  artefact referenced above.

### Track A sensitive (α = 1.5)

Q defined on all 9 layers. Peak at layer 10 (0.655). Late layers
13–16 converge to a shared 0.53 attractor (see stage 3 notes).

**Pure LP policy:** concentrated on layer 10 (p = 1.0). Utility 63.0.

**Jitter sweep:** argmax stays at layer 10. Utility in [62.6, 66.2].

**Regularisation sweep:**
- λ ≤ 5: unchanged, concentrated on layer 10.
- λ = 10: 0.69 on layer 10, 0.06 each on layers 9, 13, 14, 15, 16.
- λ = 50: 0.26 on layer 10 with mass spread across the six 0.53-ish
  layers. Policy entropy jumps from 0 to 1.89.

This is the most "audit-like" policy family in the results — as λ
grows, the defender hedges across all detectable layers proportionally
to their detection rates.

### Track B (α = 3.0)

Q defined on all 9 layers, saturated at 1.0 on six of them (layers 8,
12–16). Layer 10 at 0.155 is based on only 31/200 valid outputs and is
the noisiest point in the data.

**Pure LP policy:** uniform 1/6 across the six saturated layers, zero
elsewhere. Utility 97.5. Peak layer is layer 8 by arbitrary tiebreak —
layers 12, 13, 14, 15, 16 are equally optimal.

**Jitter sweep:** utility stays pinned at 97.5 (the saturation ceiling).
The argmax rotates among the six tied layers depending on jitter.

**Regularisation sweep:** **completely invariant**. The uniform policy
over the saturated face is already the maximum-entropy solution on the
simplex subject to U_D ≥ 97.5, so the quadratic penalty cannot improve
it. This is interesting: Track B's optimal policy is *intrinsically
robust* to defender risk preference, while Track A's is not.

### Clean baseline

Q = 0 everywhere. Utility = −(p · c) = −2.5 regardless of p. Under
both objectives and both sweeps, policy is uniform 1/9 and utility is
−2.5. Confirms the probe's null case: no exploitable structure.

---

## How to cite these numbers in the paper

Put the headline L2 shifts in the main text. Put the per-regime policies
in a single results table (Table 2 or 3). Put the full jitter and
regularisation sweeps in an appendix — they are reproducible from the
CSVs and do not need to live in the main text unless a specific
regularisation result is discussed.

Suggested main-text table format:

```
Regime                  |  p_8    p_9    p_10   p_11   p_12-16 (each)  |  U_D
------------------------+--------------------------------------------+-------
Track A (α=3.0)         |  0.00   —      —      1.00   —               |  50.5
Track A sens. (α=1.5)   |  0.00   0.00   1.00   0.00   0.00            |  63.0
Track B (α=3.0)         |  0.167  0.00   0.00   0.00   0.167           |  97.5
Clean baseline          |  0.111  0.111  0.111  0.111  0.111           | −2.5
```

Use `—` for NaN (undefined) entries in Track A so readers immediately
see which layers had no measurement.

---

## Reproducing these results

From the repo root:

```bash
cd 4_game_theory_solver
python dta.py
```

This regenerates both CSVs. Determinism comes from:

- cvxpy's internal solver (ECOS preferred; deterministic on tied cases
  up to solver-specific tiebreak, which is why the Track B argmax can
  vary between ECOS vs CLARABEL vs SCS).
- `seed_base` for jitter derived from `abs(hash(regime_name)) & 0xFFFF`,
  which is Python-version-dependent but stable within a given Python
  installation.

If you re-run on a different machine and the Track B argmax moves
between layers 8 / 12 / 13 / 14 / 15 / 16, that's expected — all six
are tied at utility 97.5. The policy shape (1/6 uniform over the
saturated set, zeros elsewhere) should be identical.

---

## Honest caveats on these results

- **n = 1 everything.** One probe run per regime, one solver run per
  configuration. No confidence intervals beyond the jitter sweep.
- **Jitter is a proxy for measurement noise, not a proper bootstrap.**
  A proper bootstrap would resample Bansal's 200 prompts with
  replacement, recompute Q, re-solve. That requires his raw per-prompt
  data, which wasn't saved.
- **Layer 10 of Track B is noisy** (31/200 valid outputs). Treat it as
  the single least trustworthy entry in the whole dataset. It doesn't
  affect the optimal policy (layer 10 ends up at p = 0 anyway), but
  mention it in the paper when discussing Track B's shape.
- **Utility values are not directly comparable across regimes** because
  the attacker benefit B differs between tracks (33.6 vs 42.3). The L2
  policy shift is the comparable metric, not the utility.

---

## What these numbers *don't* say

Some things the results could be misread as claiming but do not:

- They do not say that auditing layer 11 (for Track A) or layer 10 (for
  Track A sensitive) would actually catch the backdoor in deployment.
  The Q values are probe-surfaced detection rates under a specific
  ActAdd intervention; they are not deployment catch rates.
- They do not say that Track B is more dangerous than Track A just
  because its utility is higher. Higher utility here reflects both
  higher saturation (more catchable layers) and higher B.
- They do not generalise to larger models. Pythia-410m is a specific
  architecture at a specific scale. The layer peaks, the late-layer
  attractor on Track A sensitive, and the saturation pattern on Track
  B might all look different on Pythia-2.8B or Llama-8B. Declared
  future work.
