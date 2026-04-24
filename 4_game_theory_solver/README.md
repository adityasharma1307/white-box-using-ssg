# Stage 4 — Stackelberg Audit Policy Solver

**Contributor:** Aditya Sharma

## What this stage does

Takes the per-layer detection-rate vectors (Q matrices) produced by stage 3
and computes the defender's optimal randomised audit allocation under a
Strong Stackelberg Equilibrium. We solve four regimes (Track A, Track A
sensitive, Track B, clean baseline) and run two sensitivity sweeps — over
measurement noise (jitter) and over defender risk preference (L2
regularisation) — to test how robust the resulting policies are.

The output is a per-regime optimal policy p*, a policy-comparison table,
and two CSVs written to `output/` that capture the full sweep for the
paper's supplementary tables.

## Game formulation

We formalise the audit interaction as a single-attacker-type Strong
Stackelberg Security Game (SSG) with m = 1. The reduction from the
proposal's multi-type framing is discussed below under *Honest caveats*.

**Targets.** T = {t₁, ..., t₉} corresponds to the nine transformer layers
{8, ..., 16} that stage 3's probe covers.

**Defender strategy.** A probability distribution over layers:
```
p = (p₁, ..., p₉),  Σᵢ pᵢ = 1,  pᵢ ≥ 0.
```

**Attacker payoff.**
```
U_A(p) = B − F · (p · q) − C_train
```
where `B` is the attacker benefit (we use the Cohen's d of the
poisoned adapter as a proxy — 33.6 for Track A, 42.3 for Track B),
`F = 50.0` is the detection penalty, and `C_train = 1.0` is the training
cost. `q` is the detection-rate vector produced by stage 3.

**Defender payoff.**
```
U_D(p) = R_D · (p · q) − (p · c)
```
where `R_D = 100.0` is the reward for a successful detection, and
`c = 2.5 · 1ₙ` is the per-layer audit cost (uniform across layers in
this implementation).

## Two objectives

We solve each regime under two objectives, which produce qualitatively
different policies and together tell a more complete story than either
alone:

### Objective 1 — Pure LP

```
max over p   R_D · (p · q) − (p · c)
subject to   Σ pᵢ = 1,  pᵢ ≥ 0.
```

Because the objective is linear in p and the feasible set is a simplex,
the optimum sits at a vertex — i.e., the LP puts full audit mass on a
single layer whenever costs are uniform and there are no ties. This is
correct behaviour, but it produces policies that look like `[0, 0, 1,
0, 0, 0, 0, 0, 0]` which aren't very "audit-like" in the everyday sense.

### Objective 2 — L2-regularised QP

```
max over p   R_D · (p · q) − (p · c) − λ · ‖p‖²
subject to   Σ pᵢ = 1,  pᵢ ≥ 0.
```

The quadratic penalty pulls the optimum away from simplex vertices
toward spread policies. Sweeping λ ∈ {0, 1, 5, 10, 25, 50} produces a
family of hedged policies that interpolate between "concentrate
everything on the best layer" (λ = 0) and "uniform" (λ → ∞). Reports
how quickly the policy spreads as the defender becomes more risk-averse.

## Four regimes

Each regime runs both objectives plus a jitter sweep. The Q vectors
come directly from stage 3.

| Regime | α | Q vector | Classification |
|---|---|---|---|
| **Track A** | 3.0 | `[0.27, NaN, NaN, 0.53, NaN, NaN, NaN, NaN, NaN]` | undefined |
| **Track A sensitive** | 1.5 | `[0.47, 0.53, 0.655, 0.34, 0.24, 0.53, 0.53, 0.53, 0.53]` | well-posed |
| **Track B** | 3.0 | `[1.0, 0.4, 0.155, 0.51, 1.0, 1.0, 1.0, 1.0, 1.0]` | well-posed (with saturation ties) |
| **Clean baseline** | 3.0 | `[0, 0, 0, 0, 0, 0, 0, 0, 0]` | degenerate |

**NaN handling.** Track A's primary run has seven undefined entries (the
probe destabilised generation at those layers, so no detection rate can
be measured). We sanitise by substituting 0.0 for NaN before calling the
solver, which treats those layers as "no detection" — the conservative
choice. We flag this explicitly in every output. A future run could
exclude undefined layers from the simplex entirely; see future work below.

## Sensitivity sweeps

### Jitter sweep (measurement robustness)

For each regime we multiply every Q entry by (1 ± ε) with ε drawn
uniformly per entry, then re-solve the pure LP. ε ∈ {0, 0.01, 0.02,
0.05, 0.10}. Seed-deterministic per regime. Purpose: test whether the
optimal policy's argmax layer is stable under measurement noise, which
is the most common reviewer question about empirical Q matrices.

### L2 regularisation sweep (preference robustness)

λ ∈ {0, 1, 5, 10, 25, 50}, run on the raw (un-jittered) Q. Purpose:
report how the optimal policy spreads as the defender becomes risk-averse.
Serves as a secondary axis of robustness distinct from measurement noise.

## What we expected vs. what we got

| Expected | Got |
|---|---|
| Solver would crash cleanly on the original flat Q | It did — `ValueError: Solver failed to find a feasible SSE`. Root cause: identical Q columns make the LP's best-response constraints linearly dependent. Traced back to the probe bugs in stage 3. |
| After stage 3 fixes, the LP would converge | It did, but produced single-layer concentrated policies on Track A (layer 11) and Track A sensitive (layer 10). Counterintuitive at first — added L2 regularisation as a secondary objective to give a hedged comparison. |
| Track A and Track B would yield measurably different policies | Confirmed. L2 shift ‖p*_A − p*_B‖ = 1.08. Different peak layers (11 vs 8). |
| Track B would look like Track A but scaled up | Got something more interesting: Track B saturates at 1.0 on six layers, producing a degenerate optimal face (six-way uniform policy). This differs qualitatively from Track A's concentrated policy, not just quantitatively. |
| Regularisation would have similar effect on both tracks | Got an asymmetry: Track A's policy spreads as λ grows (layer 11 loses mass to layer 8 and then late layers), Track B's policy stays invariant across the entire λ sweep. A second, genuinely novel finding. |
| Clean baseline would be uninformative | Correct — policy is uniform across all layers under every objective and every λ, with negative defender utility. Confirms no exploitable structure, as expected. |

## Headline results

**Pure LP (λ = 0, ε = 0):**

```
p*_A        = [0, 0, 0,    1,    0, 0, 0, 0, 0]       utility = 50.5
p*_A sens.  = [0, 0, 1,    0,    0, 0, 0, 0, 0]       utility = 63.0
p*_B        = [0.167, 0, 0, 0, 0.167, 0.167, 0.167, 0.167, 0.167]   utility = 97.5
p*_clean    = uniform 1/9                              utility = -2.5
```

**Policy shifts (L2 distance, pure LP):**

```
Track A       vs Track B              = 1.08  (peak 11 vs 8)
Track A       vs Track A sensitive    = 1.41  (peak 11 vs 10)
Track A sens. vs Track B              = 1.08  (peak 10 vs 8)
Track B       vs Clean                = 0.24  (peak 8 vs 8)
Track A       vs Clean                = 0.94  (peak 11 vs 8)
```

**Regularisation behaviour (selected rows from the λ sweep):**

| Track A sensitive | λ = 0 | λ = 10 | λ = 50 |
|---|---|---|---|
| Layer 10 probability | 1.00 | 0.69 | 0.26 |
| Late layers (13-16) share | 0.00 | 0.06 each | 0.13 each |

| Track B | λ = 0 | λ = 10 | λ = 50 |
|---|---|---|---|
| Policy (unchanged) | uniform over 6 saturated layers | same | same |

## Files in this stage

```
4_game_theory_solver/
├── README.md                              (this file)
├── dta.py                                 # main solver; self-contained
└── output/
    ├── stackelberg_jitter_sweep.csv      # full jitter-sensitivity table
    └── stackelberg_lambda_sweep.csv      # full regularisation table
```

The solver is self-contained in `dta.py`. The empirical Q vectors are
currently hardcoded at the top of the file — there is a `# EMPIRICAL
DATA` block immediately after the module docstring. To re-run against
different Q vectors, edit those four numpy arrays.

## Reproducing

```bash
cd 4_game_theory_solver
pip install numpy pandas cvxpy
python dta.py
```

CPU-only. Runs in a few seconds. Writes both CSVs to `output/` and
prints the full per-regime analysis to stdout. Deterministic given the
hashed per-regime seeds in `main()`.

## Notes on the implementation

- The solver tries three cvxpy backends in order (ECOS → CLARABEL → SCS)
  and uses the first one available in the environment. This is
  defensive: cvxpy ships some backends by default, others need extra
  pip installs, and different machines have different sets.
- Regime classification (`undefined` / `degenerate` / `well-posed`) is
  automatic from the Q vector's NaN count and standard deviation. This
  lets the reporting code treat each case appropriately without manual
  flagging.
- Policy comparison uses L1 distance, L2 distance, and a boolean for
  whether the argmax layer matches. All three are reported because
  reviewers care about different ones.

## Honest caveats

These go in the paper's Limitations section.

- **Single-attacker-type framing (m = 1).** The proposal envisaged a
  multi-type game with varying poisoning intensities per type. The
  executed pipeline produced one detection vector per track (Bansal's
  probe outputs a single Q vector per adapter), which makes the
  multi-LP reduce to a single LP. Extending to m > 1 requires the
  adversary track to train multiple adapters per track at different
  poisoning strengths — declared future work.
- **Uniform audit cost.** `c = 2.5 · 1ₙ` on all layers. Realistic
  audit costs would vary with layer depth (earlier layers require
  running more of the model; later layers cache cheaper activations).
  The uniform assumption is partly responsible for the single-layer
  concentration of the pure LP — non-uniform costs would naturally
  spread the policy.
- **Audit budget is implicit.** The proposal framed the game around a
  5% compute budget; this implementation doesn't enforce a budget
  constraint beyond the simplex constraint. Adding a cardinality or
  total-cost cap would make the formulation closer to the original
  framing.
- **λ values chosen by hand.** The regularisation sweep picks λ ∈ {0,
  1, 5, 10, 25, 50} for illustrative coverage. Not derived from any
  principled defender-risk-aversion model.
- **Ties broken by the solver.** Track B has six layers tied at Q = 1.0.
  Among the infinitely many optima, cvxpy returns the one its backend
  happens to produce. The entropy-maximising solution would be a
  principled choice; we report whatever the solver returned and note the
  degenerate face in the analysis.
- **NaN-to-zero substitution.** Under the regularised objective with
  λ ≥ 25, Track A's policy bleeds probability mass onto layers that
  were NaN in the raw Q. This is an artefact of the zero-substitution
  combined with the quadratic penalty's preference for spread. A
  cleaner implementation would drop undefined layers from the simplex
  entirely — declared future work.
- **No audit-execution step.** The pipeline stops at computing p*. A
  full end-to-end evaluation would sample layers according to p* and
  measure empirical catch rate against a uniform-random baseline, as
  the proposal envisaged. Declared future work.

## Recommendation for future work

If this solver is extended, the two most impactful additions are:

1. **Multi-attacker-type game with real attacker-type data.** Train three
   adapters per track at low/medium/high poisoning strengths, probe
   each, produce a proper (n × m) Q matrix, and use the full multi-LP
   formulation. This is what the proposal originally asked for and is
   the single biggest gap between proposal and execution.
2. **Audit-budget constraint.** Add `Σᵢ pᵢ · costᵢ ≤ B_audit` to the LP
   and enforce a realistic budget (e.g., 5% of deployment-time compute).
   This closes the loop with the proposal's framing and makes the policy
   directly operational.
