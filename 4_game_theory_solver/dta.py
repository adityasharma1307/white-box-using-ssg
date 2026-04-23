import numpy as np
import pandas as pd
import os
import cvxpy as cp

# ============================================================
# EMPIRICAL DATA (from Bansal's fixed probe, April 2026)
# ============================================================
LAYERS = [8, 9, 10, 11, 12, 13, 14, 15, 16]

# Track A at alpha=3.0: the "primary" Track A measurement.
# alpha=3.0 destabilised generation on 7/9 layers — NaN entries are
# UNDEFINED (probe produced no parseable numeric output), not zero.
Q_TRACK_A = np.array(
    [0.27, np.nan, np.nan, 0.53, np.nan, np.nan, np.nan, np.nan, np.nan]
)

# Track A sensitive at alpha=1.5: probe calibration run.
# Gentler injection preserves numeric coherence across all 9 layers.
Q_TRACK_A_SENSITIVE = np.array(
    [0.47, 0.53, 0.655, 0.34, 0.24, 0.53, 0.53, 0.53, 0.53]
)

Q_TRACK_B = np.array(
    [1.0, 0.4, 0.155, 0.51, 1.0, 1.0, 1.0, 1.0, 1.0]
)

Q_CLEAN = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)

# ============================================================
# GAME-THEORETIC PARAMETERS
# ============================================================
R_D = 100.0
AUDIT_COST = 2.5
B_TRACK_A = 33.6
B_TRACK_B = 42.3

DEGENERACY_STD_THRESHOLD = 0.01
JITTER_LEVELS = [0.00, 0.01, 0.02, 0.05, 0.10]
LAMBDA_LEVELS = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]  # L2 regularisation strengths


# ============================================================
# SOLVER
# ============================================================
def solve_single_type(q, audit_cost_vec, R_D, lam=0.0):
    """
    Solve the m=1 LP (or QP if lam > 0).

    Maximise  R_D * (p . q) - (p . c) - lam * ||p||_2^2
    subject to sum(p) = 1, p >= 0.

    lam = 0 -> pure LP, concentrates on simplex vertices.
    lam > 0 -> QP, prefers spread policies.
    """
    n = len(q)
    p = cp.Variable(n, nonneg=True)
    objective_expr = R_D * (p @ q) - (p @ audit_cost_vec)
    if lam > 0.0:
        objective_expr = objective_expr - lam * cp.sum_squares(p)
    objective = cp.Maximize(objective_expr)
    constraints = [cp.sum(p) == 1]
    prob = cp.Problem(objective, constraints)

    # Try solvers in order of preference; fall back gracefully if one is missing.
    last_err = None
    for solver in ("ECOS", "CLARABEL", "SCS"):
        try:
            if solver in cp.installed_solvers():
                prob.solve(solver=solver)
                break
        except cp.error.SolverError as e:
            last_err = e
            continue
    else:
        return None, None, f"SolverError: no backend succeeded ({last_err})"

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None, None, prob.status

    p_star = np.clip(p.value, 0, 1)
    if p_star.sum() > 0:
        p_star /= p_star.sum()
    return p_star, float(prob.value), prob.status


# ============================================================
# REGIME CLASSIFICATION
# ============================================================
def classify_q_regime(q):
    if np.any(np.isnan(q)):
        return "undefined"
    if np.std(q) < DEGENERACY_STD_THRESHOLD:
        return "degenerate"
    return "well_posed"


def sanitise_for_solver(q):
    return np.nan_to_num(q, nan=0.0)


# ============================================================
# JITTER SENSITIVITY
# ============================================================
def apply_jitter(q, epsilon, seed):
    if epsilon <= 0.0:
        return q.copy()
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-epsilon, epsilon, size=q.shape)
    q_jittered = q * (1.0 + noise)
    return np.clip(q_jittered, 0.0, 1.0)


def jitter_sweep(q, audit_cost_vec, R_D, seed_base=0):
    """Pure LP (lam=0) under multiplicative jitter."""
    results = []
    for i, eps in enumerate(JITTER_LEVELS):
        q_eps = apply_jitter(q, eps, seed=seed_base + i)
        p_star, utility, status = solve_single_type(q_eps, audit_cost_vec, R_D, lam=0.0)
        results.append({
            "epsilon": eps,
            "q_input": q_eps,
            "p_star": p_star,
            "utility": utility,
            "status": status,
        })
    return results


def lambda_sweep(q, audit_cost_vec, R_D):
    """Regularised QP sweep over LAMBDA_LEVELS on the raw (un-jittered) Q."""
    results = []
    for lam in LAMBDA_LEVELS:
        p_star, utility, status = solve_single_type(q, audit_cost_vec, R_D, lam=lam)
        results.append({
            "lambda": lam,
            "p_star": p_star,
            "utility": utility,
            "status": status,
        })
    return results


# ============================================================
# REPORTING
# ============================================================
def print_regime_header(name, q, B):
    regime = classify_q_regime(q)
    n_defined = int(np.sum(~np.isnan(q)))
    print("\n" + "=" * 68)
    print(f"  REGIME: {name}")
    print("=" * 68)
    print(f"  Attacker benefit B = {B}")
    print(f"  Q vector (by layer {LAYERS}):")
    q_strs = [f"{v:.3f}" if not np.isnan(v) else "  nan" for v in q]
    print(f"    [{', '.join(q_strs)}]")
    print(f"  Defined layers: {n_defined}/{len(q)}")
    if n_defined > 0:
        defined = q[~np.isnan(q)]
        print(f"  Defined-entries stats: min={defined.min():.3f}  "
              f"max={defined.max():.3f}  std={defined.std():.3f}")
    print(f"  Regime classification: {regime.upper()}")


def print_sweep_table(sweep_results):
    print(f"\n  Jitter sensitivity:")
    print(f"  {'epsilon':>8} | {'utility':>10} | {'entropy':>8} | "
          f"{'max p*':>8} | {'argmax layer':>13} | status")
    print("  " + "-" * 78)
    for row in sweep_results:
        eps = row["epsilon"]
        util = row["utility"]
        p = row["p_star"]
        if p is None:
            print(f"  {eps:>8.2f} | {'FAILED':>10} | "
                  f"{'-':>8} | {'-':>8} | {'-':>13} | {row['status']}")
            continue
        entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
        max_p = p.max()
        argmax_layer = LAYERS[int(np.argmax(p))]
        print(f"  {eps:>8.2f} | {util:>10.4f} | {entropy:>8.4f} | "
              f"{max_p:>8.4f} | {argmax_layer:>13d} | {row['status']}")


def print_policy_detail(sweep_results, layers):
    raw = sweep_results[0]
    if raw["p_star"] is None:
        print(f"\n  Raw policy: UNAVAILABLE ({raw['status']})")
        return
    print(f"\n  Raw policy (pure LP, epsilon=0, lambda=0) by layer:")
    for layer, prob in zip(layers, raw["p_star"]):
        bar = "#" * int(prob * 40)
        print(f"    Layer {layer:2d}: p = {prob:.4f}  {bar}")


def print_lambda_sweep_table(lam_results, layers):
    print(f"\n  Regularisation sweep (L2 penalty -lambda*||p||^2):")
    print(f"  {'lambda':>8} | {'utility':>10} | {'entropy':>8} | "
          f"{'max p*':>8} | {'argmax layer':>13} | status")
    print("  " + "-" * 78)
    for row in lam_results:
        lam = row["lambda"]
        util = row["utility"]
        p = row["p_star"]
        if p is None:
            print(f"  {lam:>8.2f} | {'FAILED':>10} | "
                  f"{'-':>8} | {'-':>8} | {'-':>13} | {row['status']}")
            continue
        entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
        max_p = p.max()
        argmax_layer = layers[int(np.argmax(p))]
        print(f"  {lam:>8.2f} | {util:>10.4f} | {entropy:>8.4f} | "
              f"{max_p:>8.4f} | {argmax_layer:>13d} | {row['status']}")


def print_lambda_policy_detail(lam_results, layers):
    print(f"\n  Regularised policies by layer (each row = one lambda):")
    header = "lambda".rjust(8) + " | " + " ".join(f"L{l:02d}" for l in layers)
    print("  " + header)
    print("  " + "-" * len(header))
    for row in lam_results:
        if row["p_star"] is None:
            continue
        probs_str = " ".join(f"{v:.2f}" for v in row["p_star"])
        print(f"  {row['lambda']:>8.2f} | {probs_str}")


def compare_policies(p1, p2, name1, name2):
    if p1 is None or p2 is None:
        return None
    l1 = np.sum(np.abs(p1 - p2))
    l2 = np.linalg.norm(p1 - p2)
    argmax_same = bool(np.argmax(p1) == np.argmax(p2))
    return {
        "pair": f"{name1}  vs  {name2}",
        "L1": l1,
        "L2": l2,
        "argmax_same_layer": argmax_same,
        "argmax_1": LAYERS[int(np.argmax(p1))],
        "argmax_2": LAYERS[int(np.argmax(p2))],
    }


def export_results(all_results, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)

    # Jitter sweep export (pure LP)
    rows = []
    for regime_name, data in all_results.items():
        for row in data["sweep"]:
            p = row["p_star"]
            rows.append({
                "regime": regime_name,
                "epsilon": row["epsilon"],
                "status": row["status"],
                "utility": row["utility"],
                **{f"p_layer_{l}": (p[i] if p is not None else np.nan)
                   for i, l in enumerate(LAYERS)},
            })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "stackelberg_jitter_sweep.csv")
    df.to_csv(path, index=False)
    print(f"\n  Jitter sweep exported: {path}")

    # Lambda sweep export (regularised QP)
    rows = []
    for regime_name, data in all_results.items():
        for row in data["lambda_sweep"]:
            p = row["p_star"]
            rows.append({
                "regime": regime_name,
                "lambda": row["lambda"],
                "status": row["status"],
                "utility": row["utility"],
                **{f"p_layer_{l}": (p[i] if p is not None else np.nan)
                   for i, l in enumerate(LAYERS)},
            })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "stackelberg_lambda_sweep.csv")
    df.to_csv(path, index=False)
    print(f"  Lambda sweep exported: {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 68)
    print("  DUAL-TRACK STACKELBERG AUDIT POLICY ANALYSIS")
    print("  Single-attacker-type framing (m=1)")
    print("=" * 68)
    print(f"  Defender reward R_D = {R_D}")
    print(f"  Audit cost per layer c = {AUDIT_COST}")
    print(f"  Layers tested: {LAYERS}")
    print(f"  Jitter levels: {JITTER_LEVELS}")
    print(f"  Lambda levels (L2 reg.): {LAMBDA_LEVELS}")

    audit_cost_vec = np.full(len(LAYERS), AUDIT_COST)

    regimes = [
        ("Track A (alpha=3.0)",           Q_TRACK_A,           B_TRACK_A),
        ("Track A sensitive (alpha=1.5)", Q_TRACK_A_SENSITIVE, B_TRACK_A),
        ("Track B (alpha=3.0)",           Q_TRACK_B,           B_TRACK_B),
        ("Clean baseline",                Q_CLEAN,             0.0),
    ]

    all_results = {}

    for name, q, B in regimes:
        print_regime_header(name, q, B)
        q_clean = sanitise_for_solver(q)

        # Pure LP jitter sweep
        sweep = jitter_sweep(q_clean, audit_cost_vec, R_D,
                             seed_base=abs(hash(name)) & 0xFFFF)
        print_sweep_table(sweep)
        print_policy_detail(sweep, LAYERS)

        # Regularised QP lambda sweep on raw q
        lam_sweep = lambda_sweep(q_clean, audit_cost_vec, R_D)
        print_lambda_sweep_table(lam_sweep, LAYERS)
        print_lambda_policy_detail(lam_sweep, LAYERS)

        all_results[name] = {
            "sweep": sweep,
            "lambda_sweep": lam_sweep,
            "q_raw": q,
            "regime": classify_q_regime(q),
        }

    print("\n" + "=" * 68)
    print("  CROSS-REGIME POLICY COMPARISON (pure LP, epsilon=0, lambda=0)")
    print("=" * 68)
    pairs = [
        ("Track A (alpha=3.0)",           "Track B (alpha=3.0)"),
        ("Track A (alpha=3.0)",           "Track A sensitive (alpha=1.5)"),
        ("Track A sensitive (alpha=1.5)", "Track B (alpha=3.0)"),
        ("Track A (alpha=3.0)",           "Clean baseline"),
        ("Track B (alpha=3.0)",           "Clean baseline"),
    ]
    for n1, n2 in pairs:
        p1 = all_results[n1]["sweep"][0]["p_star"]
        p2 = all_results[n2]["sweep"][0]["p_star"]
        cmp = compare_policies(p1, p2, n1, n2)
        if cmp is None:
            print(f"\n  {n1}  vs  {n2}:  skipped (missing policy)")
            continue
        print(f"\n  {cmp['pair']}")
        print(f"    L1 distance       : {cmp['L1']:.4f}")
        print(f"    L2 distance       : {cmp['L2']:.4f}")
        print(f"    Peak layer match  : {cmp['argmax_same_layer']}  "
              f"(layer {cmp['argmax_1']} vs {cmp['argmax_2']})")

    export_results(all_results)

    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    p_A = all_results["Track A (alpha=3.0)"]["sweep"][0]["p_star"]
    p_B = all_results["Track B (alpha=3.0)"]["sweep"][0]["p_star"]
    if p_A is not None and p_B is not None:
        shift = np.linalg.norm(p_A - p_B)
        print(f"  Policy L2 shift (pure LP)  ||p*_A - p*_B||  =  {shift:.4f}")
        if shift < 0.05:
            print("  -> Allocations are STATIC: defender's optimal audit strategy is")
            print("     largely invariant to whether the backdoor was injected de novo")
            print("     or amplified from latent bias.")
        else:
            print("  -> Allocations SHIFT: de novo vs amplified backdoors demand")
            print("     measurably different audit allocations.")
    print("=" * 68)


if __name__ == "__main__":
    main()