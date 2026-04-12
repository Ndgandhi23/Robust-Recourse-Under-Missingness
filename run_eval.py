"""
Systematic batch evaluation over all denied test individuals with missing features.

Runs two conditions:
  ours     : epsilon=0.001, rho_coverage=0.90  (Rashomon + imputation robust)
  baseline : epsilon=0.0,   rho_coverage=0.90  (no Rashomon robustness)

rho is calibrated per-person per-mask from the MICE draws so that the
optimization ellipsoid covers RHO_OPT_COVERAGE of draws, and the stress-test
ellipsoid covers RHO_STRESS_COVERAGE of draws.

Metrics collected per individual:
  nominal validity, model retrain validity,
  awp sanity (rho at 90th pct), awp stress (rho at 99th pct), lof plausibility

Prints per-individual results and aggregate summary table.
"""

import time
import numpy as np

from data        import load_diabetes, train_test_split, MUTABLE_COLS
from model       import train, compute_hessian, predict
from imputer     import fit_mice, calibrate_rho
from beam_search import beam_search
from evaluate    import (
    nominal_validity, model_retrain_validity,
    awp_validity, lof_plausibility, l2_proximity,
)

# ── hyperparameters ───────────────────────────────────────────────────────────
EPSILON             = 0.001
EPSILON_EVAL        = 0.01
RHO_OPT_COVERAGE    = 0.90   # ellipsoid covers 90% of MICE draws during optimization
RHO_STRESS_COVERAGE = 0.99   # ellipsoid covers 99% of MICE draws for stress test
KAPPA_VAL           = 0.5
DELTA_MAX           = 2.0
BEAM_WIDTH          = 3
K_MAX               = 3
K_MICE              = 100
N_MODELS            = 50

# ── setup ─────────────────────────────────────────────────────────────────────
X, Xi, y, Phi, names, col_means, col_stds = load_diabetes()
train_idx, test_idx = train_test_split(len(y))

X_train,  Xi_train  = X[train_idx],  Xi[train_idx]
Phi_train, y_train  = Phi[train_idx], y[train_idx]

theta_hat, _   = train(Phi_train, y_train)
hessian_matrix = compute_hessian(Phi_train, theta_hat)
mice_imputer   = fit_mice(X_train, Xi_train)

kappa  = np.ones(len(X[0])) * KAPPA_VAL
J_edit = MUTABLE_COLS

denied_all = [
    i for i in test_idx
    if predict(Phi[[i]], theta_hat)[0] == -1 and Xi[i].sum() > 0
]
print(f"denied test individuals with missing features: {len(denied_all)}\n")


def run_condition(denied_ids, epsilon, label, k_max=K_MAX, rho_override=None):
    results = []
    for i, idx in enumerate(denied_ids):
        x0, xi0 = X[idx], Xi[idx]
        x_min   = x0 - DELTA_MAX
        x_max   = x0 + DELTA_MAX

        t0 = time.time()
        r, delta, cost, _, mu, Sigma, draws, rho_opt = beam_search(
            x0, xi0, theta_hat, hessian_matrix, mice_imputer,
            epsilon=epsilon, rho_coverage=RHO_OPT_COVERAGE, kappa=kappa, J_edit=J_edit,
            tau=0.0, beam_width=BEAM_WIDTH, K_max=k_max, K_mice=K_MICE,
            x_min=x_min, x_max=x_max,
            verbose=False, rho_override=rho_override,
        )
        elapsed = time.time() - t0

        if r is None:
            results.append({"idx": idx, "feasible": False, "time": elapsed})
            print(f"  [{label}] person {idx}: infeasible")
            continue

        rho_stress = calibrate_rho(draws, mu, Sigma, RHO_STRESS_COVERAGE)

        nom  = nominal_validity(theta_hat, x0, delta, xi0, r, mu)
        rr   = model_retrain_validity(
            X_train, Xi_train, y_train,
            x0, delta, xi0, r, mu, n_models=N_MODELS,
        )
        awp, lb = awp_validity(
            theta_hat, hessian_matrix,
            x0, delta, xi0, r, mu, Sigma,
            epsilon_eval=EPSILON, rho_eval=rho_opt,
        )
        awp_st, lb_st = awp_validity(
            theta_hat, hessian_matrix,
            x0, delta, xi0, r, mu, Sigma,
            epsilon_eval=EPSILON_EVAL, rho_eval=rho_stress,
        )
        lof = lof_plausibility(X_train, Xi_train, mice_imputer, x0, delta, xi0, r, mu)
        l2  = l2_proximity(x0, delta, xi0, r, mu)

        results.append({
            "idx": idx, "feasible": True, "cost": cost,
            "nom": nom, "rr": rr,
            "awp": awp, "lb": lb,
            "awp_st": awp_st, "lb_st": lb_st,
            "rho_opt": rho_opt, "rho_stress": rho_stress,
            "lof": lof, "l2": l2, "time": elapsed,
        })
        print(f"  [{label}] person {idx:3d}  cost={cost:.3f}  "
              f"nom={nom}  rr={rr:.2f}  "
              f"awp={awp}(rho={rho_opt:.2f})  "
              f"stress_margin={lb_st:.4f}(rho={rho_stress:.2f})  lof={lof:.3f}  l2={l2:.3f}")

    return results


# ── run all conditions ────────────────────────────────────────────────────────
print("=" * 70)
print("OURS  (epsilon=0.001, rho calibrated at 90th pct)")
print("=" * 70)
ours_results = run_condition(denied_all, epsilon=EPSILON, label="ours")

print()
print("=" * 70)
print("BASELINE  (epsilon=0, rho calibrated at 90th pct)")
print("=" * 70)
base_results = run_condition(denied_all, epsilon=0.0, label="base")

print()
print("=" * 70)
print("NO-ROBUST  (epsilon=0, rho=0 — point-imputation nominal recourse)")
print("=" * 70)
norobust_results = run_condition(denied_all, epsilon=0.0, label="no-robust",
                                 rho_override=0.0)

print()
print("=" * 70)
print("NO-REVEAL  (epsilon=0.001, rho at 90th pct, K_max=0 — edits only)")
print("=" * 70)
noreveal_results = run_condition(denied_all, epsilon=EPSILON, label="no-reveal",
                                 k_max=0)


# ── aggregate summary ─────────────────────────────────────────────────────────
def summarize(results, label):
    feas     = [r for r in results if r["feasible"]]
    n        = len(results)
    nf       = len(feas)
    time_vals = [r["time"] for r in results]
    if nf == 0:
        print(f"{label:12s}  feasible=0/{n}  "
              f"time={np.mean(time_vals):.1f}s/person")
        return
    cost_vals   = [r["cost"]       for r in feas]
    nom_vals    = [r["nom"]        for r in feas]
    rr_vals     = [r["rr"]         for r in feas]
    awp_vals    = [r["awp"]        for r in feas]
    lb_st_vals  = [r["lb_st"]      for r in feas]
    lof_vals    = [r["lof"]        for r in feas]
    l2_vals     = [r["l2"]         for r in feas]
    rho_o_vals  = [r["rho_opt"]    for r in feas]
    rho_s_vals  = [r["rho_stress"] for r in feas]
    print(
        f"{label:12s}  "
        f"feasible={nf}/{n}  "
        f"cost={np.mean(cost_vals):.3f} (min={np.min(cost_vals):.3f})  "
        f"nominal={np.mean(nom_vals):.3f}  "
        f"retrain={np.mean(rr_vals):.3f} (min={np.min(rr_vals):.3f})  "
        f"awp={np.mean(awp_vals):.3f} ({sum(awp_vals)}/{nf}, rho={np.mean(rho_o_vals):.2f})  "
        f"stress_margin={np.mean(lb_st_vals):.4f} (min={np.min(lb_st_vals):.4f}, rho={np.mean(rho_s_vals):.2f})  "
        f"lof={np.mean(lof_vals):.3f}  "
        f"l2={np.mean(l2_vals):.3f}  "
        f"time={np.mean(time_vals):.1f}s/person"
    )


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
summarize(ours_results,     "ours")
summarize(base_results,     "baseline")
summarize(norobust_results, "no-robust")
summarize(noreveal_results, "no-reveal")
