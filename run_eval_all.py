"""
Batch evaluation across all datasets.

Usage:
    python run_eval_all.py              # run both datasets
    python run_eval_all.py diabetes     # run diabetes only
"""

import sys
import time
import argparse
import multiprocessing as mp
from functools import partial
import numpy as np

from pipeline import (
    train, compute_hessian, predict,
    fit_mice,
    beam_search,
    nominal_validity, model_retrain_validity, train_bootstrap_models,
    awp_validity, lof_plausibility, fit_lof, l2_proximity,
)

# ── hyperparameters (shared across datasets) ─────────────────────────────────
EPSILON             = 0.001
RHO_OPT_COVERAGE    = 0.90
KAPPA_VAL           = 0.5
DELTA_MAX           = 2.0
BEAM_WIDTH          = 3
K_MAX               = 3
K_MICE              = 100
N_MODELS            = 50


# ── dataset registry ─────────────────────────────────────────────────────────
DATASETS = {}

def _load_diabetes():
    from pipeline import load_diabetes, train_test_split, DIABETES_MUTABLE
    X, Xi, y, Phi, names, col_means, col_stds = load_diabetes()
    train_idx, test_idx = train_test_split(len(y))
    return X, Xi, y, Phi, names, col_means, col_stds, train_idx, test_idx, DIABETES_MUTABLE

DATASETS["diabetes"] = _load_diabetes


# ── evaluation logic ─────────────────────────────────────────────────────────
def _eval_one_person(idx, X, Xi, theta_hat, hessian_matrix, mice_imputer,
                     bootstrap_models, lof_model, kappa, J_edit,
                     epsilon, k_max, rho_override):
    x0, xi0 = X[idx], Xi[idx]
    x_min = x0 - DELTA_MAX
    x_max = x0 + DELTA_MAX

    t0 = time.time()
    r, delta, cost, _, mu, Sigma, _, rho_opt = beam_search(
        x0, xi0, theta_hat, hessian_matrix, mice_imputer,
        epsilon=epsilon, rho_coverage=RHO_OPT_COVERAGE, kappa=kappa, J_edit=J_edit,
        tau=0.0, beam_width=BEAM_WIDTH, K_max=k_max, K_mice=K_MICE,
        x_min=x_min, x_max=x_max,
        verbose=False, rho_override=rho_override,
    )
    elapsed = time.time() - t0

    if r is None:
        return {"idx": idx, "feasible": False, "time": elapsed}

    nom = nominal_validity(theta_hat, x0, delta, xi0, r, mu)
    rr  = model_retrain_validity(x0, delta, xi0, r, mu, bootstrap_models)
    awp, lb = awp_validity(
        theta_hat, hessian_matrix,
        x0, delta, xi0, r, mu, Sigma,
        epsilon_eval=EPSILON, rho_eval=rho_opt,
    )
    lof = lof_plausibility(x0, delta, xi0, r, mu, lof_model)
    l2  = l2_proximity(x0, delta, xi0, r, mu)

    return {
        "idx": idx, "feasible": True, "cost": cost,
        "nom": nom, "rr": rr,
        "awp": awp, "lb": lb,
        "lof": lof, "l2": l2, "time": elapsed,
    }


def run_condition(X, Xi, Phi, theta_hat, hessian_matrix, mice_imputer,
                  bootstrap_models, lof_model, kappa, J_edit,
                  denied_ids, epsilon, label, k_max=K_MAX, rho_override=None,
                  n_workers=1):
    worker = partial(
        _eval_one_person,
        X=X, Xi=Xi, theta_hat=theta_hat, hessian_matrix=hessian_matrix,
        mice_imputer=mice_imputer, bootstrap_models=bootstrap_models,
        lof_model=lof_model, kappa=kappa, J_edit=J_edit,
        epsilon=epsilon, k_max=k_max, rho_override=rho_override,
    )

    if n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(worker, denied_ids)
    else:
        results = [worker(idx) for idx in denied_ids]

    for r in results:
        if r["feasible"]:
            print(f"  [{label}] person {r['idx']:5d}  cost={r['cost']:.3f}  "
                  f"nom={r['nom']}  rr={r['rr']:.2f}  "
                  f"awp={r['awp']}(rho=-)  "
                  f"lof={r['lof']:.3f}  l2={r['l2']:.3f}")
        else:
            print(f"  [{label}] person {r['idx']}: infeasible")

    return results


def summarize(results, label):
    feas      = [r for r in results if r["feasible"]]
    n         = len(results)
    nf        = len(feas)
    time_vals = [r["time"] for r in results]
    if nf == 0:
        print(f"{label:12s}  feasible=0/{n}  "
              f"time={np.mean(time_vals):.1f}s/person")
        return
    cost_vals = [r["cost"] for r in feas]
    nom_vals  = [r["nom"]  for r in feas]
    rr_vals   = [r["rr"]   for r in feas]
    awp_vals  = [r["awp"]  for r in feas]
    lof_vals  = [r["lof"]  for r in feas]
    l2_vals   = [r["l2"]   for r in feas]
    print(
        f"{label:12s}  "
        f"feasible={nf}/{n}  "
        f"cost={np.mean(cost_vals):.3f} (min={np.min(cost_vals):.3f})  "
        f"nominal={np.mean(nom_vals):.3f}  "
        f"retrain={np.mean(rr_vals):.3f} (min={np.min(rr_vals):.3f})  "
        f"awp={np.mean(awp_vals):.3f} ({sum(awp_vals)}/{nf})  "
        f"lof={np.mean(lof_vals):.3f}  "
        f"l2={np.mean(l2_vals):.3f}  "
        f"time={np.mean(time_vals):.1f}s/person"
    )


def run_dataset(name, loader, n_workers=1):
    print()
    print("#" * 70)
    print(f"#  DATASET: {name.upper()}")
    print("#" * 70)

    X, Xi, y, Phi, names, col_means, col_stds, train_idx, test_idx, J_edit = loader()

    X_train,  Xi_train  = X[train_idx],  Xi[train_idx]
    Phi_train, y_train  = Phi[train_idx], y[train_idx]

    theta_hat, _   = train(Phi_train, y_train)
    hessian_matrix = compute_hessian(Phi_train, theta_hat)
    mice_imputer   = fit_mice(X_train, Xi_train)

    kappa  = np.ones(len(X[0])) * KAPPA_VAL

    denied_all = [
        i for i in test_idx
        if predict(Phi[[i]], theta_hat)[0] == -1 and Xi[i].sum() > 0
    ]
    print(f"denied test individuals with missing features: {len(denied_all)}")

    print("precomputing bootstrap models and LOF...", end=" ", flush=True)
    t_pre = time.time()
    bootstrap_models = train_bootstrap_models(X_train, Xi_train, y_train, n_models=N_MODELS)
    lof_model        = fit_lof(X_train, Xi_train, mice_imputer)
    print(f"done ({time.time() - t_pre:.1f}s)\n")

    shared = dict(
        X=X, Xi=Xi, Phi=Phi,
        theta_hat=theta_hat, hessian_matrix=hessian_matrix,
        mice_imputer=mice_imputer,
        bootstrap_models=bootstrap_models, lof_model=lof_model,
        kappa=kappa, J_edit=J_edit,
        denied_ids=denied_all,
    )

    conditions = [
        ("ours",      EPSILON, K_MAX, None),
        ("baseline",  0.0,     K_MAX, None),
        ("no-robust", 0.0,     K_MAX, 0.0),
        ("no-reveal", EPSILON, 0,     None),
    ]

    all_results = {}
    for label, eps, k_max, rho_override in conditions:
        print("=" * 70)
        print(f"{label.upper()}  (epsilon={eps}, k_max={k_max}"
              f"{f', rho_override={rho_override}' if rho_override is not None else ''})")
        print("=" * 70)
        all_results[label] = run_condition(
            **shared, epsilon=eps, label=label,
            k_max=k_max, rho_override=rho_override,
            n_workers=n_workers,
        )
        print()

    print("=" * 70)
    print(f"SUMMARY — {name.upper()}")
    print("=" * 70)
    for label, _, _, _ in conditions:
        summarize(all_results[label], label)

    return all_results


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation across datasets")
    parser.add_argument("datasets", nargs="*", default=list(DATASETS.keys()),
                        help="Datasets to evaluate (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for per-person evaluation")
    args = parser.parse_args()

    for name in args.datasets:
        if name not in DATASETS:
            print(f"unknown dataset: {name}")
            print(f"available: {', '.join(DATASETS.keys())}")
            sys.exit(1)

    t_total = time.time()
    for name in args.datasets:
        run_dataset(name, DATASETS[name], n_workers=args.workers)

    print(f"\ntotal wall time: {time.time() - t_total:.0f}s")
