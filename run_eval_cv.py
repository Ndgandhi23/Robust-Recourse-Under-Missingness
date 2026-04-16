"""
4-fold stratified cross-validated evaluation with hyperparameter tuning.

Per fold:
  1. Split fold training data 80/20 into inner-train / validation
  2. Train temporary model on inner-train
  3. Grid search (epsilon, rho_coverage) on validation denied individuals
     — criterion: maximize feasibility, then robustness, then minimize cost
  4. Retrain final model on full fold training data
  5. Evaluate all conditions on held-out test split
  6. Compute retrain / ellipsoid / AWP robustness curves over εtarget sweep

Usage:
    python run_eval_cv.py
"""

import sys
import time
import argparse
import multiprocessing as mp
import pickle
from functools import partial
import numpy as np

from pipeline import (
    load_diabetes, DIABETES_MUTABLE,
    build_phi, train, compute_hessian, predict, accuracy,
    fit_mice, stratified_kfold, stratified_train_val_split,
    beam_search,
    nominal_validity, model_retrain_validity, train_bootstrap_models,
    awp_validity, lof_plausibility, fit_lof, l2_proximity,
    rashomon_distance, build_ellipsoid_evaluator, ensemble_robustness,
)

# ── fixed hyperparameters ───────────────────────────────────────────────
N_FOLDS             = 4
KAPPA_VAL           = 0.5
DELTA_MAX           = 2.0
BEAM_WIDTH          = 3
K_MAX               = 3
K_MICE              = 100
N_MODELS            = 50

# tuning grid
EPSILON_GRID        = [0.0, 0.0005, 0.001, 0.005, 0.01]
RHO_COV_GRID        = [0.85, 0.90, 0.95]
TUNE_REF_ETARGET    = 0.005   # εtarget for the evaluator used during tuning
MAX_TUNE_DENIED     = 15      # cap validation individuals for speed

# εtarget sweep for robustness curves
ETARGET_GRID        = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

COND_LABELS         = ["ours", "baseline", "no-robust", "no-reveal"]


# ── per-condition evaluation ────────────────────────────────────────────

def _eval_one_person_cv(idx, X, Xi, theta_hat, hessian_matrix, mice_imputer,
                        bootstrap_models, lof_model, kappa, J_edit,
                        epsilon, rho_coverage, k_max, rho_override):
    x0, xi0 = X[idx], Xi[idx]
    x_min = x0 - DELTA_MAX
    x_max = x0 + DELTA_MAX

    t0 = time.time()
    r, delta, cost, _, mu, Sigma, _, rho_opt = beam_search(
        x0, xi0, theta_hat, hessian_matrix, mice_imputer,
        epsilon=epsilon, rho_coverage=rho_coverage, kappa=kappa, J_edit=J_edit,
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
        epsilon_eval=epsilon, rho_eval=rho_opt,
    )
    lof = lof_plausibility(x0, delta, xi0, r, mu, lof_model)
    l2  = l2_proximity(x0, delta, xi0, r, mu)

    return {
        "idx": idx, "feasible": True, "cost": cost,
        "nom": nom, "rr": rr,
        "awp": awp, "lb": lb,
        "lof": lof, "l2": l2, "time": elapsed,
        "delta": delta, "r": r, "mu": mu,
        "Sigma": Sigma, "rho_opt": rho_opt,
    }


def run_condition(X, Xi, Phi, theta_hat, hessian_matrix, mice_imputer,
                  bootstrap_models, lof_model, kappa, J_edit,
                  denied_ids, epsilon, rho_coverage, label,
                  k_max=K_MAX, rho_override=None, n_workers=1):
    worker = partial(
        _eval_one_person_cv,
        X=X, Xi=Xi, theta_hat=theta_hat, hessian_matrix=hessian_matrix,
        mice_imputer=mice_imputer, bootstrap_models=bootstrap_models,
        lof_model=lof_model, kappa=kappa, J_edit=J_edit,
        epsilon=epsilon, rho_coverage=rho_coverage,
        k_max=k_max, rho_override=rho_override,
    )

    if n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(worker, denied_ids)
    else:
        results = [worker(idx) for idx in denied_ids]

    return results


def summarize_condition(results):
    feas = [r for r in results if r["feasible"]]
    n, nf = len(results), len(feas)
    if nf == 0:
        return {"n": n, "nf": 0, "feas_rate": 0.0}
    return {
        "n": n, "nf": nf,
        "feas_rate": nf / n,
        "cost": np.mean([r["cost"] for r in feas]),
        "nom":  np.mean([r["nom"]  for r in feas]),
        "rr":   np.mean([r["rr"]   for r in feas]),
        "awp":  np.mean([r["awp"]  for r in feas]),
        "lof":  np.mean([r["lof"]  for r in feas]),
        "l2":   np.mean([r["l2"]   for r in feas]),
        "time": np.mean([r["time"] for r in results]),
    }


_PERSON_KEYS = ("idx", "feasible", "cost", "nom", "rr", "awp", "lb", "lof", "l2", "time")


def _slim_results(raw):
    """Keep only scalar per-person metrics (drop heavy arrays like delta/Sigma)."""
    return [
        {k: float(r[k]) if isinstance(r.get(k), (np.floating, np.integer)) else r[k]
         for k in _PERSON_KEYS if k in r}
        for r in raw
    ]


# ── hyperparameter tuning ──────────────────────────────────────────────

def _tune_one_person(idx, X, Xi, theta_tmp, hessian_tmp, mice_tmp,
                     kappa, J_edit, eps, rho_c, ref_models):
    x0, xi0 = X[idx], Xi[idx]
    r, delta, cost, _, mu, Sigma, _, rho_opt = beam_search(
        x0, xi0, theta_tmp, hessian_tmp, mice_tmp,
        epsilon=eps, rho_coverage=rho_c, kappa=kappa, J_edit=J_edit,
        tau=0.0, beam_width=BEAM_WIDTH, K_max=K_MAX, K_mice=K_MICE,
        x_min=x0 - DELTA_MAX, x_max=x0 + DELTA_MAX, verbose=False,
    )
    if r is None:
        return (False, 0.0, False)
    robust = ensemble_robustness(x0, delta, xi0, r, mu, ref_models)
    return (True, cost, robust)


def tune_hyperparams(X, Xi, y, Phi, inner_trn_idx, inner_val_idx, J_edit,
                     fold_num, n_workers=1):
    """
    Grid search (epsilon, rho_coverage) on validation set.

    1. Train temp model on inner_trn_idx
    2. Build ellipsoid evaluator at TUNE_REF_ETARGET
    3. For each grid point, run beam search on val denied individuals
    4. Pick (epsilon, rho_coverage) maximizing (feasibility, robustness, -cost)
    """
    # temporary model on inner train
    X_trn, Xi_trn, y_trn = X[inner_trn_idx], Xi[inner_trn_idx], y[inner_trn_idx]
    Phi_trn = build_phi(X_trn, Xi_trn)

    theta_tmp, _ = train(Phi_trn, y_trn)
    hessian_tmp  = compute_hessian(Phi_trn, theta_tmp)
    mice_tmp     = fit_mice(X_trn, Xi_trn)
    kappa        = np.ones(X.shape[1]) * KAPPA_VAL

    # validation denied individuals with missing features
    denied_val = [
        i for i in inner_val_idx
        if predict(Phi[[i]], theta_tmp)[0] == -1 and Xi[i].sum() > 0
    ]
    if not denied_val:
        print("    tuning: no denied+missing in val — using defaults")
        return 0.001, 0.90

    denied_val = denied_val[:MAX_TUNE_DENIED]
    print(f"    tuning: {len(denied_val)} val individuals, "
          f"{len(EPSILON_GRID)}x{len(RHO_COV_GRID)} grid")

    # reference evaluator for robustness scoring
    ref_models = build_ellipsoid_evaluator(theta_tmp, hessian_tmp,
                                           TUNE_REF_ETARGET, n_models=50, seed=fold_num)

    best_score = (-1, -1, float("inf"))
    best_eps, best_rho = 0.001, 0.90

    pool = mp.Pool(processes=n_workers) if n_workers > 1 else None
    try:
        for eps in EPSILON_GRID:
            for rho_c in RHO_COV_GRID:
                worker = partial(
                    _tune_one_person,
                    X=X, Xi=Xi, theta_tmp=theta_tmp, hessian_tmp=hessian_tmp,
                    mice_tmp=mice_tmp, kappa=kappa, J_edit=J_edit,
                    eps=eps, rho_c=rho_c, ref_models=ref_models,
                )

                if pool is not None:
                    person_results = pool.map(worker, denied_val)
                else:
                    person_results = [worker(idx) for idx in denied_val]

                n_feas, n_robust, total_cost = 0, 0, 0.0
                for feasible, cost, robust in person_results:
                    if feasible:
                        n_feas += 1
                        total_cost += cost
                        if robust:
                            n_robust += 1

                n = len(denied_val)
                score = (n_feas / n, n_robust / n, -total_cost / max(n_feas, 1))

                if score > best_score:
                    best_score = score
                    best_eps, best_rho = eps, rho_c
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"    tuning: best eps={best_eps:.4f}  rho={best_rho:.2f}  "
          f"(feas={best_score[0]:.2f} rob={best_score[1]:.2f})")
    return best_eps, best_rho


# ── εtarget robustness curves ──────────────────────────────────────────

def retrain_curve(raw_results, X, Xi, bootstrap_models, theta_hat,
                  hessian, etarget_grid):
    feas = [r for r in raw_results if r["feasible"]]
    if not feas:
        return [{"et": et, "n_models": 0, "robustness": None} for et in etarget_grid]

    distances = [rashomon_distance(m, theta_hat, hessian) for m in bootstrap_models]
    curve = []
    for et in etarget_grid:
        filtered = [m for m, d in zip(bootstrap_models, distances) if d <= et]
        if not filtered:
            curve.append({"et": et, "n_models": 0, "robustness": None})
            continue
        n_robust = sum(
            1 for r in feas
            if ensemble_robustness(X[r["idx"]], r["delta"], Xi[r["idx"]],
                                   r["r"], r["mu"], filtered)
        )
        curve.append({"et": et, "n_models": len(filtered),
                      "robustness": n_robust / len(feas)})
    return curve


def ellipsoid_curve(raw_results, X, Xi, theta_hat, hessian, etarget_grid,
                    n_models=50, seed=0):
    feas = [r for r in raw_results if r["feasible"]]
    if not feas:
        return [{"et": et, "robustness": None} for et in etarget_grid]

    curve = []
    for et in etarget_grid:
        models = build_ellipsoid_evaluator(theta_hat, hessian, et,
                                           n_models=n_models, seed=seed)
        n_robust = sum(
            1 for r in feas
            if ensemble_robustness(X[r["idx"]], r["delta"], Xi[r["idx"]],
                                   r["r"], r["mu"], models)
        )
        curve.append({"et": et, "robustness": n_robust / len(feas)})
    return curve


def awp_curve(raw_results, X, Xi, theta_hat, hessian, etarget_grid):
    feas = [r for r in raw_results if r["feasible"]]
    if not feas:
        return [{"et": et, "robustness": None} for et in etarget_grid]

    curve = []
    for et in etarget_grid:
        n_pass = 0
        for r in feas:
            passed, _ = awp_validity(
                theta_hat, hessian,
                X[r["idx"]], r["delta"], Xi[r["idx"]], r["r"],
                r["mu"], r["Sigma"],
                epsilon_eval=et, rho_eval=r["rho_opt"] if r["rho_opt"] else 0.0,
            )
            if passed:
                n_pass += 1
        curve.append({"et": et, "robustness": n_pass / len(feas)})
    return curve


# ── single fold ─────────────────────────────────────────────────────────

def run_fold(fold_num, X, Xi, y, Phi, train_idx, test_idx, J_edit, n_workers=1):
    print(f"\n{'='*70}")
    print(f"  FOLD {fold_num + 1}")
    print(f"{'='*70}")

    # ── inner train/val split for tuning ────────────────────────────────
    y_fold = y[train_idx]
    inner_trn_local, inner_val_local = stratified_train_val_split(
        y_fold, val_frac=0.2, seed=fold_num,
    )
    inner_trn_idx = train_idx[inner_trn_local]
    inner_val_idx = train_idx[inner_val_local]

    print(f"  inner train={len(inner_trn_idx)}  val={len(inner_val_idx)}  test={len(test_idx)}")

    t_tune = time.time()
    best_eps, best_rho = tune_hyperparams(
        X, Xi, y, Phi, inner_trn_idx, inner_val_idx, J_edit, fold_num,
        n_workers=n_workers,
    )
    print(f"    tuning time: {time.time() - t_tune:.1f}s")

    # ── retrain final model on full fold training data ──────────────────
    X_train, Xi_train = X[train_idx], Xi[train_idx]
    Phi_train, y_train = Phi[train_idx], y[train_idx]
    Phi_test, y_test = Phi[test_idx], y[test_idx]

    theta_hat, _ = train(Phi_train, y_train)
    hessian      = compute_hessian(Phi_train, theta_hat)
    mice_imputer = fit_mice(X_train, Xi_train)
    kappa        = np.ones(X.shape[1]) * KAPPA_VAL

    train_acc = accuracy(Phi_train, theta_hat, y_train)
    test_acc  = accuracy(Phi_test,  theta_hat, y_test)

    denied = [
        i for i in test_idx
        if predict(Phi[[i]], theta_hat)[0] == -1 and Xi[i].sum() > 0
    ]

    print(f"  final model: acc={train_acc:.3f}/{test_acc:.3f}  denied+missing={len(denied)}")

    if not denied:
        print("    (no denied individuals — skipping)")
        return None

    t0 = time.time()
    bootstrap_models = train_bootstrap_models(X_train, Xi_train, y_train, n_models=N_MODELS)
    lof_model        = fit_lof(X_train, Xi_train, mice_imputer)
    dists = [rashomon_distance(m, theta_hat, hessian) for m in bootstrap_models]
    print(f"    bootstrap + LOF: {time.time() - t0:.1f}s  "
          f"Rashomon dist: min={min(dists):.4f} med={np.median(dists):.4f} max={max(dists):.4f}")

    # ── conditions (using tuned params) ─────────────────────────────────
    conditions = [
        # (label,       epsilon,   rho_cov,   k_max, rho_override)
        ("ours",        best_eps,  best_rho,  K_MAX, None),
        ("baseline",    0.0,       best_rho,  K_MAX, None),
        ("no-robust",   0.0,       best_rho,  K_MAX, 0.0),
        ("no-reveal",   best_eps,  best_rho,  0,     None),
    ]

    shared = dict(
        X=X, Xi=Xi, Phi=Phi,
        theta_hat=theta_hat, hessian_matrix=hessian,
        mice_imputer=mice_imputer,
        bootstrap_models=bootstrap_models, lof_model=lof_model,
        kappa=kappa, J_edit=J_edit,
        denied_ids=denied,
    )

    fold_summaries = {}
    fold_raw       = {}
    fold_persons   = {}
    fold_curves    = {}

    for label, eps, rho_c, k_max, rho_override in conditions:
        t0 = time.time()
        raw = run_condition(**shared, epsilon=eps, rho_coverage=rho_c, label=label,
                            k_max=k_max, rho_override=rho_override,
                            n_workers=n_workers)
        summary = summarize_condition(raw)
        fold_summaries[label] = summary
        fold_raw[label] = raw
        fold_persons[label] = _slim_results(raw)

        fold_curves[label] = {
            "retrain":   retrain_curve(raw, X, Xi, bootstrap_models,
                                       theta_hat, hessian, ETARGET_GRID),
            "ellipsoid": ellipsoid_curve(raw, X, Xi, theta_hat, hessian,
                                         ETARGET_GRID, seed=fold_num),
            "awp":       awp_curve(raw, X, Xi, theta_hat, hessian, ETARGET_GRID),
        }

        nf = summary["nf"]
        elapsed = time.time() - t0
        if nf > 0:
            print(f"    {label:12s}  feas={nf}/{summary['n']}  "
                  f"rr={summary['rr']:.3f}  awp={summary['awp']:.3f}  "
                  f"l2={summary['l2']:.3f}  ({elapsed:.1f}s)")
        else:
            print(f"    {label:12s}  feas=0/{summary['n']}  ({elapsed:.1f}s)")

    return {
        "summaries": fold_summaries,
        "curves": fold_curves,
        "raw": fold_persons,
        "best_eps": best_eps,
        "best_rho": best_rho,
    }


# ── aggregate across folds ──────────────────────────────────────────────

def _fmt(vals):
    m  = np.mean(vals)
    se = np.std(vals, ddof=0) / np.sqrt(len(vals))
    return f"{m:.3f}±{se:.3f}"


def print_summary(all_folds):
    n_folds = len(all_folds)
    metrics = ["cost", "nom", "rr", "awp", "lof", "l2"]

    print(f"\n{'#' * 80}")
    print(f"#  SUMMARY — DIABETES  ({n_folds} folds, mean ± SE)")
    print(f"{'#' * 80}")

    # ── tuned hyperparameters ───────────────────────────────────────────
    print("\nTuned hyperparameters per fold:")
    for i, f in enumerate(all_folds):
        print(f"  Fold {i+1}: eps={f['best_eps']:.4f}  rho={f['best_rho']:.2f}")

    # ── main table ──────────────────────────────────────────────────────
    cols = ["Condition", "Feasible", "Cost", "Nominal", "Retrain", "AWP", "LOF", "L2"]
    widths = [12, 12, 12, 12, 12, 12, 12, 12]
    header = "  ".join(f"{c:>{w}s}" for c, w in zip(cols, widths))
    print(f"\n{header}")
    print("-" * len(header))

    for cond in COND_LABELS:
        fold_stats = [f["summaries"][cond] for f in all_folds]
        feas_vals = [s["feas_rate"] for s in fold_stats]
        feas_str  = _fmt(feas_vals)

        with_feas = [s for s in fold_stats if s["nf"] > 0]
        if not with_feas:
            row = f"{cond:>12s}  {feas_str:>12s}" + "  ".join(f"{'—':>12s}" for _ in metrics)
            print(row)
            continue

        parts = [f"{cond:>12s}", f"{feas_str:>12s}"]
        for key in metrics:
            parts.append(f"{_fmt([s[key] for s in with_feas]):>12s}")
        print("  ".join(parts))

    # ── retrain robustness curve ────────────────────────────────────────
    print(f"\n{'#' * 80}")
    print(f"#  RETRAIN ROBUSTNESS vs εtarget  ({n_folds} folds, mean ± SE)")
    print(f"{'#' * 80}\n")

    hdr = [f"{'εtarget':>8s}", f"{'#models':>8s}"]
    for c in COND_LABELS: hdr.append(f"{c:>12s}")
    print("  ".join(hdr))
    print("-" * (10 + 10 + 14 * len(COND_LABELS)))

    for i, et in enumerate(ETARGET_GRID):
        nm = np.mean([f["curves"][COND_LABELS[0]]["retrain"][i]["n_models"] for f in all_folds])
        parts = [f"{et:8.3f}", f"{nm:8.1f}"]
        for cond in COND_LABELS:
            vals = [f["curves"][cond]["retrain"][i]["robustness"]
                    for f in all_folds
                    if f["curves"][cond]["retrain"][i]["robustness"] is not None]
            parts.append(f"{_fmt(vals):>12s}" if vals else f"{'—':>12s}")
        print("  ".join(parts))

    # ── ellipsoid robustness curve ──────────────────────────────────────
    print(f"\n{'#' * 80}")
    print(f"#  ELLIPSOID ROBUSTNESS vs εtarget  ({n_folds} folds, mean ± SE)")
    print(f"#  (50 models sampled from Rashomon ellipsoid per εtarget)")
    print(f"{'#' * 80}\n")

    hdr = [f"{'εtarget':>8s}"]
    for c in COND_LABELS: hdr.append(f"{c:>12s}")
    print("  ".join(hdr))
    print("-" * (10 + 14 * len(COND_LABELS)))

    for i, et in enumerate(ETARGET_GRID):
        parts = [f"{et:8.3f}"]
        for cond in COND_LABELS:
            vals = [f["curves"][cond]["ellipsoid"][i]["robustness"]
                    for f in all_folds
                    if f["curves"][cond]["ellipsoid"][i]["robustness"] is not None]
            parts.append(f"{_fmt(vals):>12s}" if vals else f"{'—':>12s}")
        print("  ".join(parts))

    # ── AWP robustness curve ────────────────────────────────────────────
    print(f"\n{'#' * 80}")
    print(f"#  AWP ROBUSTNESS vs εtarget  ({n_folds} folds, mean ± SE)")
    print(f"{'#' * 80}\n")

    hdr = [f"{'εtarget':>8s}"]
    for c in COND_LABELS: hdr.append(f"{c:>12s}")
    print("  ".join(hdr))
    print("-" * (10 + 14 * len(COND_LABELS)))

    for i, et in enumerate(ETARGET_GRID):
        parts = [f"{et:8.3f}"]
        for cond in COND_LABELS:
            vals = [f["curves"][cond]["awp"][i]["robustness"]
                    for f in all_folds
                    if f["curves"][cond]["awp"][i]["robustness"] is not None]
            parts.append(f"{_fmt(vals):>12s}" if vals else f"{'—':>12s}")
        print("  ".join(parts))

    print()


# ── main ────────────────────────────────────────────────────────────────

def print_fold_diagnostics(X, Xi, y, Phi, names, folds):
    """Print per-fold class balance and missingness rates."""
    print(f"\n{'#' * 80}")
    print(f"#  FOLD DIAGNOSTICS")
    print(f"{'#' * 80}\n")

    print(f"  Overall: n={len(y)}  "
          f"not_diabetic(+1)={( y==1).sum()} ({100*(y==1).mean():.1f}%)  "
          f"diabetic(-1)={( y==-1).sum()} ({100*(y==-1).mean():.1f}%)  "
          f"any-missing={100*(Xi.sum(axis=1)>0).mean():.1f}%\n")

    hdr = (f"  {'Fold':>4s}  {'Train':>5s}  {'Test':>5s}  "
           f"{'Trn +1%':>7s}  {'Tst +1%':>7s}  "
           f"{'Trn miss%':>9s}  {'Tst miss%':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for i, (train_idx, test_idx) in enumerate(folds):
        y_trn, y_tst = y[train_idx], y[test_idx]
        print(f"  {i+1:4d}  {len(train_idx):5d}  {len(test_idx):5d}  "
              f"{100*(y_trn==1).mean():6.1f}%  {100*(y_tst==1).mean():6.1f}%  "
              f"{100*(Xi[train_idx].sum(axis=1)>0).mean():8.1f}%  "
              f"{100*(Xi[test_idx].sum(axis=1)>0).mean():8.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="4-fold stratified CV evaluation")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for per-person evaluation")
    args = parser.parse_args()

    print("Loading diabetes dataset...")
    X, Xi, y, Phi, names, col_means, col_stds = load_diabetes()
    J_edit = DIABETES_MUTABLE

    folds = stratified_kfold(y, n_splits=N_FOLDS, seed=42, Xi=Xi)

    print_fold_diagnostics(X, Xi, y, Phi, names, folds)

    t_total = time.time()

    all_folds = []
    for fold_num, (train_idx, test_idx) in enumerate(folds):
        fold_result = run_fold(fold_num, X, Xi, y, Phi, train_idx, test_idx, J_edit,
                               n_workers=args.workers)
        if fold_result is not None:
            all_folds.append(fold_result)

    if not all_folds:
        print("No valid folds.")
        return

    print_summary(all_folds)

    # save results for figures.py
    save_path = "results_cv.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "folds": all_folds,
            "etarget_grid": list(ETARGET_GRID),
            "cond_labels": list(COND_LABELS),
        }, f)
    print(f"Results saved to {save_path}")
    print(f"total wall time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
