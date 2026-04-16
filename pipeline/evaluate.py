import warnings
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .data_utils import build_phi
from .model      import train, score
from .recourse   import compute_A_b, worst_case_lower_bound

def _recourse_point(x0, delta, xi0, r, mu):
    """Post-action feature vector with mu plugged into still-missing entries."""
    x_rec = (x0 + delta).copy()
    xi_c  = xi0 - r
    miss_idx = np.where(xi_c == 1)[0]
    if len(miss_idx) > 0:
        x_rec[miss_idx] = mu
    return x_rec, xi_c


def nominal_validity(theta_hat, x0, delta, xi0, r, mu, tau=0.0):
    """Returns True if the recourse flips the decision under the trained model."""
    x_rec, xi_c = _recourse_point(x0, delta, xi0, r, mu)
    phi_rec     = build_phi(x_rec.reshape(1, -1), xi_c.reshape(1, -1))
    return bool(score(phi_rec, theta_hat)[0] >= tau)


def train_bootstrap_models(X_train, Xi_train, y_train, n_models=50, seed=0):
    """Train bootstrap-resampled models once. Returns list of theta vectors."""
    rng = np.random.RandomState(seed)
    n   = len(y_train)
    models = []
    for _ in range(n_models):
        idx   = rng.choice(n, size=n, replace=True)
        Phi_b = build_phi(X_train[idx], Xi_train[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            theta_b, _ = train(Phi_b, y_train[idx])
        models.append(theta_b)
    return models


def model_retrain_validity(
    x0, delta, xi0, r, mu,
    bootstrap_models, tau=0.0,
):
    """
    Score the recourse point against pre-trained bootstrap models.
    Returns fraction for which the recourse action achieves score >= tau.
    """
    x_rec, xi_c = _recourse_point(x0, delta, xi0, r, mu)
    phi_rec     = build_phi(x_rec.reshape(1, -1), xi_c.reshape(1, -1))

    valid = 0
    for theta_b in bootstrap_models:
        s = score(phi_rec, theta_b)[0]
        if np.isfinite(s) and s >= tau:
            valid += 1

    return valid / len(bootstrap_models)




def awp_validity(
    theta_hat, hessian_matrix,
    x0, delta, xi0, r, mu, Sigma,
    epsilon_eval, rho_eval, tau=0.0,
):
    """
    Joint adversarial worst-case over both the Rashomon ellipsoid (model
    uncertainty) and the imputation ellipsoid (missingness uncertainty).

    Computes the closed-form lower bound from Proposition 2:
        LB = theta_hat^T v
             - sqrt(2*eps) * ||H^{-1/2} v||
             - rho         * ||Sigma^{1/2} A^T theta_hat||
             - sqrt(2*eps) * rho * ||H^{-1/2} A Sigma^{1/2}||
    where v = A @ mu + b.

    Call twice:
      - epsilon_eval = epsilon_opt, rho_eval = rho_opt  -> sanity check,
        should always pass by construction if the solver succeeded.
      - epsilon_eval > epsilon_opt, rho_eval > rho_opt  -> out-of-sample
        stress test; failure here is meaningful.

    Returns (passes: bool, lower_bound: float).
    """
    A, b, _ = compute_A_b(x0, xi0, delta, r)
    lb = worst_case_lower_bound(
        theta_hat, hessian_matrix, A, b, mu, Sigma,
        epsilon_eval, rho_eval,
    )
    return bool(lb >= tau), float(lb)


def fit_lof(X_train, Xi_train, mice_imputer, n_neighbors=20, K=20, seed=0):
    """Fit LOF on MICE-imputed training data once. Returns fitted LOF model."""
    rng   = np.random.RandomState(seed)
    X_nan = X_train.copy().astype(float)
    X_nan[Xi_train == 1] = np.nan

    draws = []
    for _ in range(K):
        mice_imputer.random_state_ = np.random.RandomState(rng.randint(0, 100000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            draws.append(mice_imputer.transform(X_nan))
    X_lof = np.mean(draws, axis=0)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_lof)
    return lof


def lof_plausibility(x0, delta, xi0, r, mu, lof_model):
    """
    Score a recourse point against a pre-fitted LOF model.
    Returns a value >= 1; scores close to 1 indicate the recourse point lies
    in a realistic, high-density region.
    """
    x_rec, _ = _recourse_point(x0, delta, xi0, r, mu)
    lof_score = -lof_model.score_samples(x_rec.reshape(1, -1))[0]
    return float(lof_score)


def l2_proximity(x0, delta, xi0, r, mu):
    """ℓ2 distance over post-action observed features only (xi_c == 0).

    Still-missing features are excluded: their contribution would be
    mu - 0 (imputed mean minus zero-fill), which reflects imputation
    noise rather than anything the user changed.
    """
    x_rec, xi_c = _recourse_point(x0, delta, xi0, r, mu)
    obs_idx = np.where(xi_c == 0)[0]
    return float(np.linalg.norm((x_rec - x0)[obs_idx]))


def rashomon_distance(theta, theta_hat, hessian):
    """(1/2)(θ − θ̂)ᵀ H (θ − θ̂) — how far a model sits from the ERM in the Rashomon ellipsoid."""
    d = theta - theta_hat
    return 0.5 * d @ hessian @ d


def build_ellipsoid_evaluator(theta_hat, hessian, epsilon_target, n_models=50, seed=0):
    """Sample n_models parameter vectors uniformly from the Rashomon ellipsoid."""
    rng = np.random.RandomState(seed)
    p   = len(theta_hat)
    L_inv_T = np.linalg.inv(np.linalg.cholesky(hessian)).T  # H^{-1/2}

    models = []
    for _ in range(n_models):
        v  = rng.randn(p)
        v /= np.linalg.norm(v)                        # uniform direction
        v *= rng.uniform(0, 1) ** (1.0 / p)           # uniform in ball
        v *= np.sqrt(2 * epsilon_target)               # scale to ellipsoid radius
        models.append(theta_hat + L_inv_T @ v)
    return models


def ensemble_robustness(x0, delta, xi0, r, mu, eval_models, tau=0.0):
    """True iff recourse point is approved (score >= tau) by every model in the list."""
    x_rec, xi_c = _recourse_point(x0, delta, xi0, r, mu)
    phi_rec     = build_phi(x_rec.reshape(1, -1), xi_c.reshape(1, -1))
    for theta_e in eval_models:
        s = score(phi_rec, theta_e)[0]
        if not (np.isfinite(s) and s >= tau):
            return False
    return True


def print_recourse_summary(x0, xi0, best_r, best_delta, col_means, col_stds, names,
                           kappa=None, metrics=None):
    """
    Print a feature-level qualitative summary of a recourse recommendation
    with values in original (unstandardized) units.

    Columns:
      Original    — pre-action observed value; '---' if missing
      Miss        — whether the feature was missing before the action
      Reveal      — whether the action reveals this feature
      Post-Action — value after applying delta; '---' if still missing
      Change      — numeric shift for observed edits; 'revealed' for newly revealed features
    """
    d    = len(x0)
    xi_c = xi0 - best_r

    sep = "-" * 80
    print(sep)
    print(f"{'Feature':<28} {'Original':>9} {'Miss':>5} {'Reveal':>7} {'Post-Action':>12} {'Change':>11}")
    print(sep)

    for j in range(d):
        orig_str    = f"{x0[j]*col_stds[j]+col_means[j]:.2f}" if xi0[j] == 0 else "---"
        missing_str = "yes" if xi0[j] == 1 else "no"
        reveal_str  = "yes" if best_r[j] == 1 else "-"

        if xi_c[j] == 0:                                        # observed after action
            post_val = (x0[j] + best_delta[j]) * col_stds[j] + col_means[j]
            post_str = f"{post_val:.2f}"
            if xi0[j] == 0:                                     # was observed before too
                change = best_delta[j] * col_stds[j]
                change_str = f"{change:+.2f}" if abs(change) > 1e-3 else "-"
            else:                                               # was missing, now revealed
                change_str = "revealed"
        else:                                                   # still missing after action
            post_str   = "---"
            change_str = "-"

        print(f"{names[j]:<28} {orig_str:>9} {missing_str:>5} {reveal_str:>7} {post_str:>12} {change_str:>11}")

    print(sep)

    if metrics is not None:
        if kappa is not None:
            reveal_cost = float(kappa @ best_r)
            edit_cost   = metrics["total_cost"] - reveal_cost
            print(f"  cost             : total={metrics['total_cost']:.4f}  "
                  f"reveal={reveal_cost:.4f}  edit={edit_cost:.4f}")
        else:
            print(f"  cost             : {metrics['total_cost']:.4f}")
        print(f"  nominal validity : {metrics['nominal']}")
        print(f"  retrain validity : {metrics['retrain']:.3f}")
        print(f"  awp (sanity)     : {metrics['awp_sanity']}  lb={metrics['lb_sanity']:.4f}")
        print(f"  lof plausibility : {metrics['lof']:.4f}  (1.0 = in-distribution)")
    print()


if __name__ == "__main__":
    from data        import load_diabetes, train_test_split
    from model       import compute_hessian, predict
    from imputer     import fit_mice, calibrate_rho
    from beam_search import beam_search

    X, Xi, y, Phi, names, col_means, col_stds = load_diabetes()
    train_idx, test_idx = train_test_split(len(y))

    theta_hat, _   = train(Phi[train_idx], y[train_idx])
    hessian_matrix = compute_hessian(Phi[train_idx], theta_hat)
    mice_imputer   = fit_mice(X[train_idx], Xi[train_idx])

    denied_with_missing = [
        i for i in test_idx
        if predict(Phi[[i]], theta_hat)[0] == -1 and Xi[i].sum() > 0
    ]

    kappa   = np.ones(len(X[0])) * 0.5
    n_cases = min(3, len(denied_with_missing))

    # precompute shared objects
    bootstrap_models = train_bootstrap_models(
        X[train_idx], Xi[train_idx], y[train_idx], n_models=50,
    )
    lof_model = fit_lof(X[train_idx], Xi[train_idx], mice_imputer)

    for case_num, person_idx in enumerate(denied_with_missing[:n_cases]):
        x0, xi0 = X[person_idx], Xi[person_idx]
        missing_names = [names[j] for j in np.where(xi0 == 1)[0]]
        print(f"\n{'='*80}")
        print(f"Case {case_num+1}  (person {person_idx})  missing: {missing_names}")
        print(f"{'='*80}")

        best_r, best_delta, best_cost, _, best_mu, best_Sigma, best_draws, best_rho_opt = beam_search(
            x0, xi0, theta_hat, hessian_matrix, mice_imputer,
            epsilon=0.001, rho_coverage=0.90, kappa=kappa,
            J_edit=list(range(len(x0))), verbose=False,
        )

        if best_r is None:
            print("no feasible recourse found")
            continue

        rho_stress       = calibrate_rho(best_draws, best_mu, best_Sigma, 0.99)
        mu, Sigma        = best_mu, best_Sigma
        nom              = nominal_validity(theta_hat, x0, best_delta, xi0, best_r, mu)
        rr               = model_retrain_validity(
            x0, best_delta, xi0, best_r, mu, bootstrap_models,
        )
        awp_sanity, lb_sanity = awp_validity(
            theta_hat, hessian_matrix,
            x0, best_delta, xi0, best_r, mu, Sigma,
            epsilon_eval=0.001, rho_eval=best_rho_opt,
        )
        awp_stress, lb_stress = awp_validity(
            theta_hat, hessian_matrix,
            x0, best_delta, xi0, best_r, mu, Sigma,
            epsilon_eval=0.01, rho_eval=rho_stress,
        )
        lof = lof_plausibility(x0, best_delta, xi0, best_r, mu, lof_model)

        metrics = {
            "total_cost": best_cost,
            "nominal":    nom,
            "retrain":    rr,
            "awp_sanity": awp_sanity, "lb_sanity": lb_sanity,
            "awp_stress": awp_stress, "lb_stress": lb_stress,
            "lof":        lof,
        }
        print_recourse_summary(
            x0, xi0, best_r, best_delta, col_means, col_stds, names,
            kappa=kappa, metrics=metrics,
        )
