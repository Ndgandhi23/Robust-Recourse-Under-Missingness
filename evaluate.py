import numpy as np
from data    import build_phi
from model   import train, score


def _recourse_point(x0, delta, xi0, r, mu):
    """Post-action feature vector with mu plugged into still-missing entries."""
    x_rec = (x0 + delta).copy()
    xi_c  = xi0 - r
    miss_idx = np.where(xi_c == 1)[0]
    if len(miss_idx) > 0:
        x_rec[miss_idx] = mu
    return x_rec, xi_c


def random_retrain_validity(
    X_train, Xi_train, y_train,
    x0, delta, xi0, r, mu,
    tau=0.0, n_models=50, seed=0,
):
    """
    Retrain on bootstrap samples n_models times. Returns fraction of
    retrained models for which the recourse action achieves score >= tau.
    """
    rng = np.random.RandomState(seed)
    n   = len(y_train)
    x_rec, xi_c = _recourse_point(x0, delta, xi0, r, mu)
    phi_rec     = build_phi(x_rec.reshape(1, -1), xi_c.reshape(1, -1))

    valid = 0
    for _ in range(n_models):
        idx    = rng.choice(n, size=n, replace=True)
        Phi_b  = build_phi(X_train[idx], Xi_train[idx])
        theta_b, _ = train(Phi_b, y_train[idx])
        if score(phi_rec, theta_b)[0] >= tau:
            valid += 1

    return valid / n_models


def rashomon_dropout_validity(
    theta_hat,
    x0, delta, xi0, r, mu,
    dropout_rate=0.1, n_samples=200, tau=0.0, seed=0,
):
    """
    Rashomon Dropout (Hsu et al.). Apply random binary dropout masks to
    theta_hat at inference time — each thinned subnetwork approximates a
    member of the Rashomon set. Returns fraction of masked models for which
    the recourse action achieves score >= tau.

    For linear models: randomly zero out weight components, rescale to
    preserve expected value.
    """
    rng = np.random.RandomState(seed)
    p   = len(theta_hat)
    x_rec, xi_c = _recourse_point(x0, delta, xi0, r, mu)
    phi_rec     = build_phi(x_rec.reshape(1, -1), xi_c.reshape(1, -1))
    phi_aug     = np.append(phi_rec[0], 1.0)

    valid = 0
    for _ in range(n_samples):
        mask          = (rng.uniform(0, 1, p) > dropout_rate).astype(float)
        theta_thinned = theta_hat * mask / (1.0 - dropout_rate)  # rescale
        if phi_aug @ theta_thinned >= tau:
            valid += 1

    return valid / n_samples


def awp_validity(
    theta_hat, hessian_matrix,
    x0, delta, xi0, r, mu,
    epsilon, tau=0.0,
):
    """
    Adversarial Weight Perturbation. Finds the worst-case theta within
    R_ellip(epsilon) for this specific recourse point using the closed form:

        theta_adv = theta_hat - sqrt(2*eps) * H^{-1} phi / sqrt(phi^T H^{-1} phi)

    Returns (passes: bool, adversarial_score: float).
    Always passes for our method by construction — useful for evaluating baselines.
    """
    x_rec, xi_c = _recourse_point(x0, delta, xi0, r, mu)
    phi_rec     = build_phi(x_rec.reshape(1, -1), xi_c.reshape(1, -1))
    phi_aug     = np.append(phi_rec[0], 1.0)

    L          = np.linalg.cholesky(hessian_matrix)
    H_inv_sqrt = np.linalg.inv(L).T
    H_inv_phi  = H_inv_sqrt @ (H_inv_sqrt.T @ phi_aug)

    denom = np.sqrt(phi_aug @ H_inv_phi)
    if denom < 1e-12:
        return True, float(theta_hat @ phi_aug)

    theta_adv  = theta_hat - np.sqrt(2 * epsilon) * H_inv_phi / denom
    score_adv  = float(phi_aug @ theta_adv)
    return bool(score_adv >= tau), score_adv


if __name__ == "__main__":
    from data        import load_diabetes, train_test_split
    from model       import compute_hessian, predict
    from imputer     import fit_mice, get_imputation_params
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
    person_idx = denied_with_missing[0]
    x0, xi0    = X[person_idx], Xi[person_idx]

    best_r, best_delta, best_cost, _ = beam_search(
        x0, xi0, theta_hat, hessian_matrix, mice_imputer,
        epsilon=0.001, rho=1.0, kappa=np.ones(len(x0)) * 0.5,
        J_edit=list(range(len(x0))), verbose=False,
    )

    if best_r is None:
        print("no feasible recourse found")
    else:
        xi_c = xi0 - best_r
        mu, Sigma, _ = get_imputation_params(mice_imputer, x0, xi_c, K=100)

        rr  = random_retrain_validity(
            X[train_idx], Xi[train_idx], y[train_idx],
            x0, best_delta, xi0, best_r, mu, n_models=50,
        )
        rd  = rashomon_dropout_validity(
            theta_hat, x0, best_delta, xi0, best_r, mu,
            dropout_rate=0.1, n_samples=200,
        )
        awp, score_adv = awp_validity(
            theta_hat, hessian_matrix,
            x0, best_delta, xi0, best_r, mu,
            epsilon=0.001,
        )

        print(f"random retrain validity   : {rr:.3f}")
        print(f"rashomon dropout validity : {rd:.3f}")
        print(f"awp passes                : {awp}  (adversarial score={score_adv:.4f})")
