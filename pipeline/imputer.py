import warnings
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def fit_mice(X_train, Xi_train, max_iter=10, seed=42):
    # convert zero-fill back to NaN so the imputer learns real distributions
    X_nan = X_train.copy().astype(float)
    X_nan[Xi_train == 1] = np.nan

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        sample_posterior=True,  # Gaussian posterior draws for uncertainty estimation
        random_state=seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        imputer.fit(X_nan)
    return imputer


def get_mice_draws(imputer, x0, xi_c, K=200, seed=42):
    rng      = np.random.RandomState(seed)
    miss_idx = np.where(xi_c == 1)[0]
    d_miss   = len(miss_idx)

    if d_miss == 0:
        return np.empty((K, 0))

    draws = np.zeros((K, d_miss))
    for k in range(K):
        x_nan = x0.copy().astype(float)
        x_nan[xi_c == 1] = np.nan

        imputer.random_state_ = np.random.RandomState(rng.randint(0, 100000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_imputed = imputer.transform(x_nan.reshape(1, -1))[0]
        draws[k]  = x_imputed[miss_idx]

    return draws


def compute_mu_sigma(draws, eta=1e-4):
    _, d_miss = draws.shape
    mu    = draws.mean(axis=0)
    Sigma = np.atleast_2d(np.cov(draws.T))
    Sigma += eta * np.eye(d_miss)  # ridge for invertibility
    return mu, Sigma


def calibrate_rho(draws, mu, Sigma, coverage=0.90):
    """
    Empirical `coverage`-quantile of Mahalanobis distances of MICE draws.

    Returns the rho such that `coverage` fraction of draws fall inside the
    ellipsoid {x: (x-mu)^T Sigma^{-1} (x-mu) <= rho^2}.
    """
    K, d_miss = draws.shape
    if d_miss == 0 or K <= 1:
        return 0.0
    Sigma_inv = np.linalg.inv(Sigma)
    diffs     = draws - mu
    mahal     = np.sqrt(np.maximum(
        np.einsum('ki,ij,kj->k', diffs, Sigma_inv, diffs), 0.0
    ))
    return float(np.quantile(mahal, coverage))


def get_imputation_params(imputer, x0, xi_c, K=200, seed=42):
    draws = get_mice_draws(imputer, x0, xi_c, K=K, seed=seed)

    if draws.shape[1] == 0:
        return np.array([]), np.array([[]]), draws

    mu, Sigma = compute_mu_sigma(draws)

    return mu, Sigma, draws


if __name__ == "__main__":
    from data  import load_diabetes, train_test_split
    from model import train, predict

    X, Xi, y, Phi, names, means, stds = load_diabetes()
    train_idx, test_idx = train_test_split(len(y))

    mice_imputer = fit_mice(X[train_idx], Xi[train_idx])

    theta_hat, _ = train(Phi[train_idx], y[train_idx])
    denied_idx   = test_idx[predict(Phi[test_idx], theta_hat) == -1]
    person_idx   = denied_idx[0]

    x0, xi0 = X[person_idx], Xi[person_idx]
    print(f"person {person_idx}  missing: {[names[j] for j in np.where(xi0==1)[0]]}")

    mu, Sigma, draws = get_imputation_params(mice_imputer, x0, xi0, K=100)
    print(f"mu={mu}  sigma diag={np.diag(Sigma)}")