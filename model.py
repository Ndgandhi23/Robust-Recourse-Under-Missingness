import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression


def train(Phi_train, y_train, C=1.0):
    model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(Phi_train, y_train)
    theta_hat = np.append(model.coef_[0], model.intercept_[0])
    return theta_hat, model


def compute_hessian(Phi_train, theta_hat, gamma=1e-3):
    n = Phi_train.shape[0]

    # add bias column to match theta_hat dimension
    Phi_aug = np.hstack([Phi_train, np.ones((n, 1))])

    # hessian of logistic loss: H = (1/n) * Phi^T W Phi
    # where W = diag(p_i * (1 - p_i))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            raw_scores      = np.clip(Phi_aug @ theta_hat, -500, 500)
            probs           = 1.0 / (1.0 + np.exp(-raw_scores))
            hessian_weights = probs * (1.0 - probs)
            hessian_matrix  = (Phi_aug.T * hessian_weights) @ Phi_aug / n
    hessian_matrix += gamma * np.eye(hessian_matrix.shape[0])  # damp for invertibility

    return hessian_matrix


def score(Phi, theta_hat):
    n = Phi.shape[0]
    Phi_aug = np.hstack([Phi, np.ones((n, 1))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return Phi_aug @ theta_hat


def predict(Phi, theta_hat, tau=0.0):
    return np.where(score(Phi, theta_hat) >= tau, 1, -1)


def accuracy(Phi, theta_hat, y, tau=0.0):
    return (predict(Phi, theta_hat, tau) == y).mean()


if __name__ == "__main__":
    from data import load_diabetes, train_test_split

    X, Xi, y, Phi, names, means, stds = load_diabetes()
    train_idx, test_idx = train_test_split(len(y))

    theta_hat, model   = train(Phi[train_idx], y[train_idx])
    hessian_matrix     = compute_hessian(Phi[train_idx], theta_hat)

    print(f"theta_hat shape : {theta_hat.shape}")
    print(f"hessian shape   : {hessian_matrix.shape}")
    print(f"min eigenvalue  : {np.linalg.eigvalsh(hessian_matrix).min():.6f}")
    print(f"train acc       : {accuracy(Phi[train_idx], theta_hat, y[train_idx]):.3f}")
    print(f"test acc        : {accuracy(Phi[test_idx],  theta_hat, y[test_idx]):.3f}")