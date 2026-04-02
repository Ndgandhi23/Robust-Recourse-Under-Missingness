import numpy as np
import cvxpy as cp


def compute_A_b(x0, xi0, delta, r):
    """
    For a given action (delta, r), finds A and b such that:
        phi(x', xi_c) = A @ x_miss + b

    phi = [x, xi, 1]  shape (2d+1,)
    A   shape (2d+1, d_miss)
    b   shape (2d+1,)
    """
    d        = len(x0)
    xi_c     = xi0 - r
    xc       = x0  + delta
    miss_idx = np.where(xi_c == 1)[0]
    obs_idx  = np.where(xi_c == 0)[0]
    d_miss   = len(miss_idx)

    A = np.zeros((2*d+1, d_miss))
    b = np.zeros(2*d+1)

    # block 1: x  (rows 0..d-1)
    for j in obs_idx:
        b[j] = xc[j]
    for k, j in enumerate(miss_idx):
        A[j, k] = 1.0

    # block 2: xi  (rows d..2d-1)
    for j in range(d):
        b[d + j] = float(xi_c[j])

    # block 3: bias
    b[2*d] = 1.0

    return A, b, miss_idx


def worst_case_lower_bound(theta_hat, hessian_matrix, A, b, mu, Sigma, epsilon, rho):
    """
    Closed-form lower bound on the worst-case score over the
    Rashomon ellipsoid and imputation ellipsoid simultaneously (Proposition 2).

    LB = theta_hat^T v
         - sqrt(2*eps) * ||H^{-1/2} v||
         - rho         * ||Sigma^{1/2} A^T theta_hat||
         - sqrt(2*eps) * rho * ||H^{-1/2} A Sigma^{1/2}||
    where v = A @ mu + b
    """
    L          = np.linalg.cholesky(hessian_matrix)
    H_inv_sqrt = np.linalg.inv(L).T

    v     = A @ mu + b
    term0 = theta_hat @ v
    term1 = np.sqrt(2 * epsilon) * np.linalg.norm(H_inv_sqrt @ v)

    if A.shape[1] == 0:
        return term0 - term1

    Sigma_sqrt = np.linalg.cholesky(Sigma)
    term2      = rho * np.linalg.norm(Sigma_sqrt @ (A.T @ theta_hat))
    term3      = np.sqrt(2 * epsilon) * rho * np.linalg.norm(H_inv_sqrt @ A @ Sigma_sqrt, ord=2)

    return term0 - term1 - term2 - term3


def solve_delta(x0, xi0, r, theta_hat, hessian_matrix, mu, Sigma, epsilon, rho,
                kappa, J_edit, tau=0.0, x_min=None, x_max=None):
    """
    For fixed r, solves a convex SOCP over delta:
        min  ||delta_{J_edit}||_2 + sum_j kappa_j r_j
        s.t. worst_case_lower_bound(delta) >= tau
             delta_j = 0 for j not editable
             delta_j = 0 for unrevealed missing features
    """
    d        = len(x0)
    xi_c     = xi0 - r
    miss_idx = np.where(xi_c == 1)[0]
    d_miss   = len(miss_idx)

    L          = np.linalg.cholesky(hessian_matrix)
    H_inv_sqrt = np.linalg.inv(L).T

    # precompute constants that don't depend on delta
    A, b0, _ = compute_A_b(x0, xi0, np.zeros(d), r)

    if d_miss > 0:
        Sigma_sqrt   = np.linalg.cholesky(Sigma)
        const_term2  = rho * np.linalg.norm(Sigma_sqrt @ (A.T @ theta_hat))
        const_term3  = np.sqrt(2 * epsilon) * rho * np.linalg.norm(
            H_inv_sqrt @ A @ Sigma_sqrt, ord=2
        )
    else:
        const_term2 = 0.0
        const_term3 = 0.0
        A           = np.zeros((2*d+1, 0))

    # b is affine in delta — only observed-after-action features contribute
    B_delta = np.zeros((2*d+1, d))
    for j in np.where(xi_c == 0)[0]:
        B_delta[j, j] = 1.0

    delta = cp.Variable(d)
    v     = A @ mu + b0 + B_delta @ delta

    term0 = theta_hat @ v
    term1 = np.sqrt(2 * epsilon) * cp.norm(H_inv_sqrt @ v, 2)

    robust_constraint = term0 - term1 - const_term2 - const_term3 >= tau

    constraints = [robust_constraint]

    # can't edit features outside J_edit
    for j in range(d):
        if j not in J_edit:
            constraints.append(delta[j] == 0)

    # can't edit unrevealed missing features
    for j in range(d):
        if xi0[j] == 1 and r[j] == 0:
            constraints.append(delta[j] == 0)

    if x_min is not None:
        constraints.append(x0 + delta >= x_min)
    if x_max is not None:
        constraints.append(x0 + delta <= x_max)

    edit_mask    = np.zeros(d)
    edit_mask[J_edit] = 1.0
    reveal_cost  = float(kappa @ r)
    edit_cost    = cp.norm(cp.multiply(edit_mask, delta), 2)

    prob = cp.Problem(cp.Minimize(edit_cost + reveal_cost), constraints)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except cp.SolverError:
        return None, np.inf

    if prob.status == "optimal" and delta.value is not None:
        return delta.value, float(prob.value)
    return None, np.inf


if __name__ == "__main__":
    from data    import load_diabetes, train_test_split
    from model   import train, compute_hessian, predict
    from imputer import fit_mice, get_imputation_params

    X, Xi, y, Phi, names, means, stds = load_diabetes()
    train_idx, test_idx = train_test_split(len(y))

    theta_hat, _   = train(Phi[train_idx], y[train_idx])
    hessian_matrix = compute_hessian(Phi[train_idx], theta_hat)
    mice_imputer   = fit_mice(X[train_idx], Xi[train_idx])

    denied_idx = test_idx[predict(Phi[test_idx], theta_hat) == -1]
    person_idx = denied_idx[0]
    x0, xi0    = X[person_idx], Xi[person_idx]

    r              = np.zeros(len(x0), dtype=int)
    mu, Sigma, _ = get_imputation_params(mice_imputer, x0, xi0, K=100)

    delta_opt, cost = solve_delta(
        x0, xi0, r, theta_hat, hessian_matrix, mu, Sigma,
        epsilon=0.001, rho=1.0, kappa=np.ones(len(x0))*0.5,
        J_edit=list(range(len(x0))), tau=0.0
    )

    if delta_opt is not None:
        print(f"feasible  cost={cost:.4f}")
        for j, name in enumerate(names):
            if abs(delta_opt[j]) > 1e-4:
                print(f"  {name}: {x0[j]:.3f} -> {x0[j]+delta_opt[j]:.3f}")
    else:
        print("infeasible with r=0")