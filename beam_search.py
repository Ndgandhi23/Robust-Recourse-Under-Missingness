import numpy as np
from imputer  import get_imputation_params, calibrate_rho
from recourse import solve_delta


def beam_search(
    x0, xi0,
    theta_hat, hessian_matrix,
    mice_imputer,
    epsilon, rho_coverage, kappa, J_edit,
    tau=0.0,
    beam_width=5,
    K_max=3,
    K_mice=100,
    x_min=None,
    x_max=None,
    verbose=True,
    seed=0,
    rho_override=None,
):
    """
    Beam search over reveal patterns r, solving an SOCP for delta at each node.

    rho_coverage : float in (0, 1)
        Fraction of MICE draws the imputation ellipsoid should cover.
        rho is calibrated empirically per candidate mask from the MICE draws,
        so the ellipsoid always covers the same fraction of the imputer's
        distribution regardless of how many features are missing.

    Returns
    -------
    best_r, best_delta, best_cost, history, best_mu, best_Sigma, best_draws, best_rho
        best_draws : (K_mice, d_miss) array of MICE draws used for the best solution
        best_rho   : calibrated rho value used for the best solution
    """
    rng  = np.random.RandomState(seed)
    d    = len(x0)
    beam = [np.zeros(d, dtype=int)]  # start with reveal nothing
    history = []

    best_r, best_delta, best_cost = None, None, np.inf
    best_mu, best_Sigma           = None, None
    best_draws, best_rho          = None, None

    for step in range(K_max + 1):
        if verbose:
            print(f"\nstep {step}: {len(beam)} candidate(s)")

        step_results = []

        for r in beam:
            xi_c = xi0 - r
            mu, Sigma, draws = get_imputation_params(
                mice_imputer, x0, xi_c, K=K_mice, seed=int(rng.randint(0, 2**31))
            )
            rho = calibrate_rho(draws, mu, Sigma, rho_coverage)
            if rho_override is not None:
                rho = rho_override

            delta_opt, cost = solve_delta(
                x0, xi0, r,
                theta_hat, hessian_matrix,
                mu, Sigma, epsilon, rho,
                kappa, J_edit, tau=tau,
                x_min=x_min, x_max=x_max,
            )

            feasible = delta_opt is not None
            record   = {"r": r.copy(), "delta": delta_opt, "cost": cost, "feasible": feasible}
            history.append(record)
            step_results.append(record)

            if verbose:
                status = f"cost={cost:.4f}" if feasible else "infeasible"
                print(f"  r={r.tolist()}  {status}")

            if feasible and cost < best_cost:
                best_cost  = cost
                best_r     = r.copy()
                best_delta = delta_opt.copy()
                best_mu    = mu.copy()    if mu    is not None and len(mu) > 0    else mu
                best_Sigma = Sigma.copy() if Sigma is not None and Sigma.size > 0 else Sigma
                best_draws = draws.copy()
                best_rho   = rho

        if step == K_max:
            break

        # expand: try revealing one more missing feature
        feasible_results = [s for s in step_results if s["feasible"]]
        candidates = feasible_results if feasible_results else step_results
        candidates = sorted(candidates, key=lambda x: x["cost"])[:beam_width]

        new_beam, seen = [], set()
        for record in candidates:
            for j in range(d):
                if xi0[j] == 1 and record["r"][j] == 0:
                    r_new = record["r"].copy()
                    r_new[j] = 1
                    key = tuple(r_new)
                    if key not in seen:
                        seen.add(key)
                        new_beam.append(r_new)

        if not new_beam:
            if verbose:
                print("no more features to reveal")
            break

        beam = new_beam

    return best_r, best_delta, best_cost, history, best_mu, best_Sigma, best_draws, best_rho


if __name__ == "__main__":
    from data  import load_diabetes, train_test_split
    from model import train, compute_hessian, predict
    from imputer import fit_mice

    X, Xi, y, Phi, names, means, stds = load_diabetes()
    train_idx, test_idx = train_test_split(len(y))

    theta_hat, _   = train(Phi[train_idx], y[train_idx])
    hessian_matrix = compute_hessian(Phi[train_idx], theta_hat)
    mice_imputer   = fit_mice(X[train_idx], Xi[train_idx])

    denied_idx = test_idx[predict(Phi[test_idx], theta_hat) == -1]
    person_idx = denied_idx[0]
    x0, xi0    = X[person_idx], Xi[person_idx]

    print(f"person {person_idx}  missing: {[names[j] for j in np.where(xi0==1)[0]]}")

    best_r, best_delta, best_cost, history, best_mu, best_Sigma, best_draws, best_rho = beam_search(
        x0, xi0, theta_hat, hessian_matrix, mice_imputer,
        epsilon=0.001, rho_coverage=0.90, kappa=np.ones(len(x0))*0.5,
        J_edit=list(range(len(x0))), beam_width=3, K_max=3, verbose=True,
    )

    if best_r is not None:
        print(f"\ncost={best_cost:.4f}")
        print(f"reveal: {[names[j] for j in np.where(best_r==1)[0]]}")
        print(f"calibrated rho (90th pct): {best_rho:.3f}")
    else:
        print("no feasible recourse found")
