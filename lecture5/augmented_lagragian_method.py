import numpy as np
import matplotlib.pyplot as plt

def solve_inequality_constrained_kkt(f, grad_f, hessian_f,
                                     h_list, grad_h_list, hessian_h_list=None,
                                     x0=None, mu0=None,
                                     rho=10.0, rho_max=1e6,
                                     tol=1e-6, max_outer=50, max_inner=50,
                                     beta=1e-6):
    """
    Solve an inequality-constrained optimization problem:
        min_x f(x)
        subject to h_i(x) ≤ 0
    using Augmented Lagrangian Method (ALM).

    This version:
    - Includes mu_i * h_i(x) terms.
    - Uses hessian_h_list if provided to incorporate second-order constraint info.
    - Adds regularization to KKT system to ensure numerical stability.
    """

    x = x0.copy().astype(float)
    m = len(h_list)
    if mu0 is None:
        mu = np.zeros(m, dtype=float)
    else:
        mu = mu0.copy().astype(float)

    def f_aug(x):
        val = f(x)
        for i in range(m):
            hi = h_list[i](x)[0]
            viol = max(0, hi)
            val += mu[i]*hi + 0.5*rho*(viol**2)
        return val

    def grad_f_aug(x):
        g = grad_f(x).copy()
        for i in range(m):
            hi = h_list[i](x)[0]
            gh = grad_h_list[i](x)
            g += mu[i]*gh
            if hi > 0:
                g += rho*hi*gh
        return g

    def hess_f_aug(x):
        H = hessian_f(x).copy()
        for i in range(m):
            hi = h_list[i](x)[0]
            gh = grad_h_list[i](x)

            # If we have second-order info of constraints:
            if hessian_h_list is not None:
                H += mu[i] * hessian_h_list[i](x)

            # If hi > 0, add penalty Hessian: rho * gh * gh^T
            if hi > 0:
                H += rho * np.outer(gh, gh)
        return H

    def solve_kkt_system(H, A, g, r, beta=1e-6):
        """
        Solve the KKT system with regularization to ensure m negative eigenvalues.
        K = [H, A^T; A, -beta*I]
        """
        n, m = H.shape[0], A.shape[0]
        reg_matrix_H = np.zeros((n, n))
        reg_matrix_A = np.zeros((m, m))
        reg_value = beta

        while True:
            # Construct the regularized KKT matrix
            K = np.block([
                [H + reg_matrix_H, A.T],
                [A, -reg_matrix_A]
            ])
            rhs = np.concatenate([-g, -r])  # Right-hand side: [-g; -r]

            try:
                # Compute eigenvalues to check distribution
                eigvals = np.linalg.eigvalsh(K)
                num_positive = np.sum(eigvals > 0)
                num_negative = np.sum(eigvals < 0)

                if num_positive == n and num_negative == m:
                    # Solve the regularized KKT system
                    solution = np.linalg.solve(K, rhs)
                    return solution[:n], solution[n:]  # Return dx, d_lambda
            except np.linalg.LinAlgError:
                pass  # Continue increasing regularization if needed
            
            # Increase regularization value if conditions are not met
            reg_matrix_H += reg_value * np.eye(n)
            reg_matrix_A += reg_value * np.eye(m)
            reg_value *= 10

    def solve_unconstrained_newton(x):
        """
        Solve the unconstrained subproblem using the KKT system with line search.
        """
        x_inner_history = [x.copy()]  # Initialize history with the starting point
        for _ in range(max_inner):
            H = hess_f_aug(x)
            g = grad_f_aug(x)
            A = np.vstack([grad_h(x) for grad_h in grad_h_list])
            r = np.array([h(x)[0] for h in h_list])  # Residuals of constraints

            dx, _ = solve_kkt_system(H, A, g, r, beta=beta)
            
            # Line search to determine step length
            alpha = 1.0  # Initial step size
            # c1 = 1e-4    # Armijo parameter
            # while f_aug(x + alpha * dx) > f_aug(x) + c1 * alpha * np.dot(grad_f_aug(x), dx):
            #     alpha *= 0.5  # Reduce step size

            # Update x
            x += alpha * dx
            x_inner_history.append(x.copy())  # Save the updated x

            if np.linalg.norm(dx) < tol:
                break
        return x, x_inner_history

    x_history = [x.copy()]

    for outer_iter in range(max_outer):
        x, x_inner_history = solve_unconstrained_newton(x)
        x_history.extend(x_inner_history)  # Append the inner history to the overall history

        h_vals = np.array([h(x)[0] for h in h_list])
        mu = np.maximum(0.0, mu + rho * h_vals)

        feasible = np.all(h_vals < tol)
        complementarity = np.max(np.abs(mu * h_vals))

        if feasible and complementarity < tol:
            print(f"ALM converged in {outer_iter + 1} outer iterations.")
            break
        else:
            if rho < rho_max:
                rho *= 2.0

    return x, mu, x_history

def plot_landscape(f, curve=None, points=None):
    """
    Plot the objective function landscape, optional constraint curve, and iteration path.
    """
    Nsamp = 100
    Xsamp = np.linspace(-4, 4, Nsamp)
    Ysamp = np.linspace(-4, 4, Nsamp)
    Xgrid, Ygrid = np.meshgrid(Xsamp, Ysamp)

    Zsamp = np.zeros_like(Xgrid, dtype=float)
    for j in range(Nsamp):
        for k in range(Nsamp):
            Zsamp[j, k] = f(np.array([Xgrid[j, k], Ygrid[j, k]]))

    plt.contour(Xgrid, Ygrid, Zsamp, levels=20, cmap="viridis")

    if curve is not None:
        x_curve, y_curve = curve
        plt.plot(x_curve, y_curve, "y", label="h(x)=0")

    if points is not None:
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], "rx-", label="Iteration path")

    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.title("Objective with Inequality Constraint & ALM")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Define Q and objective
    Q = np.diag([0.5, 1.0])
    f = lambda x: 0.5*np.dot((x - np.array([1., 0.])), Q @ (x - np.array([1.,0.])))
    grad_f = lambda x: Q @ (x - np.array([1.,0.]))
    hessian_f = lambda x: Q

    # Constraint: h(x)= (x0^2+2x0)-x1 ≤0
    # grad h: [2x0+2, -1]
    # Hessian h: [[2,0],[0,0]]
    h_list = [lambda x: np.array([ (x[0]**2 + 2.0*x[0] - x[1]) ])]
    grad_h_list = [lambda x: np.array([2.0*x[0]+2.0, -1.0])]
    hessian_h_list = [lambda x: np.array([[2.0, 0.0],
                                          [0.0, 0.0]])]

    x0 = np.array([-3.0, 2.0])

    x_sol, mu_sol, x_hist = solve_inequality_constrained_kkt(
        f, grad_f, hessian_f,
        h_list, grad_h_list, hessian_h_list=hessian_h_list,
        x0=x0, mu0=None,
        rho=10.0, tol=1e-6,
        max_outer=20, max_inner=50,
        beta=1e-6
    )

    print("Solution x:", x_sol)
    print("Multipliers mu:", mu_sol)

    # Plot the constraint curve h(x)=0 => x1 = x0^2+2x0
    xc = np.linspace(-3.2, 1.2, 100)
    yc = xc**2 + 2.0*xc

    plot_landscape(f, curve=(xc, yc), points=x_hist)
