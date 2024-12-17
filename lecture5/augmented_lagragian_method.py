import numpy as np
import matplotlib.pyplot as plt

def solve_unconstrained_newton(f, grad_f, hess_f, x0, tol=1e-6, max_iter=50):
    """
    Solve an unconstrained optimization problem:
        min_x f(x)
    using Newton's method.
    """
    x = x0.copy().astype(float)
    for i in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        if np.linalg.norm(g) < tol:
            break
        dx = np.linalg.solve(H, -g)
        x += dx
        if np.linalg.norm(dx) < tol:
            break
    return x

def solve_inequality_constrained_kkt(f, grad_f, hessian_f,
                                     h_list, grad_h_list,
                                     x0, mu0=None,
                                     rho=10.0, rho_max=1e6,
                                     tol=1e-6, max_outer=50, max_inner=50):
    """
    Solve an inequality-constrained optimization problem (h_i(x) ≤ 0) using ALM, 
    but always include the mu_i * h_i(x) term regardless of feasibility.

    L_rho(x, mu) = f(x) + sum_i [ mu_i * h_i(x) + (rho/2)*max(0,h_i(x))^2 ]

    Args:
        f, grad_f, hessian_f: objective and its derivatives.
        h_list: list of constraints h_i(x).
        grad_h_list: list of gradients of h_i.
        x0: initial guess.
        mu0: initial multipliers, if None, set to zero.
        rho: initial penalty parameter.
        rho_max: maximum penalty parameter.
        tol: tolerance for convergence.
        max_outer: max outer iterations.
        max_inner: max inner iterations for Newton steps.

    Returns:
        x: solution
        mu: multipliers
        x_history: iteration history (outer steps)
    """

    x = x0.copy().astype(float)
    m = len(h_list)
    if mu0 is None:
        mu = np.zeros(m, dtype=float)
    else:
        mu = mu0.copy().astype(float)

    x_history = [x.copy()]

    def f_aug(x):
        val = f(x)
        for i in range(m):
            hi = h_list[i](x)[0]
            viol = max(0, hi)   # violation if hi>0, else 0
            val += mu[i]*hi + 0.5*rho*(viol**2)
        return val

    def grad_f_aug(x):
        g = grad_f(x).copy()
        for i in range(m):
            hi = h_list[i](x)[0]
            gh = grad_h_list[i](x)
            viol = max(0, hi)
            # d/dx [ mu[i]*hi + (rho/2)*viol^2 ]
            # = mu[i]*gh + rho*viol*(1 if hi>0 else 0)*gh
            # Since viol = max(0,hi), if hi>0, viol=hi
            if hi > 0:
                g += mu[i]*gh + rho*hi*gh
            else:
                g += mu[i]*gh
        return g

    def hess_f_aug(x):
        H = hessian_f(x).copy()
        for i in range(m):
            hi = h_list[i](x)[0]
            gh = grad_h_list[i](x)
            # Hessian contribution from mu[i]*hi is 0 since mu[i] and hi are linear in x,
            # but from the penalty part (rho/2)*viol^2 = (rho/2)*max(0,hi)^2
            # If hi>0: second derivative adds rho*gh*gh^T
            if hi > 0:
                H += rho * np.outer(gh, gh)
            # No additional term from mu[i]*hi since it's linear in x
        return H

    for outer_iter in range(max_outer):
        x = solve_unconstrained_newton(f_aug, grad_f_aug, hess_f_aug, x, tol=tol, max_iter=max_inner)

        h_vals = np.array([h(x)[0] for h in h_list])
        # Update multipliers: mu_i = max(0, mu_i + rho*h_i(x))
        mu = np.maximum(0.0, mu + rho*h_vals)

        x_history.append(x.copy())

        # Check conditions: h_i(x) ≤ 0 feasible
        feasible = np.all(h_vals < tol)
        complementarity = np.max(np.abs(mu * h_vals))

        if feasible and complementarity < tol:
            print(f"ALM converged in {outer_iter+1} outer iterations.")
            break
        else:
            if rho < rho_max:
                rho *= 2.0

    return x, mu, x_history

def plot_landscape(f, curve=None, points=None):
    """
    Plot the objective function landscape with optional inequality constraint boundary (h(x)=0)
    and iteration points.
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
        plt.plot(x_curve, y_curve, "y", label="Inequality constraint boundary: h(x)=0")

    if points is not None:
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], "rx-", label="Iteration path")

    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.title("Objective Function Landscape with Inequality Constraint (h(x) ≤ 0)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    Q = np.diag([0.5, 1.0])
    f = lambda x: 0.5*np.dot((x - np.array([1., 0.])), Q @ (x - np.array([1.,0.])))
    grad_f = lambda x: Q @ (x - np.array([1.,0.]))
    hessian_f = lambda x: Q

    # h(x)= x1-(x0^2+2x0) ≤ 0
    h_list = [lambda x: np.array([ x[1] - (x[0]**2 + 2.0*x[0]) ])]
    grad_h_list = [lambda x: np.array([-2.0*x[0]-2.0, 1.0])]

    x0 = np.array([-1.0, -1.0])
    x_sol, mu_sol, x_hist = solve_inequality_constrained_kkt(f, grad_f, hessian_f,
                                                             h_list, grad_h_list,
                                                             x0, mu0=None,
                                                             rho=10.0, tol=1e-6,
                                                             max_outer=20, max_inner=50)
    print("Solution x:", x_sol)
    print("Multipliers mu:", mu_sol)

    xc = np.linspace(-3.2, 1.2, 100)
    yc = xc**2 + 2.0*xc
    plot_landscape(f, curve=(xc, yc), points=x_hist)
