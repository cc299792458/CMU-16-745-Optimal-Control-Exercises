import numpy as np
import matplotlib.pyplot as plt

def solve_kkt_system(grad_f, hessian_f, c_list, grad_c_list, hessian_c_list=None, x0=None, lambda0=None, tol=1e-6, max_iter=50):
    """
    Solves the equality-constrained optimization problem using the KKT conditions with Newton or Gauss-Newton methods.

    Args:
        grad_f (function): Gradient of the objective function, returns (n,).
        hessian_f (function): Hessian of the objective function, returns (n,n).
        c_list (list): List of constraint functions, each returns (1,) for a single constraint.
        grad_c_list (list): List of gradients of constraint functions, each returns (n,).
        hessian_c_list (list or None): List of Hessians of constraint functions. If None, use Gauss-Newton approximation.
        x0 (np.ndarray): Initial guess for primal variables (x), shape (n,).
        lambda0 (np.ndarray): Initial guess for Lagrange multipliers, shape (m,).
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: Solution for x (n,), lambda (m,), and iteration history (list of x).
    """
    # Ensure floating type for x and lambda
    x = x0.astype(float)
    lambda_ = lambda0.astype(float)

    x_history = [x.copy()]  # Store iteration history

    for i in range(max_iter):
        # Compute the gradient of f and the Hessian
        grad = grad_f(x)             # (n,) float
        H = hessian_f(x).copy()      # (n,n) float

        # Assemble constraints and Jacobians
        c_vals = np.concatenate([c(x) for c in c_list])  # (m,) float
        
        C = np.vstack([grad_c(x).reshape(1, -1) for grad_c in grad_c_list])  # (m,n) float

        # Add Hessian contributions from constraints if available
        if hessian_c_list is not None:
            # Newton method
            for hessian_c, lambda_i in zip(hessian_c_list, lambda_):
                H += lambda_i * hessian_c(x)
        else:
            # Gauss-Newton method
            H += C.T @ C

        # Form the KKT system
        KKT_matrix = np.block([
            [H, C.T],
            [C, np.zeros((C.shape[0], C.shape[0]))]
        ])

        # KKT_rhs is (n+m,) float
        KKT_rhs = -np.concatenate((grad + C.T @ lambda_, c_vals))

        # Solve the KKT system
        delta = np.linalg.solve(KKT_matrix, KKT_rhs)  # (n+m,) float

        # Extract updates
        n = x.size
        delta_x = delta[:n]      # (n,)
        delta_lambda = delta[n:] # (m,)

        # Update variables
        x = x + delta_x
        lambda_ = lambda_ + delta_lambda

        # Store iteration history
        x_history.append(x.copy())

        # Check for convergence
        if np.linalg.norm(delta_x) < tol and np.linalg.norm(delta_lambda) < tol:
            print(f"Converged in {i+1} iterations.")
            break
    else:
        print("Maximum iterations reached without convergence.")

    return x, lambda_, x_history

def plot_landscape(f, curve=None, points=None):
    """
    Plots the objective function landscape with optional constraint curves and points.

    Args:
        f (function): The objective function to plot.
        curve (tuple of np.ndarray, optional): A constraint curve to plot, e.g., (x_values, y_values).
        points (list of np.ndarray, optional): Points to overlay on the plot, e.g., iteration history.
    """
    # Number of samples for the landscape
    Nsamp = 100

    # Generate the sampling grid
    Xsamp = np.linspace(-4, 4, Nsamp)
    Ysamp = np.linspace(-4, 4, Nsamp)
    Xgrid, Ygrid = np.meshgrid(Xsamp, Ysamp)

    # Compute Z values for the objective function
    Zsamp = np.zeros_like(Xgrid, dtype=float)
    for j in range(Nsamp):
        for k in range(Nsamp):
            Zsamp[j, k] = f(np.array([Xgrid[j, k], Ygrid[j, k]]))

    # Plot the landscape
    plt.contour(Xgrid, Ygrid, Zsamp, levels=20, cmap="viridis")

    # Plot the constraint curve, if provided
    if curve is not None:
        x_curve, y_curve = curve
        plt.plot(x_curve, y_curve, "y", label="Constraint Curve: c(x) = 0")

    # Plot points, if provided
    if points is not None:
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], "rx-", label="Iteration Path")

    # Add labels, legend, and grid
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.title("Objective Function Landscape with Constraint")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Define Q as a diagonal matrix
    Q = np.diag([0.5, 1.0])

    # Objective function components
    f = lambda x: 0.5 * np.dot((x - np.array([1.0, 0.0])), Q @ (x - np.array([1.0, 0.0])))
    grad_f = lambda x: Q @ (x - np.array([1.0, 0.0]))
    hessian_f = lambda x: Q

    # Constraint components
    c_list = [lambda x: np.array([x[0]**2 + 2.0 * x[0] - x[1]])]  # returns (1,)
    grad_c_list = [lambda x: np.array([2.0 * x[0] + 2.0, -1.0])]  # returns (2,)
    hessian_c_list = [lambda x: np.array([[2.0, 0.0],
                                          [0.0, 0.0]])]
    # hessian_c_list = None   # Gauss-Newton method

    # Initial guesses as float arrays
    x0 = np.array([-1.0, -1.0], dtype=float)
    lambda0 = np.array([0.0], dtype=float)

    # Solve the KKT system using Newton's method
    x_sol, lambda_sol, x_history = solve_kkt_system(grad_f, hessian_f, c_list, grad_c_list, hessian_c_list, x0, lambda0)
    print(f"Optimal solution x: {x_sol}")
    print(f"Optimal lambda: {lambda_sol}")

    # Define the constraint curve
    xc = np.linspace(-3.2, 1.2, 100)
    yc = xc**2 + 2.0 * xc

    # Plot the landscape with iteration path
    plot_landscape(f, curve=(xc, yc), points=x_history)
