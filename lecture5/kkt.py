import numpy as np

from lecture3.root_finding import iterative_solver

def solve_kkt_system(f, grad_f, hessian_f, c_list, grad_c_list, hessian_c_list=None, x0=None, lambda0=None, tol=1e-6, max_iter=50):
    """
    Solves the equality-constrained optimization problem using the KKT conditions with Newton or Gauss-Newton methods.

    Args:
        f (function): Objective function.
        grad_f (function): Gradient of the objective function.
        hessian_f (function): Hessian of the objective function.
        c_list (list): List of constraint functions.
        grad_c_list (list): List of gradients of constraint functions.
        hessian_c_list (list or None): List of Hessians of constraint functions. If None, use Gauss-Newton approximation.
        x0 (np.ndarray): Initial guess for primal variables (x).
        lambda0 (np.ndarray): Initial guess for Lagrange multipliers.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: Solution for x, lambda, and iteration history (list of x).
    """
    # Initialization
    x = x0.copy().reshape(-1, 1)  # Ensure x is a column vector
    lambda_ = lambda0.copy().reshape(-1, 1)  # Ensure lambda is a column vector
    x_history = [x.copy()]  # Store iteration history

    for i in range(max_iter):
        # Compute the gradient of f and the Hessian
        grad = grad_f(x)
        H = hessian_f(x).copy()

        # Assemble constraints and Jacobians
        c_vals = np.vstack([c(x) for c in c_list])  # Constraint values
        C = np.vstack([grad_c(x) for grad_c in grad_c_list])  # Constraint Jacobian

        # Add Hessian contributions from constraints if available
        if hessian_c_list is not None:
            # Newton method
            for hessian_c, lambda_i in zip(hessian_c_list, lambda_):
                H += hessian_c(x) * lambda_i
        else:
            # Gauss-Newton method
            pass
        
        # Form the KKT system
        KKT_matrix = np.block([
            [H, C.T],
            [C, np.zeros((C.shape[0], C.shape[0]))]
        ])
        KKT_rhs = -np.block([
            grad + C.T @ lambda_,
            c_vals
        ])

        # Solve the KKT system
        delta = np.linalg.solve(KKT_matrix, KKT_rhs)
        delta_x = delta[:x.shape[0]]  # Primal variable update
        delta_lambda = delta[x.shape[0]:]  # Dual variable update

        # Update variables
        x += delta_x
        lambda_ += delta_lambda

        # Store iteration history
        x_history.append(x.copy())

        # Check for convergence
        if np.linalg.norm(delta_x) < tol and np.linalg.norm(delta_lambda) < tol:
            print(f"Converged in {i+1} iterations.")
            break
        else:
            print("Maximum iterations reached without convergence.")

        return x.flatten(), lambda_.flatten(), x_history


if __name__ == "__main__":
    # Define the objective function, gradient, and Hessian using lambda
    Q = np.diag([0.5, 1.0])
    f = lambda x: 0.5 * np.dot((x - np.array([1, 0])), Q @ (x - np.array([1, 0])))
    grad_f = lambda x: Q @ (x - np.array([1, 0]))
    hessian_f = lambda x: Q

    # Define the constraint function, gradient, and Hessian using lambda
    c = lambda x: np.array([x[0]**2 + 2 * x[0] - x[1]])
    grad_c = lambda x: np.array([[2 * x[0] + 2, -1]])
    hessian_c = lambda x: np.array([[2, 0], [0, 0]])

    