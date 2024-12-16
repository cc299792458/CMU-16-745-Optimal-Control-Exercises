import numpy as np
import matplotlib.pyplot as plt

from lecture3.root_finding import iterative_solver  

def minimization(f, grad_f, hessian_f, x0, tol=1e-6, max_iter=100, regularization=None, armijo_params=None):
    """
    Newton's method for minimizing a function using iterative_solver.

    Args:
        f (function): Objective function.
        grad_f (function): Gradient of the objective function.
        hessian_f (function): Hessian matrix of the objective function.
        x0 (float or np.array): Initial guess.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        regularization (float or None): Small positive value added iteratively to the diagonal of the Hessian to ensure positive definiteness. If None, no regularization is applied.
        armijo_params (dict or None): Parameters for Armijo Rule (alpha_0, rho, c). If None, no Armijo step is applied.
        
    Returns:
        float: Minimum point.
        float: Function value at the minimum point.
        int: Number of iterations performed.
        list: Iterates at each step.
    """
    # Define the residual and Jacobian for Newton's method
    def residual(x):
        return grad_f(x)  # The gradient acts as the residual

    def jacobian(x):
        hess = hessian_f(x)
        if isinstance(hess, np.ndarray) and hess.ndim == 2 and regularization is not None:
            # Ensure positive definiteness by iterative regularization
            reg_value = regularization
            while True:
                try:
                    # Check if Hessian is positive definite
                    np.linalg.cholesky(hess)
                    break
                except np.linalg.LinAlgError:
                    hess += reg_value * np.eye(hess.shape[0])
                    reg_value *= 10  # Gradually increase regularization if needed
        return hess

    # Store the iterates
    iterates = [x0]

    # Use iterative_solver to minimize the function
    def recording_residual(x, record_iterates=False):
        if record_iterates:  # Only store points from main iterations
            iterates.append(x.item() if isinstance(x, np.ndarray) else x)
        return residual(x)

    x_min, _, iterations = iterative_solver(recording_residual, derivative=jacobian, x0=x0, tol=tol, max_iter=max_iter, armijo_params=armijo_params)

    return x_min.item() if isinstance(x_min, np.ndarray) else x_min, f(x_min), iterations, iterates

if __name__ == "__main__":
    # Define the objective function, gradient, and Hessian for x^4 + x^3 - x^2 - x
    f = lambda x: x**4 + x**3 - x**2 - x if isinstance(x, np.ndarray) else np.array(x**4 + x**3 - x**2 - x)
    grad_f = lambda x: 4 * x**3 + 3 * x**2 - 2 * x - 1 if isinstance(x, np.ndarray) else np.array(4 * x**3 + 3 * x**2 - 2 * x - 1)
    hessian_f = lambda x: np.atleast_2d(np.squeeze(12 * x**2 + 6 * x - 2)) # Return 2D Hessian for 1D problem

    # Initial guess
    x0 = 0.0 # 1.0, -1.5

    # Regularization parameter
    regularization = 1e-5  # Ensures positive definiteness of the Hessian

    # Armijo Rule parameters
    armijo_params = {
        "alpha_0": 1.0,  # Initial step size
        "rho": 0.5,      # Reduction factor
        "c": 1e-4        # Sufficient decrease constant
    }

    # Perform minimization
    x_min, f_min, iterations, iterates = minimization(f, grad_f, hessian_f, x0, regularization=regularization, armijo_params=armijo_params)

    print(f"Minimum at x = {x_min}, f(x) = {f_min}, iterations = {iterations}")

    # Plot the function and iterates
    x = np.linspace(-2, 2, 500)
    y = np.array([f(i) for i in x])  # Ensure consistent scalar values

    plt.figure()
    plt.plot(x, y, label="f(x) = x^4 + x^3 - x^2 - x")

    # Use colors to distinguish iterations
    colors = plt.cm.viridis(np.linspace(1, 0, len(iterates)))
    for idx, (xi, yi) in enumerate(zip(iterates, [f(i) for i in iterates])):
        plt.scatter(xi, yi, color=colors[idx], label=f"Iter {idx}" if idx == 0 else None)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Function Minimization with Iterates")
    plt.legend()
    plt.grid(True)
    plt.show()
