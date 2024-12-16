import numpy as np
import matplotlib.pyplot as plt

from lecture4.line_search import armijo_step

def iterative_solver(func, derivative=None, x0=None, tol=1e-8, max_iter=100, armijo_params=None):
    """
    General iterative solver for fixed-point iteration or Newton's method.
    Supports both single-variable and multi-variable cases.

    Args:
        func (function): Target function. For fixed-point iteration, g(x). For Newton's method, F(x).
        derivative (function or None): Derivative or Jacobian function. If None, fixed-point iteration is used.
        x0 (float or np.array): Initial guess (scalar or array).
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        armijo_params (dict or None): Parameters for Armijo Rule (alpha_0, rho, c). If None, no Armijo step is applied.

    Returns:
        np.array or float: Approximation of the fixed point or root.
        list: Residuals at each iteration.
        int: Number of iterations performed.
    """
    x = np.array(x0, dtype=float) if not np.isscalar(x0) else x0  # Handle scalar or array input
    residuals = []

    for i in range(max_iter):
        Fx = np.array(func(x, True))  # Evaluate the function and ensure it's a NumPy array

        if derivative is not None:  # Newton's method
            D = np.array(derivative(x))  # Compute Derivative or Jacobian and ensure it's a NumPy array
            if np.isscalar(x) or (isinstance(x, np.ndarray) and x.shape == ()):  # Single-variable case
                delta = -Fx / D
            else:  # Multi-variable case
                delta = np.linalg.solve(D, -Fx)
            delta = delta.item() if np.isscalar(x) else delta.reshape(x.shape)
            alpha = armijo_step(func, lambda _: Fx, x, delta, **armijo_params) if armijo_params is not None else 1  # Apply armijo rule
            x_new = x + alpha * delta
        else:  # Fixed-point iteration
            x_new = Fx

        residual = np.linalg.norm(x_new - x) if not np.isscalar(x) else abs(x_new - x)
        residuals.append(residual)

        # Check convergence
        if residual < tol:
            return x_new, residuals, i + 1

        x = x_new

    # Return result without raising an error if not converged
    return x, residuals, max_iter

# Example for fixed-point iteration
def fixed_point_example():
    # Single-variable fixed-point function
    g = lambda x: (x + 1) ** 0.5  # Fixed-point function
    x0 = 2.0  # Initial guess
    solution, residuals, iterations = iterative_solver(g, x0=x0)
    print(f"Fixed Point: {solution}, Residuals: {residuals}, Iterations: {iterations}")
    return residuals

# Example for Newton's method
def newton_example():
    # Single-variable equation: x^2 - x - 1 = 0
    F = lambda x: x**2 - x - 1  # Function
    dF = lambda x: 2 * x - 1    # Derivative
    x0 = 2.0  # Initial guess
    solution, residuals, iterations = iterative_solver(F, derivative=dF, x0=x0)
    print(f"Root: {solution}, Residuals: {residuals}, Iterations: {iterations}")
    return residuals

if __name__ == "__main__":
    print("Fixed-Point Iteration Example:")
    fp_residuals = fixed_point_example()

    print("\nNewton's Method Example:")
    newton_residuals = newton_example()

    # Plotting residuals
    plt.figure()
    plt.plot(range(len(fp_residuals)), fp_residuals, label="Fixed-Point Iteration")
    plt.plot(range(len(newton_residuals)), newton_residuals, label="Newton's Method")
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Residual")
    plt.title("Residual vs Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()
