import numpy as np
import matplotlib.pyplot as plt

from lecture3.root_finding import iterative_solver  

def minimization(f, grad_f, hessian_f, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for minimizing a function using iterative_solver.

    Args:
        f (function): Objective function.
        grad_f (function): Gradient of the objective function.
        hessian_f (function): Hessian matrix of the objective function.
        x0 (float or np.array): Initial guess.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

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
        return hessian_f(x)  # The Hessian is the Jacobian of the gradient

    # Store the iterates
    iterates = [x0]

    # Use iterative_solver to minimize the function
    def recording_residual(x):
        iterates.append(x.item() if isinstance(x, np.ndarray) else x)
        return residual(x)

    x_min, _, iterations = iterative_solver(recording_residual, derivative=jacobian, x0=x0, tol=tol, max_iter=max_iter)

    return x_min.item() if isinstance(x_min, np.ndarray) else x_min, f(x_min), iterations, iterates

if __name__ == "__main__":
    # Define the objective function, gradient, and Hessian for x^4 + x^3 - x^2 - x
    f = lambda x: np.squeeze(x**4 + x**3 - x**2 - x) if isinstance(x, np.ndarray) else x**4 + x**3 - x**2 - x
    grad_f = lambda x: 4 * x**3 + 3 * x**2 - 2 * x - 1
    hessian_f = lambda x: np.array([[12 * x**2 + 6 * x - 2]])  # Return 2D Hessian for 1D problem

    # Initial guess
    x0 = 2.0 # -1.5, 0.0

    # Perform minimization
    x_min, f_min, iterations, iterates = minimization(f, grad_f, hessian_f, x0)

    print(f"Minimum at x = {x_min}, f(x) = {f_min}, iterations = {iterations}")

    # Plot the function and iterates
    x = np.linspace(-2, 2, 500)
    y = np.array([f(i) for i in x])  # Ensure consistent scalar values

    plt.figure()
    plt.plot(x, y, label="f(x) = x^4 + x^3 - x^2 - x")
    plt.scatter(iterates, [f(i) for i in iterates], color="red", label="Iterates")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Function Minimization with Iterates")
    plt.legend()
    plt.grid(True)
    plt.show()