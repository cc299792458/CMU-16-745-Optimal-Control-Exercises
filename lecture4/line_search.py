import numpy as np

def armijo_step(f, grad_f, x, p, alpha_0=1.0, rho=0.5, c=1e-4):
    """
    Perform Armijo Rule line search to determine step size.

    Args:
        f (function): Objective function.
        grad_f (function): Gradient of the objective function.
        x (np.array or float): Current point.
        p (np.array or float): Descent direction.
        alpha_0 (float): Initial step size (default: 1.0).
        rho (float): Reduction factor (default: 0.5).
        c (float): Sufficient decrease constant (default: 1e-4).

    Returns:
        float: Step size satisfying Armijo condition.
    """
    alpha = alpha_0
    fx = f(x)
    grad_dot_p = np.dot(grad_f(x), p) if isinstance(x, np.ndarray) else grad_f(x) * p

    # Perform backtracking until the Armijo condition is satisfied
    while f(x + alpha * p) > fx + c * alpha * grad_dot_p:
        alpha *= rho

    return alpha
