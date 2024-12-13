"""In this file, we define the dynamics class and various numerical integration methods."""

import numpy as np

from lecture3.root_finding import iterative_solver

class Dynamics:
    def plot(self, trajectory, dt):
        """
        Placeholder for plotting the simulation results.

        Args:
            trajectory (np.array): The trajectory of the states.
            dt (float): Time step.
        """
        raise NotImplementedError("Plot method must be implemented by the subclass.")
    """
    Base class for system dynamics.
    All specific dynamics models should inherit from this class and implement the dynamics_equation method.
    """
    
    def dynamics_equation(self, state, control):
        """
        Placeholder for the specific system dynamics equation.

        Args:
            state (np.array): Current state of the system.
            control (np.array): Control input to the system.

        Returns:
            np.array: Derivative of the state.
        """
        raise NotImplementedError("Dynamics equation must be implemented by the subclass.")
    
    def jacobian_dynamics_equation(self, state, control):
        """
        Placeholder for the Jacobian of the system dynamics equation.

        Args:
            state (np.array): Current state of the system.
            control (np.array): Control input to the system.

        Returns:
            np.array: Jacobian matrix of the system dynamics equation.
        """
        raise NotImplementedError("Jacobian of dynamics equation must be implemented by the subclass.")

    def forward_euler_step(self, state, control, dt):
        """
        Perform a single forward Euler integration step.

        Args:
            state (np.array): Current state.
            control (np.array): Control input.
            dt (float): Time step.

        Returns:
            np.array: Next state.
        """
        return state + self.dynamics_equation(state, control) * dt

    def backward_euler_step(self, state, control, dt, tol=1e-6, max_iter=100):
        """
        Perform a single backward Euler integration step using Newton's method.

        Args:
            state (np.array): Current state.
            control (np.array): Control input.
            dt (float): Time step.
            tol (float): Convergence tolerance for Newton's method.
            max_iter (int): Maximum number of iterations.

        Returns:
            np.array: Next state.
        """
        # Define the residual function for backward Euler
        def residual(next_state):
            return next_state - state - dt * self.dynamics_equation(next_state, control)

        # Define the Jacobian of the residual
        def jacobian(next_state):
            return np.eye(len(state)) - dt * self.jacobian_dynamics_equation(next_state, control)

        # Use iterative_solver to solve for the next state
        next_state, _, _ = iterative_solver(residual, derivative=jacobian, x0=state, tol=tol, max_iter=max_iter)

        return next_state

    def rk4_step(self, state, control, dt):
        """
        Perform a single RK4 integration step.

        Args:
            state (np.array): Current state.
            control (np.array): Control input.
            dt (float): Time step.

        Returns:
            np.array: Next state.
        """
        k1 = self.dynamics_equation(state, control)
        k2 = self.dynamics_equation(state + 0.5 * dt * k1, control)
        k3 = self.dynamics_equation(state + 0.5 * dt * k2, control)
        k4 = self.dynamics_equation(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, initial_state, control, dt, steps, method):
        """
        Simulate the dynamics using the specified numerical method.

        Args:
            initial_state (np.array): Initial state of the system.
            control (np.array): Control input to the system.
            dt (float): Time step.
            steps (int): Number of simulation steps.
            method (str): Integration method ('forward_euler', 'backward_euler', 'rk4').

        Returns:
            np.array: Trajectory of states.
        """
        state = initial_state
        trajectory = [state]
        
        for _ in range(steps):
            if method == 'forward_euler':
                state = self.forward_euler_step(state, control, dt)
            elif method == 'backward_euler':
                state = self.backward_euler_step(state, control, dt)
            elif method == 'rk4':
                state = self.rk4_step(state, control, dt)
            else:
                raise ValueError(f"Unknown integration method: {method}")

            trajectory.append(state)

        return np.array(trajectory)