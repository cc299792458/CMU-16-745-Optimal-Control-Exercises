"""In this file, we define the dynamics class and various numerical integration methods."""

import numpy as np
import matplotlib.pyplot as plt

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
        next_state = state.copy()  # Initial guess

        for _ in range(max_iter):
            # Compute the residual g(next_state) = next_state - state - dt * f(next_state)
            residual = next_state - state - dt * self.dynamics_equation(next_state, control)

            # Use analytical Jacobian
            jacobian = np.eye(len(state)) - dt * self.jacobian_dynamics_equation(next_state, control)

            # Newton's update
            delta = np.linalg.solve(jacobian, -residual)
            next_state += delta

            # Check for convergence
            if np.linalg.norm(delta) < tol:
                return next_state

        raise ValueError("Backward Euler did not converge within the maximum number of iterations.")

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

class PendulumDynamics(Dynamics):
    """
    Dynamics model for a simple pendulum.
    """
    def __init__(self, g=9.81, l=1.0):
        self.g = g
        self.l = l

    def dynamics_equation(self, state, control=None):
        """
        Compute the dynamics of a simple pendulum.

        Args:
            state (np.array): Current state [theta, omega].
            control (np.array): Control input (not used in simple pendulum).

        Returns:
            np.array: [dtheta/dt, domega/dt].
        """
        theta, omega = state
        dtheta_dt = omega
        domega_dt = -(self.g / self.l) * np.sin(theta)
        return np.array([dtheta_dt, domega_dt])

    def simulate(self, initial_state, control, dt, steps, method):
        """
        Simulate the pendulum dynamics and return the trajectory.

        Args:
            initial_state (np.array): Initial state of the system.
            control (np.array): Control input to the system.
            dt (float): Time step.
            steps (int): Number of simulation steps.
            method (str): Integration method ('forward_euler', 'backward_euler', 'rk4').

        Returns:
            np.array: Trajectory of states.
        """
        return super().simulate(initial_state, control, dt, steps, method)

    def plot(self, trajectory, dt):
        """
        Plot the pendulum simulation trajectory.

        Args:
            trajectory (np.array): The trajectory of the states.
            dt (float): Time step.
        """
        time = np.arange(0, len(trajectory) * dt, dt)
        theta = trajectory[:, 0]
        omega = trajectory[:, 1]

        plt.figure(figsize=(10, 5))

        plt.subplot(2, 1, 1)
        plt.plot(time, theta)
        plt.title('Pendulum Angle (Theta)')
        plt.xlabel('Time (s)')
        plt.ylabel('Theta (rad)')

        plt.subplot(2, 1, 2)
        plt.plot(time, omega)
        plt.title('Pendulum Angular Velocity (Omega)')
        plt.xlabel('Time (s)')
        plt.ylabel('Omega (rad/s)')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dt = 0.01
    steps = 1000

    pendulum = PendulumDynamics()
    initial_state = np.array([np.pi / 4, 0.0])  # Initial state: 45 degrees, 0 angular velocity

    # Simulate using RK4
    trajectory = pendulum.simulate(initial_state, None, dt, steps, method='rk4')

    # Plot the results
    pendulum.plot(trajectory, dt)
