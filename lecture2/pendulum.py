import numpy as np
import matplotlib.pyplot as plt

from lecture2.dynamics import Dynamics

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
    
    def jacobian_dynamics_equation(self, state, control=None):
        """
        Compute the Jacobian matrix of the dynamics equation for the pendulum system.

        The Jacobian matrix represents the partial derivatives of the dynamics with respect
        to the state variables (theta, omega). It is used in Newton's method for solving
        implicit integration schemes like Backward Euler.

        Args:
            state (np.array): Current state [theta, omega].
            control (np.array): Control input (not used in simple pendulum).

        Returns:
            np.array: 2x2 Jacobian matrix.
                    [[∂(dtheta/dt)/∂theta, ∂(dtheta/dt)/∂omega],
                    [∂(domega/dt)/∂theta, ∂(domega/dt)/∂omega]].
        """
        theta, omega = state
        return np.array([
            [0, 1],
            [-(self.g / self.l) * np.cos(theta), 0]
        ])

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
    steps = 10000

    pendulum = PendulumDynamics()
    initial_state = np.array([np.pi / 4, 0.0])  # Initial state: 45 degrees, 0 angular velocity

    # Simulate using RK4
    trajectory = pendulum.simulate(initial_state, None, dt, steps, method='backward_euler')

    # Plot the results
    pendulum.plot(trajectory, dt)
