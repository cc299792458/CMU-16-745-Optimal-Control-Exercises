import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from osqp import OSQP
from scipy.sparse import csc_matrix
from scipy.linalg import solve_discrete_are

# Planar Quadrotor Dynamics
def quad_dynamics(x, u):
    theta = x[2]
    
    x_ddot = -(1 / m) * (u[0] + u[1]) * np.sin(theta)
    y_ddot = (1 / m) * (u[0] + u[1]) * np.cos(theta) - g
    theta_ddot = (1 / J) * (ell / 2) * (u[1] - u[0])
    
    return np.hstack((x[3:], [x_ddot, y_ddot, theta_ddot]))

def quad_dynamics_rk4(x, u):
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5 * h * f1, u)
    f3 = quad_dynamics(x + 0.5 * h * f2, u)
    f4 = quad_dynamics(x + h * f3, u)
    return x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

class LQR:
    def __init__(self, A, B, Q, R, u_hover):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.u_hover = u_hover
        self.P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ self.P @ B) @ (B.T @ self.P @ A)

    def control(self, x, x_ref):
        return self.u_hover - self.K @ (x - x_ref)

class MPC:
    def __init__(self, A, B, Q, R, P, u_hover, umin, umax, Nh):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.u_hover = u_hover
        self.umin = umin
        self.umax = umax
        self.Nh = Nh
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]

        # Cost matrices
        self.H = np.zeros((Nh * (self.Nx + self.Nu), Nh * (self.Nx + self.Nu)))
        for i in range(Nh - 1):
            self.H[i * (self.Nx + self.Nu):(i + 1) * (self.Nx + self.Nu),
                    i * (self.Nx + self.Nu):(i + 1) * (self.Nx + self.Nu)] = np.block([
                [R, np.zeros((self.Nu, self.Nx))],
                [np.zeros((self.Nx, self.Nu)), Q]
            ])
        self.H[(Nh - 1) * (self.Nx + self.Nu):, (Nh - 1) * (self.Nx + self.Nu):] = np.block([
            [R, np.zeros((self.Nu, self.Nx))],
            [np.zeros((self.Nx, self.Nu)), P]
        ])

        # Dynamics and constraints matrices
        self.C = np.zeros((Nh * self.Nx, Nh * (self.Nx + self.Nu)))
        for i in range(Nh):
            if i == 0:
                self.C[:self.Nx, :self.Nu] = B  # B for u_0
                self.C[:self.Nx, self.Nu:(self.Nu + self.Nx)] = -np.eye(self.Nx)  # -I for x_1
            else:
                self.C[i * self.Nx:(i + 1) * self.Nx,
                        (i - 1) * (self.Nx + self.Nu) + self.Nu:i * (self.Nx + self.Nu)] = A  # A for x_k
                self.C[i * self.Nx:(i + 1) * self.Nx,
                        i * (self.Nx + self.Nu):i * (self.Nx + self.Nu) + self.Nu] = B  # B for u_k
                self.C[i * self.Nx:(i + 1) * self.Nx,
                        i * (self.Nx + self.Nu) + self.Nu:(i + 1) * (self.Nx + self.Nu)] = -np.eye(self.Nx)  # -I for x_{k+1}

        self.U = np.zeros((Nh * self.Nu, Nh * (self.Nx + self.Nu)))
        for i in range(Nh):
            self.U[i * self.Nu:(i + 1) * self.Nu, i * (self.Nx + self.Nu):i * (self.Nx + self.Nu) + self.Nu] = np.eye(self.Nu)

        self.THETA = np.zeros((Nh, Nh * (self.Nx + self.Nu)))
        for i in range(Nh):
            self.THETA[i, i * (self.Nx + self.Nu) + 4] = 1

        self.D = np.vstack([self.C, self.U, self.THETA])
        self.lb = np.hstack([np.zeros(Nh * self.Nx), np.tile(umin - u_hover, Nh), -0.2 * np.ones(Nh)])
        self.ub = np.hstack([np.zeros(Nh * self.Nx), np.tile(umax - u_hover, Nh), 0.2 * np.ones(Nh)])

        # Convert matrices to sparse format
        self.H_sparse = csc_matrix(self.H)
        self.D_sparse = csc_matrix(self.D)

        # OSQP solver setup
        self.prob = OSQP()
        self.prob.setup(P=self.H_sparse, q=np.zeros(Nh * (self.Nx + self.Nu)), A=self.D_sparse, l=self.lb, u=self.ub, verbose=False, eps_abs=1e-8, eps_rel=1e-8, polish=True)

    def control(self, x, x_ref):
        # Update constraints
        self.lb[:self.Nx] = -self.A @ x
        self.ub[:self.Nx] = -self.A @ x

        b = np.zeros(self.Nh * (self.Nx + self.Nu))
        for j in range(self.Nh - 1):
            b[(self.Nu + j * (self.Nx + self.Nu)):(j + 1) * (self.Nx + self.Nu)] = -self.Q @ x_ref
        b[(self.Nu + (self.Nh - 1) * (self.Nx + self.Nu)):self.Nh * (self.Nx + self.Nu)] = -self.P @ x_ref

        self.prob.update(q=b, l=self.lb, u=self.ub)
        results = self.prob.solve()
        delta_u = results.x[:self.Nu]

        return self.u_hover + delta_u

def rollout(x0, controller, N):
    xhist = np.zeros((len(x0), N))
    uhist = np.zeros((2, N - 1))
    xhist[:, 0] = x0

    for k in range(N - 1):
        u = controller(k, xhist[:, k])
        # There might be slight numerical overflow.
        uhist[:, k] = np.clip(u, umin, umax)
        xhist[:, k + 1] = quad_dynamics_rk4(xhist[:, k], uhist[:, k])

    return xhist, uhist

def animate_trajectory(xhist, title, save_path=None):
    """
    Creates an animation of the trajectory using x and y from xhist.

    Args:
        xhist: ndarray of shape (Nx, Nt) representing the trajectory states over time.
        title: str, title of the animation.
        save_path: str or None, path to save the animation file, or None to skip saving.
    """
    # Ensure xhist is valid
    assert xhist.ndim == 2, "xhist must be a 2D NumPy array"
    assert xhist.shape[0] >= 3, "xhist must have at least 3 rows for x, y, and theta"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-12, 12)  # Adjust as per your trajectory range
    ax.set_ylim(0, 15)
    ax.set_title(title)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid(True)
    
    # Initialize point and orientation line
    point, = ax.plot([], [], 'bo', label="Quadrotor Position")
    orientation_line, = ax.plot([], [], 'r-', label="Orientation")
    ax.legend(loc="upper right")

    def init():
        point.set_data([], [])
        orientation_line.set_data([], [])
        return point, orientation_line

    def update(frame):
        if frame >= xhist.shape[1]:
            raise IndexError("Frame index out of range for xhist")

        # Extract x, y, and theta
        x, y, theta = xhist[0, frame], xhist[1, frame], xhist[2, frame]
        point.set_data([x], [y])  # Ensure x, y are sequences
        orientation_line.set_data(
            [x, x + 0.5 * np.cos(theta)],  # Line endpoint for orientation
            [y, y + 0.5 * np.sin(theta)]
        )
        return point, orientation_line

    ani = animation.FuncAnimation(
        fig, update, frames=xhist.shape[1],
        init_func=init, blit=True, interval=50
    )

    if save_path:
        ani.save(save_path, writer='imagemagick', fps=20)
    else:
        plt.show()

if __name__ == '__main__':
    # Log path
    log_dir = os.path.dirname(os.path.abspath(__file__))

    # Model parameters
    g = 9.81  # m/s^2
    m = 1.0   # kg
    ell = 0.3  # meters
    J = 0.2 * m * ell**2

    # Thrust limits
    umin = np.array([0.2 * m * g, 0.2 * m * g])
    umax = np.array([0.6 * m * g, 0.6 * m * g])

    h = 0.05  # time step (20 Hz)
    Nh = 20  # Horizon

    # Linearized dynamics for hovering
    x_hover = np.zeros(6)
    u_hover = np.array([0.5 * m * g, 0.5 * m * g])

    A = np.zeros((6, 6))  # Placeholder for Jacobian of dynamics w.r.t. state
    B = np.zeros((6, 2))  # Placeholder for Jacobian of dynamics w.r.t. input

    # NOTE: Discrete-time system
    for i in range(6):
        dx = np.zeros(6)
        dx[i] = 1e-5
        A[:, i] = (quad_dynamics_rk4(x_hover + dx, u_hover) - quad_dynamics_rk4(x_hover, u_hover)) / dx[i]

    for i in range(2):
        du = np.zeros(2)
        du[i] = 1e-5
        B[:, i] = (quad_dynamics_rk4(x_hover, u_hover + du) - quad_dynamics_rk4(x_hover, u_hover)) / du[i]

    # Cost weights
    Q = np.eye(6)
    R = 0.01 * np.eye(2)
    P = solve_discrete_are(A, B, Q, R)

    # Simulation
    x_ref = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    x0 = np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    T = 6.0
    Nt = int(T / h) + 1
    thist = np.linspace(0, T, Nt)

    lqr = LQR(A, B, Q, R, u_hover)
    mpc = MPC(A, B, Q, R, P, u_hover, umin, umax, Nh)

    xhist1, uhist1 = rollout(x0, lambda t, x: lqr.control(x, x_ref), Nt)
    xhist2, uhist2 = rollout(x0, lambda t, x: mpc.control(x, x_ref), Nt)

    # Plot each state dimension in separate subplots
    fig, axes = plt.subplots(xhist1.shape[0], 1, figsize=(10, 12), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(thist, xhist1[i, :], label=f"State {i+1} (LQR)")
        ax.plot(thist, xhist2[i, :], label=f"State {i+1} (MPC)")
        ax.set_ylabel(f"State {i+1}")
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    # Generate animations for LQR and MPC
    animate_trajectory(xhist1, title="LQR Trajectory", save_path=log_dir + "/lqr_trajectory.gif")
    animate_trajectory(xhist2, title="MPC Trajectory", save_path=log_dir + "/mpc_trajectory.gif")