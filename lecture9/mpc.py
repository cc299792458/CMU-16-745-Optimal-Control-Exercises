import numpy as np
import matplotlib.pyplot as plt
from osqp import OSQP
from scipy.sparse import csc_matrix
from scipy.linalg import solve_discrete_are

# Model parameters
g = 9.81  # m/s^2
m = 1.0   # kg
ell = 0.3  # meters
J = 0.2 * m * ell**2

# Thrust limits
umin = np.array([0.2 * m * g, 0.2 * m * g])
umax = np.array([0.6 * m * g, 0.6 * m * g])

h = 0.05  # time step (20 Hz)

# Planar Quadrotor Dynamics
def quad_dynamics(x, u):
    theta = x[2]
    
    x_ddot = (1 / m) * (u[0] + u[1]) * np.sin(theta)
    y_ddot = (1 / m) * (u[0] + u[1]) * np.cos(theta) - g
    theta_ddot = (1 / J) * (ell / 2) * (u[1] - u[0])
    
    return np.hstack((x[3:], [x_ddot, y_ddot, theta_ddot]))

def quad_dynamics_rk4(x, u):
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5 * h * f1, u)
    f3 = quad_dynamics(x + 0.5 * h * f2, u)
    f4 = quad_dynamics(x + h * f3, u)
    return x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

# Linearized dynamics for hovering
x_hover = np.zeros(6)
u_hover = np.array([0.5 * m * g, 0.5 * m * g])

A = np.zeros((6, 6))  # Placeholder for Jacobian of dynamics w.r.t. state
B = np.zeros((6, 2))  # Placeholder for Jacobian of dynamics w.r.t. input

for i in range(6):
    dx = np.zeros(6)
    dx[i] = 1e-5
    A[:, i] = (quad_dynamics_rk4(x_hover + dx, u_hover) - quad_dynamics_rk4(x_hover, u_hover)) / 1e-5

for i in range(2):
    du = np.zeros(2)
    du[i] = 1e-5
    B[:, i] = (quad_dynamics_rk4(x_hover, u_hover + du) - quad_dynamics_rk4(x_hover, u_hover)) / 1e-5

# Cost weights
Q = np.eye(6)
R = 0.01 * np.eye(2)
Qn = np.eye(6)

# LQR hover controller
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

def lqr_controller(t, x, K, x_ref):
    return u_hover - K @ (x - x_ref)

# MPC setup
Nh = 20  # Horizon
Nx = 6
Nu = 2

# Cost matrices
H = np.zeros((Nh * (Nx + Nu), Nh * (Nx + Nu)))
b = np.zeros(Nh * (Nx + Nu))

for i in range(Nh - 1):
    H[i * (Nx + Nu):(i + 1) * (Nx + Nu), i * (Nx + Nu):(i + 1) * (Nx + Nu)] = np.block([
        [R, np.zeros((Nu, Nx))],
        [np.zeros((Nx, Nu)), Q]
    ])
H[(Nh - 1) * (Nx + Nu):, (Nh - 1) * (Nx + Nu):] = np.block([
    [R, np.zeros((Nu, Nx))],
    [np.zeros((Nx, Nu)), P]
])

# Dynamics and constraints matrices
C = np.zeros((Nh * Nx, Nh * (Nx + Nu)))
for i in range(Nh):
    if i == 0:
        # First step: Initial control input and state
        C[:Nx, :Nu] = B  # B for u_0
        C[:Nx, Nu:(Nu + Nx)] = -np.eye(Nx)  # -I for x_1
    else:
        # Subsequent steps: Dynamics relationship
        C[i * Nx:(i + 1) * Nx, (i - 1) * (Nx + Nu) + Nu:i* (Nx + Nu)] = A  # A for x_k
        C[i * Nx:(i + 1) * Nx, i * (Nx + Nu):i * (Nx + Nu) + Nu] = B  # B for u_k
        C[i * Nx:(i + 1) * Nx, i * (Nx + Nu) + Nu:(i + 1) * (Nx + Nu)] = -np.eye(Nx)  # -I for x_{k+1}

U = np.zeros((Nh * Nu, Nh * (Nx + Nu)))
for i in range(Nh):
    U[i * Nu:(i + 1) * Nu, i * (Nx + Nu):i * (Nx + Nu) + Nu] = np.eye(Nu)

THETA = np.zeros((Nh, Nh * (Nx + Nu)))
for i in range(Nh):
    THETA[i, i * (Nx + Nu) + 4] = 1

D = np.vstack([C, U, THETA])
lb = np.hstack([np.zeros(Nh * Nx), np.tile(umin - u_hover, Nh), -0.2 * np.ones(Nh)])
ub = np.hstack([np.zeros(Nh * Nx), np.tile(umax - u_hover, Nh), 0.2 * np.ones(Nh)])

# Convert matrices to sparse format
H_sparse = csc_matrix(H)
D_sparse = csc_matrix(D)

# OSQP solver setup
prob = OSQP()
prob.setup(P=H_sparse, q=b, A=D_sparse, l=lb, u=ub, verbose=False, eps_abs=1e-8, eps_rel=1e-8, polish=True)

def mpc_controller(t, x, x_ref):
    # Update constraints
    lb[:Nx] = -A @ x
    ub[:Nx] = -A @ x

    for j in range(Nh - 1):
        b[(Nu + j * (Nx + Nu)):(j + 1) * (Nx + Nu)] = -Q @ x_ref
    b[(Nu + (Nh - 1) * (Nx + Nu)):Nh * (Nx + Nu)] = -P @ x_ref

    prob.update(q=b, l=lb, u=ub)
    
    # Solve QP
    results = prob.solve()
    delta_u = results.x[:Nu]

    return u_hover + delta_u

def closed_loop(x0, controller, N):
    xhist = np.zeros((len(x0), N))
    uhist = np.zeros((2, N - 1))
    xhist[:, 0] = x0

    for k in range(N - 1):
        u = controller(k, xhist[:, k])
        uhist[:, k] = np.clip(u, umin, umax)
        xhist[:, k + 1] = quad_dynamics_rk4(xhist[:, k], uhist[:, k])

    return xhist, uhist

# Simulation
x_ref = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
x0 = np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0])
T = 6.0
Nt = int(T / h) + 1
thist = np.linspace(0, T, Nt)

xhist1, uhist1 = closed_loop(x0, lambda t, x: lqr_controller(t, x, K, x_ref), Nt)
xhist2, uhist2 = closed_loop(x0, lambda t, x: mpc_controller(t, x, x_ref), Nt)

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
