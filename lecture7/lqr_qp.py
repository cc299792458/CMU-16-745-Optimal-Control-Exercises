import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import block_diag, kron, eye, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

class LQR:
    def __init__(self, A, B, Q, R, Qn, x0, h, Tfinal):
        self.A = A
        self.B = B
        self.Q = csc_matrix(Q)
        self.R = csc_matrix(R)
        self.Qn = csc_matrix(Qn)
        self.x0 = x0
        self.h = h
        self.Tfinal = Tfinal
        self.N = int(Tfinal / h) + 1
        self.thist = np.linspace(0, Tfinal, self.N)

    def cost(self, xhist, uhist):
        cost = 0.5 * xhist[:, -1].T @ self.Qn @ xhist[:, -1]
        for k in range(self.N - 1):
            cost += 0.5 * xhist[:, k].T @ self.Q @ xhist[:, k] + 0.5 * uhist[k].T @ self.R @ uhist[k]
        return cost

    def solve(self):
        n, m = self.A.shape[0], self.B.shape[1]

        # Construct the block diagonal H matrix for the cost function
        H = block_diag(
            [self.R] + [block_diag([self.Q, self.R]) for _ in range(self.N - 2)] + [self.Qn],
            format="csc"
        )

        # Construct the equality constraint matrix C using lil_matrix for efficiency
        C = lil_matrix((n * (self.N - 1), (n + m) * (self.N - 1)))
        I_N_minus_1 = eye(self.N - 1, format="csc")
        block_B_minus_I = kron(I_N_minus_1, np.hstack((self.B, -np.eye(n))), format="lil")
        C[:, :] = block_B_minus_I

        for k in range(self.N - 2):
            start = k * n
            C[start : start + n, k * (n + m) : k * (n + m) + n] += self.A

        C = C.tocsc()  # Convert to csc_matrix for computations

        # Construct the d vector for the equality constraint
        d = np.zeros(C.shape[0])
        d[:n] = -self.A @ self.x0

        # Solve the KKT system
        zero_block = csc_matrix((C.shape[0], C.shape[0]))
        KKT_matrix = block_diag([H, zero_block], format="csc")
        KKT_rhs = np.hstack([np.zeros(H.shape[0]), d])

        try:
            solution = spsolve(KKT_matrix, KKT_rhs)
        except Exception as e:
            raise ValueError("KKT system could not be solved. Check constraints or matrix structure.") from e

        # Extract state and control trajectories
        z = solution[: H.shape[0]]  # states and controls [u0, x1, u1, ..., xN]
        Z = z.reshape(n + m, self.N - 1, order="F")
        xhist = np.hstack((self.x0.reshape(-1, 1), Z[m:, :]))
        uhist = Z[:m, :].flatten()

        return xhist, uhist

    def plot_results(self, xhist, uhist):
        plt.figure(figsize=(10, 6))
        plt.plot(self.thist, xhist[0, :], label="Position")
        plt.plot(self.thist, xhist[1, :], label="Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.legend()
        plt.title("State Trajectories")
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.thist[:-1], uhist, label="Control")
        plt.xlabel("Time (s)")
        plt.ylabel("Control Input")
        plt.legend()
        plt.title("Control Trajectory")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    # Define dynamics parameters
    h = 0.1  # time step
    A = np.array([[1, h],
                  [0, 1]])  # state transition matrix
    B = np.array([[0.5 * h ** 2],
                  [h]])  # control input matrix

    n = 2  # number of states
    m = 1  # number of controls
    Tfinal = 100.0  # final time

    # Define cost weights
    Q = np.eye(2)  # state cost
    R = 0.1 * np.eye(1)  # control cost
    Qn = np.eye(2)  # terminal state cost

    # Initial conditions
    x0 = np.array([1.0, 0])  # initial state

    # Solve LQR problem
    lqr = LQR(A, B, Q, R, Qn, x0, h, Tfinal)
    xhist, uhist = lqr.solve()
    lqr.plot_results(xhist, uhist)
