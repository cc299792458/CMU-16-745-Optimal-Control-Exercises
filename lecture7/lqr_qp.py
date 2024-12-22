import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class LQR:
    def __init__(self, A, B, Q, R, QN, x0, h, T):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.QN = QN
        self.x0 = x0
        self.h = h
        self.T = T
        self.N = int(T / h) + 1
        self.t_hist = np.linspace(0, T, self.N)

    def cost(self, x_hist, u_hist):
        # LQR cost
        val = 0.5 * x_hist[:, -1].T @ self.QN @ x_hist[:, -1]
        for k in range(self.N - 1):
            val += 0.5 * (x_hist[:, k].T @ self.Q @ x_hist[:, k]
                          + u_hist[k].T @ self.R @ u_hist[k])
        return val

    def solve(self):
        # Dimensions
        n, m = self.A.shape[0], self.B.shape[1]

        # Build block-diagonal H
        H_blocks = [self.R]
        for _ in range(self.N - 2):
            H_blocks.append(block_diag(self.Q, self.R))
        H_blocks.append(self.QN)
        H = block_diag(*H_blocks)

        # Build constraint C and d
        C = np.zeros((n * (self.N - 1), (n + m) * (self.N - 1)))
        I_Nm1 = np.eye(self.N - 1)
        block_BI = np.kron(I_Nm1, np.hstack((self.B, -np.eye(n))))
        C[:, :] = block_BI
        for k in range(self.N - 2):
            r = (k + 1) * n
            c = (k + 1) * m + k * n
            C[r : r + n, c : c + n] += self.A
        d = np.zeros(C.shape[0])
        d[:n] = -self.A @ self.x0

        # Solve KKT
        zero_block = np.zeros((C.shape[0], C.shape[0]))
        KKT_top = np.hstack((H, C.T))
        KKT_bottom = np.hstack((C, zero_block))
        KKT_mat = np.vstack((KKT_top, KKT_bottom))
        KKT_rhs = np.concatenate((np.zeros(H.shape[0]), d))
        try:
            sol = np.linalg.solve(KKT_mat, KKT_rhs)
        except np.linalg.LinAlgError as e:
            raise ValueError("KKT solve failed.") from e

        # Extract x, u
        z = sol[: (n + m) * (self.N - 1)]
        Z = z.reshape((n + m, self.N - 1), order="F")
        x_hist = np.hstack((self.x0.reshape(-1, 1), Z[m:, :]))
        u_hist = Z[:m, :].flatten()

        return x_hist, u_hist

    def plot_results(self, x_hist, u_hist):
        plt.figure()
        plt.plot(self.t_hist, x_hist[0, :], label="Position")
        plt.plot(self.t_hist, x_hist[1, :], label="Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(self.t_hist[:-1], u_hist, label="Control")
        plt.xlabel("Time (s)")
        plt.ylabel("Control")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Parameters
    h = 0.1
    A = np.array([[1, h],
                  [0, 1]])
    B = np.array([[0.5 * h**2],
                  [h]])
    n = 2
    m = 1
    T = 5.0
    Q = np.eye(n)
    R = 0.1 * np.eye(m)
    QN = np.eye(n)
    x0 = np.array([1.0, 0.0])

    # Solve
    lqr = LQR(A, B, Q, R, QN, x0, h, T)
    x_hist, u_hist = lqr.solve()
    lqr.plot_results(x_hist, u_hist)
