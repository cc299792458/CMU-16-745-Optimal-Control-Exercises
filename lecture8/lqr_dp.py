import numpy as np
import matplotlib.pyplot as plt

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
        cost = 0.5 * x_hist[:, -1].T @ self.QN @ x_hist[:, -1]
        for k in range(self.N - 1):
            cost += 0.5 * (x_hist[:, k].T @ self.Q @ x_hist[:, k] + u_hist[:, k].T @ self.R @ u_hist[:, k])
        return cost

    def solve(self):
        n = self.A.shape[0]
        m = self.B.shape[1]

        P = np.zeros((n, n, self.N))
        K = np.zeros((m, n, self.N - 1))
        P[:, :, -1] = self.QN

        # Backward Riccati recursion
        for k in range(self.N - 2, -1, -1):
            K[:, :, k] = np.linalg.inv(self.R + self.B.T @ P[:, :, k + 1] @ self.B) @ (self.B.T @ P[:, :, k + 1] @ self.A)
            P[:, :, k] = (
                self.Q + K[:, :, k].T @ self.R @ K[:, :, k]
                + (self.A - self.B @ K[:, :, k]).T @ P[:, :, k + 1] @ (self.A - self.B @ K[:, :, k])
            )

        # Forward rollout
        x_hist = np.zeros((n, self.N))
        u_hist = np.zeros((m, self.N - 1))
        x_hist[:, 0] = self.x0

        for k in range(self.N - 1):
            u_hist[:, k] = -K[:, :, k] @ x_hist[:, k]
            x_hist[:, k + 1] = self.A @ x_hist[:, k] + self.B @ u_hist[:, k]

        return x_hist, u_hist, K

    def plot_results(self, x_hist, u_hist):
        plt.figure()
        plt.plot(self.t_hist, x_hist[0, :], label="Position (DP)")
        plt.plot(self.t_hist, x_hist[1, :], label="Velocity (DP)")
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(self.t_hist[:-1], u_hist[0, :], label="Control (DP)")
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

    # Solve LQR-DP
    lqr_dp = LQR(A, B, Q, R, QN, x0, h, T)
    x_hist, u_hist, K = lqr_dp.solve()

    # Plot results
    lqr_dp.plot_results(x_hist, u_hist)

    # Print cost
    total_cost = lqr_dp.cost(x_hist, u_hist)
    print(f"Total cost: {total_cost}")
