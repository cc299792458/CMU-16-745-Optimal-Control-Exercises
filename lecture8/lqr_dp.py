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
        self.thist = np.linspace(0, T, self.N)

    def cost(self, xhist, uhist):
        cost = 0.5 * xhist[:, -1].T @ self.QN @ xhist[:, -1]
        for k in range(self.N - 1):
            cost += 0.5 * (xhist[:, k].T @ self.Q @ xhist[:, k] + uhist[:, k].T @ self.R @ uhist[:, k])
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
        xhist = np.zeros((n, self.N))
        uhist = np.zeros((m, self.N - 1))
        xhist[:, 0] = self.x0

        for k in range(self.N - 1):
            uhist[:, k] = -K[:, :, k] @ xhist[:, k]
            xhist[:, k + 1] = self.A @ xhist[:, k] + self.B @ uhist[:, k]

        return xhist, uhist, K

    def plot_results(self, xhist, uhist):
        plt.figure()
        plt.plot(self.thist, xhist[0, :], label="Position (DP)")
        plt.plot(self.thist, xhist[1, :], label="Velocity (DP)")
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(self.thist[:-1], uhist[0, :], label="Control (DP)")
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
    xhist, uhist, K = lqr_dp.solve()

    # Plot results
    lqr_dp.plot_results(xhist, uhist)

    # Print cost
    total_cost = lqr_dp.cost(xhist, uhist)
    print(f"Total cost: {total_cost}")
