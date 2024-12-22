import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

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
            cost += 0.5 * (xhist[:, k].T @ self.Q @ xhist[:, k] + uhist[:, k, 0].T @ self.R @ uhist[:, k, 0])
        return cost

    def solve(self):
        n = self.A.shape[0]
        m = self.B.shape[1]

        P = np.zeros((n, n, self.N))
        K = np.zeros((m, n, self.N - 1))

        P[:, :, -1] = self.QN

        for k in range(self.N - 2, -1, -1):
            K[:, :, k] = np.linalg.inv(self.R + self.B.T @ P[:, :, k + 1] @ self.B) @ (self.B.T @ P[:, :, k + 1] @ self.A)
            P[:, :, k] = self.Q + self.A.T @ P[:, :, k + 1] @ (self.A - self.B @ K[:, :, k])

        xhist = np.zeros((n, self.N))
        uhist = np.zeros((m, self.N - 1, 1))
        xhist[:, 0] = self.x0

        for k in range(self.N - 1):
            uhist[:, k, 0] = -K[:, :, k] @ xhist[:, k]
            xhist[:, k + 1] = self.A @ xhist[:, k] + self.B @ uhist[:, k, 0]

        return xhist, uhist, K

    def plot_results(self, xhist, uhist, xhist_inf=None, uhist_inf=None):
        plt.figure()
        plt.plot(self.thist, xhist[0, :], label="Position (Finite K)")
        plt.plot(self.thist, xhist[1, :], label="Velocity (Finite K)")
        if xhist_inf is not None:
            plt.plot(self.thist, xhist_inf[0, :], label="Position (Infinite K)", linestyle='--')
            plt.plot(self.thist, xhist_inf[1, :], label="Velocity (Infinite K)", linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(self.thist[:-1], uhist[0, :, 0], label="Control (Finite K)")
        if uhist_inf is not None:
            plt.plot(self.thist[:-1], uhist_inf[0, :, 0], label="Control (Infinite K)", linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Control")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
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

    lqr = LQR(A, B, Q, R, QN, x0, h, T)
    xhist, uhist, K = lqr.solve()

    # Compute infinite-horizon solution
    P_inf = solve_discrete_are(A, B, Q, R)
    K_inf = np.linalg.inv(R + B.T @ P_inf @ B) @ (B.T @ P_inf @ A)

    # Forward rollout with constant K_inf
    xhist_inf = np.zeros((n, lqr.N))
    uhist_inf = np.zeros((m, lqr.N - 1, 1))
    xhist_inf[:, 0] = x0

    for k in range(lqr.N - 1):
        uhist_inf[:, k, 0] = -K_inf @ xhist_inf[:, k]
        xhist_inf[:, k + 1] = A @ xhist_inf[:, k] + B @ uhist_inf[:, k, 0]

    # Plot results
    lqr.plot_results(xhist, uhist, xhist_inf, uhist_inf)

    plt.figure()
    plt.plot(np.arange(K.shape[2]), K[0, 0, :], label="K[0,0]")
    plt.plot(np.arange(K.shape[2]), K[0, 1, :], label="K[0,1]")
    plt.xlabel("Time step")
    plt.ylabel("Gain")
    plt.legend()
    plt.grid()
    plt.show()

    eigvals = np.linalg.eigvals(A - B @ K_inf)
    print("Closed-loop eigenvalues:")
    print(eigvals)

    # Compute costs for finite and infinite horizon
    cost_finite = lqr.cost(xhist, uhist)
    cost_infinite = lqr.cost(xhist_inf, uhist_inf)
    
    print(f"Finite horizon cost: {cost_finite}")
    print(f"Infinite horizon cost: {cost_infinite}")
