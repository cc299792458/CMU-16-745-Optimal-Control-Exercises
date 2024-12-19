import numpy as np
import matplotlib.pyplot as plt

class LQR:
    def __init__(self, A, B, Q, R, QN, x0, h, T):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = np.array([[R]])  # Ensure R is a 2D array
        self.QN = QN
        self.x0 = x0
        self.h = h
        self.T = T
        self.N = int(T / h) + 1
        self.thist = np.linspace(0, T, self.N)
        self.xhist = np.tile(x0.reshape(-1, 1), self.N)
        self.uhist = np.zeros(self.N - 1)
        self.Δu = np.ones(self.N - 1)
        self.λhist = np.zeros((A.shape[0], self.N))

    def cost(self, xhist, uhist):
        cost = 0.5 * xhist[:, -1].T @ self.QN @ xhist[:, -1]
        for k in range(self.N - 1):
            cost += 0.5 * xhist[:, k].T @ self.Q @ xhist[:, k] + 0.5 * uhist[k] ** 2 * self.R[0, 0]
        return cost

    def rollout(self, xhist, uhist):
        xnew = np.zeros_like(xhist)
        xnew[:, 0] = xhist[:, 0]
        for k in range(self.N - 1):
            xnew[:, k + 1] = self.A @ xnew[:, k] + self.B.flatten() * uhist[k]
        return xnew

    def solve(self, tol=1e-2, b=1e-2):
        self.xhist = self.rollout(self.xhist, self.uhist)
        print("Initial cost:", self.cost(self.xhist, self.uhist))

        α = 1.0
        iter_count = 0

        while np.max(np.abs(self.Δu)) > tol:
            # Backward pass to compute λ and Δu
            self.λhist[:, -1] = self.QN @ self.xhist[:, -1]
            for k in range(self.N - 2, -1, -1):
                self.Δu[k] = -(self.uhist[k] + np.linalg.solve(self.R, self.B.T @ self.λhist[:, k + 1]).item())
                self.λhist[:, k] = self.Q @ self.xhist[:, k] + self.A.T @ self.λhist[:, k + 1]

            # Forward pass with line search
            α = 1.0
            unew = self.uhist + α * self.Δu
            xnew = self.rollout(self.xhist, unew)

            while self.cost(xnew, unew) > self.cost(self.xhist, self.uhist) - b * α * np.sum(self.Δu ** 2):
                α *= 0.5
                unew = self.uhist + α * self.Δu
                xnew = self.rollout(self.xhist, unew)

            # Update control and state trajectories
            self.uhist = unew
            self.xhist = xnew
            iter_count += 1

        print("Iterations:", iter_count)
        print("Final cost:", self.cost(self.xhist, self.uhist))
        return self.xhist, self.uhist

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.thist, self.xhist[0, :], label="Position")
        plt.plot(self.thist, self.xhist[1, :], label="Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.legend()
        plt.title("State Trajectories")
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.thist[:-1], self.uhist, label="Control")
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
    T = 5.0  # final time

    # Define cost weights
    Q = np.eye(2)  # state cost
    R = 0.1  # control cost
    QN = np.eye(2)  # terminal state cost

    # Initial conditions
    x0 = np.array([1.0, 0])  # initial state

    # Solve LQR problem
    lqr = LQR(A, B, Q, R, QN, x0, h, T)
    xhist, uhist = lqr.solve()
    lqr.plot_results()
