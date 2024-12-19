import numpy as np
import matplotlib.pyplot as plt

class LQR:
    def __init__(self, A, B, Q, R, QN, x0, h, T):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R  # Ensure R is a 2D array
        self.QN = QN
        self.x0 = x0
        self.h = h
        self.T = T
        self.N = int(T / h) + 1
        self.t_hist = np.linspace(0, T, self.N)
        self.x_hist = np.tile(x0.reshape(-1, 1), self.N)
        self.u_hist = np.zeros(self.N - 1)
        self.delta_u = np.ones(self.N - 1)
        self.λ_hist = np.zeros((A.shape[0], self.N))

    def cost(self, x_hist, u_hist):
        cost = 0.5 * x_hist[:, -1].T @ self.QN @ x_hist[:, -1]
        for k in range(self.N - 1):
            cost += 0.5 * x_hist[:, k].T @ self.Q @ x_hist[:, k] + 0.5 * u_hist[k] ** 2 * self.R[0, 0]
        return cost

    def rollout(self, x_hist, u_hist):
        xnew = np.zeros_like(x_hist)
        xnew[:, 0] = x_hist[:, 0]
        for k in range(self.N - 1):
            xnew[:, k + 1] = self.A @ xnew[:, k] + self.B.flatten() * u_hist[k]
        return xnew

    def solve(self, tol=1e-2, b=1e-2):
        self.x_hist = self.rollout(self.x_hist, self.u_hist)
        print("Initial cost:", self.cost(self.x_hist, self.u_hist))

        α = 1.0
        iter_count = 0

        while np.max(np.abs(self.delta_u)) > tol:
            # Backward pass to compute λ and delta_u
            self.λ_hist[:, -1] = self.QN @ self.x_hist[:, -1]
            for k in range(self.N - 2, -1, -1):
                self.delta_u[k] = -(self.u_hist[k] + np.linalg.solve(self.R, self.B.T @ self.λ_hist[:, k + 1]).item())
                self.λ_hist[:, k] = self.Q @ self.x_hist[:, k] + self.A.T @ self.λ_hist[:, k + 1]

            # Forward pass with line search
            α = 1.0
            unew = self.u_hist + α * self.delta_u
            xnew = self.rollout(self.x_hist, unew)

            while self.cost(xnew, unew) > self.cost(self.x_hist, self.u_hist) - b * α * np.sum(self.delta_u ** 2):
                α *= 0.5
                unew = self.u_hist + α * self.delta_u
                xnew = self.rollout(self.x_hist, unew)

            # Update control and state trajectories
            self.u_hist = unew
            self.x_hist = xnew
            iter_count += 1

        print("Iterations:", iter_count)
        print("Final cost:", self.cost(self.x_hist, self.u_hist))
        return self.x_hist, self.u_hist

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_hist, self.x_hist[0, :], label="Position")
        plt.plot(self.t_hist, self.x_hist[1, :], label="Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.legend()
        plt.title("State Trajectories")
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.t_hist[:-1], self.u_hist, label="Control")
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
    R = np.array([[0.1]])  # control cost
    QN = np.eye(2)  # terminal state cost

    # Initial conditions
    x0 = np.array([1.0, 0])  # initial state

    # Solve LQR problem
    lqr = LQR(A, B, Q, R, QN, x0, h, T)
    x_hist, u_hist = lqr.solve()
    lqr.plot_results()
