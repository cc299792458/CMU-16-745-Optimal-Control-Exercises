import numpy as np

if __name__ == "__main__":
    # Parameters
    h = 0.1
    A = np.array([[1, h],
                  [0, 1]])
    B = np.array([[0.5 * h**2],
                  [h]])
    n = 2

    C = B
    for i in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))
    controllable = np.linalg.matrix_rank(C) == n

    print(controllable)