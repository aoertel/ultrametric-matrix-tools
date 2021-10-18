import ultrametric_multiplication as um
import numpy as np
import time


def main():
    size = 100
    matrix = um.TestFunctions().generate_random_connectivity_matrix(size, 0.5)
    normal_start = time.time()
    power_iteration_normal(matrix, 10000)
    normal_end = time.time()
    fast_start = time.time()
    power_iteration_fast(matrix, 10000)
    fast_end = time.time()
    print("Time for fast:", fast_end - fast_start)
    print("Time for normal:", normal_end - normal_start)

    normal_start = time.time()
    arnoldi_iteration_normal(matrix, 10000)
    normal_end = time.time()
    fast_start = time.time()
    arnoldi_iteration_fast(matrix, 10000)
    fast_end = time.time()
    print("Time for fast:", fast_end - fast_start)
    print("Time for normal:", normal_end - normal_start)


def power_iteration_normal(A, num_simulations: int):
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def power_iteration_fast(A, num_simulations: int):
    b_k = np.random.rand(A.shape[1])
    tree = um.RootedTreeVertex(A)

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = tree.mult_with_tree(b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def arnoldi_iteration_normal(A, n: int):
    m = A.shape[0]
    h = np.zeros((n + 1, n))
    Q = np.zeros((m, n + 1))
    b = np.random.rand(m)
    q = b / np.linalg.norm(b)  # Normalize the input vector
    Q[:, 0] = q  # Use it as the first Krylov vector

    for k in range(n):
        v = A.dot(q)  # Generate a new candidate vector
        for j in range(k + 1):  # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        eps = 1e-12  # If v is shorter than this threshold it is the zero vector
        if h[k + 1, k] > eps:  # Add the produced vector to the list, unless
            q = v / h[k + 1, k]  # the zero vector is produced.
            Q[:, k + 1] = q
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h


def arnoldi_iteration_fast(A, n: int):
    m = A.shape[0]
    h = np.zeros((n + 1, n))
    Q = np.zeros((m, n + 1))
    b = np.random.rand(m)
    q = b / np.linalg.norm(b)  # Normalize the input vector
    Q[:, 0] = q  # Use it as the first Krylov vector
    tree = um.RootedTreeVertex(A)

    for k in range(n):
        v = tree.mult_with_tree(q)  # Generate a new candidate vector
        for j in range(k + 1):  # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        eps = 1e-12  # If v is shorter than this threshold it is the zero vector
        if h[k + 1, k] > eps:  # Add the produced vector to the list, unless
            q = v / h[k + 1, k]  # the zero vector is produced.
            Q[:, k + 1] = q
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h


main()
