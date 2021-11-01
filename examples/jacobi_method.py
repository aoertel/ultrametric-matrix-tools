from ultrametric_matrix_tools import UltrametricTree
import numpy as np

matrix = np.array([[7.0, 1.0, 3.0, 1.0], [1.0, 5.0, 1.0, 1.0], [
    3.0, 1.0, 8.0, 1.0], [1.0, 1.0, 1.0, 9.0]])
off_diag = matrix
diag = np.zeros(4)
for i in range(4):
    diag[i] = matrix[i, i]
    matrix[i, i] = 0.0
b = np.array([3.0, 2.0, 6.0, 7.0])
x = np.zeros(4)
eps = 10e-12

off_diag_tree = UltrametricTree(off_diag)
full_tree = UltrametricTree(matrix)
conv = np.linalg.norm(full_tree.mult(x) - b) / np.linalg.norm(b)
for _ in range(100):
    if conv <= eps:
        break
    sigma = off_diag_tree.mult(x)
    diff = b - sigma
    for i in range(4):
        x[i] = diff[i] / diag[i]
    conv = np.linalg.norm(full_tree.mult(x) - b) / np.linalg.norm(b)
print("Solution x to the equation system Ax=b:", x)
