import ultrametric_multiplication as um
import numpy as np

matrix = np.array([[0.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 1.0], [
    3.0, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

b_k = np.ones(matrix.shape[1])
tree = um.RootedTreeVertex(matrix)
for _ in range(100):
    b_k1 = tree.mult(b_k)
    b_k1_norm = np.linalg.norm(b_k1)
    b_k = b_k1 / b_k1_norm
print(b_k)
