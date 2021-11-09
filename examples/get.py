from ultrametric_matrix_tools import UltrametricTree
import numpy as np

matrix = np.array([[0.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 1.0], [
                  3.0, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

tree = UltrametricTree(matrix)
print("Element of the matrix with index (0, 2):", tree.get(0, 2))
print("Requesting element outside the matrix result:", tree.get(0, 4))
