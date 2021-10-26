from ultrametric_tree import UltrametricTree
import numpy as np

matrix = np.array([[0.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 1.0], [
                  3.0, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

tree = UltrametricTree(matrix)
permutation_matrix = tree.get_permutation_matrix()
print("Permutation matrix associated with the tree:\n", permutation_matrix)
