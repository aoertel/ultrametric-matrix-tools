import ultrametric_multiplication
import numpy as np

matrix = np.array([[0., 1, 1, 1, 2, 1, 1, 2],
                   [1, 0, 1, 3, 1, 2, 2, 1],
                   [1, 1, 4, 1, 1, 1, 1, 1],
                   [1, 3, 1, 0, 1, 2, 2, 1],
                   [2, 1, 1, 1, 2, 1, 1, 2],
                   [1, 2, 1, 2, 1, 0, 2, 1],
                   [1, 2, 1, 2, 1, 2, 0, 1],
                   [2, 1, 1, 1, 2, 1, 1, 0]])
vector = np.array([1., 2, 3, 4, 5, 6, 7, 8])

tree = ultrametric_multiplication.RootedTreeVertex(matrix)
tree.print_rooted_tree()
product = tree.mult_with_tree(vector)
perm_mat = tree.get_perm_matrix()
print(product)
print(matrix.dot(vector))
print(perm_mat)
