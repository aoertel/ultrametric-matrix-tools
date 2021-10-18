import ultrametric_multiplication as um
import numpy as np

matrix = np.array([[0.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 1.0], [
                  3.0, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
vector = np.array([4.0, 2.0, 7.0, 5.0])

tree = um.RootedTreeVertex(matrix)
fast_product = tree.mult_with_tree(vector)
print(fast_product)

normal_product = matrix.dot(vector)
print(normal_product)
