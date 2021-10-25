import ultrametric_multiplication as um
import numpy as np

matrix = np.array([[0.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 1.0], [
                  3.0, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
vector = np.array([4.0, 2.0, 7.0, 5.0])

tree = um.RootedTreeVertex(matrix)
product = tree.mult(vector)
print("Product:", product)
tree.prune_tree()
pruned_tree_product = matrix.dot(vector)
print("Product with pruned tree:", pruned_tree_product)
