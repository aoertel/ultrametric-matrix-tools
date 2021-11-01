from ultrametric_matrix_tools import UltrametricTree
import numpy as np

matrix = np.array([[0.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 1.0], [
                  3.0, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
vector = np.array([4.0, 2.0, 7.0, 5.0])

tree = UltrametricTree(matrix)
fast_product = tree.mult(vector)
print("Product using our method:", fast_product)

normal_product = matrix.dot(vector)
print("Product using normal multiplication:", normal_product)
