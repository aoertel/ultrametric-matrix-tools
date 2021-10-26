import ultrametric_tree as ut
import numpy as np

matrix = np.array([[0.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 1.0], [
                  3.0, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

print("This matrix should be ultrametric (true):", ut.is_ultrametric(matrix))

matrix[0, 3] = 2.0
matrix[3, 0] = 2.0

print("This matrix should not be ultrametric (false):", ut.is_ultrametric(matrix))
