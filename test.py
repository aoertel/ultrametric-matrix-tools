import ultrametric_multiplication as um
import numpy as np
import time

size = 1000

matrix = um.TestFunctions().generate_random_connectivity_matrix(size, 0.01)
vector = np.random.rand(size)

fast_start = time.time()
tree = um.RootedTreeVertex(matrix)
after_tree = time.time()
fast_product = tree.mult_with_tree(vector)
fast_end = time.time()

normal_start = time.time()
normal_product = matrix.dot(vector)
normal_end = time.time()

perm_mat = tree.get_perm_matrix()
#print("Our Product:")
# print(fast_product)
#print("Normal Product:")
# print(normal_product)
#print("Permutation Matrix:")
# print(perm_mat)

print("Time for tree generation:", after_tree - fast_start)
print("Time for product with known tree:", fast_end - after_tree)
print("Time for full fast product:", fast_end - fast_start)
print("Time for normal product:", normal_end - normal_start)
