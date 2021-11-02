//! `UltrametricTree` implementation.

use nalgebra::{DMatrix, DVector};
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn};
use ptree::builder::TreeBuilder;
use ptree::output::print_tree;
use pyo3::prelude::*;
use std::ops;

/// Tree that represents the structure of an ultrametric matrix.
///
/// Actually, ```UltrametricTree``` only represents one vertex of the tree. However, a vertex and its ```children``` represent the structure of an ultrametric tree. To reconstruct the original ultrametric matrix, the root vertex of the tree has to be stored.
#[pyclass]
#[derive(Default, Clone)]
pub struct UltrametricTree {
    /// Vector storing indices associated with the vertex
    partition: Vec<usize>,
    /// Vector storing indices, where the vertex is a leaf
    partition_leaves: Vec<usize>,
    /// Value of the vertex
    level: f64,
    /// Value of partial product used for multiplication
    sum: f64,
    /// Children vertices of the vertex
    children: Vec<Box<UltrametricTree>>,
}

/// Implementation of multiplication operator for `tree * &vector`.
impl<'b> ops::Mul<&'b DVector<f64>> for UltrametricTree {
    type Output = DVector<f64>;

    fn mul(self, vector: &'b DVector<f64>) -> DVector<f64> {
        let mut tree = self.clone();
        tree.mult(vector)
    }
}

/// Implementation of multiplication operator for `&tree * &vector`.
impl<'a, 'b> ops::Mul<&'b DVector<f64>> for &'a UltrametricTree {
    type Output = DVector<f64>;

    fn mul(self, vector: &'b DVector<f64>) -> DVector<f64> {
        let mut tree = self.clone();
        tree.mult(vector)
    }
}

/// Implementation of multiplication operator for `tree * vector`.
impl ops::Mul<DVector<f64>> for UltrametricTree {
    type Output = DVector<f64>;

    fn mul(self, vector: DVector<f64>) -> DVector<f64> {
        let mut tree = self.clone();
        tree.mult(&vector)
    }
}

/// Implementation of multiplication operator for `&tree * vector`.
impl<'a> ops::Mul<DVector<f64>> for &'a UltrametricTree {
    type Output = DVector<f64>;

    fn mul(self, vector: DVector<f64>) -> DVector<f64> {
        let mut tree = self.clone();
        tree.mult(&vector)
    }
}

impl UltrametricTree {
    /// Create a new vertex using `partition`.
    fn new(partition: Vec<usize>) -> Self {
        UltrametricTree {
            partition: partition,
            ..Default::default()
        }
    }

    /// Construct a `UltrametricTree` from an ultrametric matrix that represents the structure of the matrix.
    ///
    /// This function does not check if the matrix is ultrametric. The value retured by this function is the root of the tree that represents the ultrametric matrix ```matrix```. Thus, the function returns the tree that represents ```matrix```.
    ///
    /// # Example:
    ///
    /// ```
    /// let matrix = ultrametric_matrix_tools::na::DMatrix::from_vec(4, 4,
    ///     vec![0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// let tree = ultrametric_matrix_tools::UltrametricTree::from_matrix(&matrix);
    /// ```
    pub fn from_matrix(matrix: &DMatrix<f64>) -> Self {
        let vertex_ids: Vec<usize> = (0..matrix.nrows()).collect();
        let mut root = UltrametricTree::new(vertex_ids);
        root.from_matrix_recursive(&matrix);
        return root;
    }

    /// Recursive function used to construct the tree from an ultrametric matrix.
    fn from_matrix_recursive(&mut self, matrix: &DMatrix<f64>) {
        let first_i = self.partition[0];
        if self.partition.len() == 1 {
            self.level = matrix[(first_i, first_i)];
            self.partition_leaves.push(first_i);
        } else {
            let mut left_partition: Vec<usize> = vec![first_i];
            let mut right_partition: Vec<usize> = Vec::new();
            let mut min = f64::MAX;
            for &i in &self.partition[1..] {
                if min > matrix[(first_i, i)] {
                    min = matrix[(first_i, i)];
                    left_partition.extend(right_partition.iter());
                    right_partition.clear();
                    right_partition.push(i);
                } else {
                    if min == matrix[(first_i, i)] {
                        right_partition.push(i);
                    } else {
                        left_partition.push(i);
                    }
                }
            }
            self.level = min;
            let left_child = Box::new(UltrametricTree::new(left_partition));
            let right_child = Box::new(UltrametricTree::new(right_partition));
            self.children.push(left_child);
            self.children.push(right_child);

            self.children[0].as_mut().from_matrix_recursive(matrix);
            self.children[1].as_mut().from_matrix_recursive(matrix);
        }
    }

    /// Construct a `UltrametricTree` from an ultrametric matrix that approximately represents the structure of the matrix.
    ///
    /// This function is similar to [`from_matrix`](UltrametricTree::from_matrix). The difference is that the elements associated with the indices in the right_partition are at most `min + eps` instead of equal to `min`. Only the minimal value is stored in the vertex.
    ///
    /// # Example:
    ///
    /// ```
    /// let matrix = ultrametric_matrix_tools::na::DMatrix::from_vec(4, 4,
    ///     vec![0.0, 1.1, 3.0, 1.0, 1.1, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// let tree = ultrametric_matrix_tools::UltrametricTree::from_matrix_approx(&matrix, 0.1);
    /// ```
    pub fn from_matrix_approx(matrix: &DMatrix<f64>, eps: f64) -> Self {
        let vertex_ids: Vec<usize> = (0..matrix.nrows()).collect();
        let mut root = UltrametricTree::new(vertex_ids);
        root.from_matrix_approx_recursive(&matrix, eps);
        return root;
    }

    /// Recursive function used to construct the tree that approximately represents the structure of an ultrametric matrix.
    fn from_matrix_approx_recursive(&mut self, matrix: &DMatrix<f64>, eps: f64) {
        let first_i = self.partition[0];
        if self.partition.len() == 1 {
            self.level = matrix[(first_i, first_i)];
            self.partition_leaves.push(first_i);
        } else {
            let mut left_partition: Vec<usize> = vec![first_i];
            let mut right_partition: Vec<usize> = Vec::new();
            let mut min = f64::MAX;

            for &i in &self.partition[1..] {
                if matrix[(first_i, i)] < min {
                    min = matrix[(first_i, i)];
                }
            }
            for &i in &self.partition[1..] {
                if matrix[(first_i, i)] <= min + eps {
                    right_partition.push(i);
                } else {
                    left_partition.push(i);
                }
            }

            self.level = min;
            let left_child = Box::new(UltrametricTree::new(left_partition));
            let right_child = Box::new(UltrametricTree::new(right_partition));
            self.children.push(left_child);
            self.children.push(right_child);

            self.children[0]
                .as_mut()
                .from_matrix_approx_recursive(matrix, eps);
            self.children[1]
                .as_mut()
                .from_matrix_approx_recursive(matrix, eps);
        }
    }

    /// Construct the ultrametric matrix that is represented by the `UltrametricTree`.
    pub fn to_matrix(&self) -> DMatrix<f64> {
        let size = self.partition.len();
        let mut matrix = DMatrix::<f64>::zeros(size, size);
        self.to_matrix_recursive(&mut matrix);
        return matrix;
    }

    /// Recursive function to construct the ultrametric matrix that is represented by the `UltrametricTree`.
    fn to_matrix_recursive(&self, matrix: &mut DMatrix<f64>) {
        for &i in self.partition_leaves.iter() {
            matrix[(i, i)] = self.level;
        }
        for child_id1 in 0..self.children.len() {
            for &i1 in self.children[child_id1].partition.iter() {
                for child_id2 in (child_id1 + 1)..self.children.len() {
                    for &i2 in self.children[child_id2].partition.iter() {
                        matrix[(i1, i2)] = self.level;
                        matrix[(i2, i1)] = self.level;
                    }
                }
                for &i in self.partition_leaves.iter() {
                    matrix[(i, i1)] = self.level;
                    matrix[(i1, i)] = self.level;
                }
            }
            self.children[child_id1].to_matrix_recursive(matrix);
        }
        for leaf_id1 in 0..self.partition_leaves.len() {
            for leaf_id2 in (leaf_id1 + 1)..self.partition_leaves.len() {
                let leaf1 = self.partition_leaves[leaf_id1];
                let leaf2 = self.partition_leaves[leaf_id2];
                matrix[(leaf1, leaf2)] = self.level;
                matrix[(leaf2, leaf1)] = self.level;
            }
        }
    }

    /// Calculate the product of an ultrametric matrix represented by an `UltrametricTree` and a vector.
    ///
    /// The multiplication is done in two steps. The first step is to annotate the `UltrametricTree` with the partial product. The second step sums the partial products to the full product for each element of the product vector.
    ///
    /// # Example:
    ///
    /// ```
    /// let matrix = ultrametric_matrix_tools::na::DMatrix::from_vec(4, 4,
    ///     vec![0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// let vector = ultrametric_matrix_tools::na::DVector::from_vec(vec![4.0, 2.0, 7.0, 5.0]);
    /// let mut tree = ultrametric_matrix_tools::UltrametricTree::from_matrix(&matrix);
    /// let product = tree.mult(&vector);
    ///
    /// assert_eq!(product, nalgebra::DVector::from_vec(vec![28.0, 22.0, 54.0, 18.0]));
    /// ```
    pub fn mult(&mut self, vector: &DVector<f64>) -> DVector<f64> {
        self.calculate_partial_product(vector, 0.0);
        let mut product: DVector<f64> = DVector::<f64>::zeros(vector.nrows());
        self.calculate_full_product(&mut product, 0.0);
        return product;
    }

    /// Recursive function to calculate the partial product, which is the first step to calculate the product in [`multiply`](UltrametricTree::multiply).
    fn calculate_partial_product(&mut self, vector: &DVector<f64>, parent_val: f64) -> f64 {
        let mut sum = 0.;
        for &leaf_idx in self.partition_leaves.iter() {
            sum += vector[leaf_idx];
        }
        for child in self.children.iter_mut() {
            sum += child.calculate_partial_product(vector, self.level);
        }
        self.sum = (self.level - parent_val) * sum;
        return sum;
    }

    /// Recursive function to calculate the full product, which is the second step to calculate the product in [`multiply`](UltrametricTree::multiply).
    fn calculate_full_product(&self, product: &mut DVector<f64>, prev_sum: f64) {
        let sum = prev_sum + self.sum;
        for &leaf_id in self.partition_leaves.iter() {
            product[leaf_id] = sum;
        }
        for child in self.children.iter() {
            child.calculate_full_product(product, sum);
        }
    }

    /// Construct the permutation matrix of the `UltrametricTree`.
    ///
    /// The permutation matrix is implicitly used to partition the matrix for the construction of the `UltrametrixTree` via [`from_matrix`](UltrametricTree::from_matrix).
    pub fn get_permutation_matrix(&self) -> DMatrix<f64> {
        let size = self.partition.len();
        let permutations = self.get_permutation_matrix_recursive();
        let mut perm_mat = DMatrix::<f64>::zeros(size, size);
        for (i, &j) in permutations.iter().enumerate() {
            perm_mat[(i, j)] = 1.;
        }
        return perm_mat;
    }

    /// Recursive function to construct the permutation matrix.
    fn get_permutation_matrix_recursive(&self) -> Vec<usize> {
        let mut perm: Vec<usize> = Vec::new();
        for child in self.children.iter() {
            perm.extend(child.get_permutation_matrix_recursive());
        }
        perm.extend(self.partition_leaves.clone());
        return perm;
    }

    /// Recursive function to display the `UltrametricTree`.
    fn print_tree_recursive(&self, output_tree: &mut TreeBuilder) {
        if self.children.is_empty() {
            output_tree.add_empty_child(format!(
                "{:?}, {:?}, {}",
                self.partition, self.partition_leaves, self.level
            ));
        } else {
            output_tree.begin_child(format!(
                "{:?}, {:?}, {}",
                self.partition, self.partition_leaves, self.level
            ));
            for child in self.children.iter() {
                child.print_tree_recursive(output_tree);
            }
            output_tree.end_child();
        }
    }
}

#[pymethods]
impl UltrametricTree {
    /// Python wrapper for [`from_matrix`](UltrametricTree::from_matrix).
    #[new]
    pub fn from_matrix_py(py_matrix: PyReadonlyArrayDyn<f64>) -> Self {
        let size = py_matrix.shape()[0];
        let py_array = py_matrix.as_array();
        let mut matrix = DMatrix::<f64>::zeros(size, size);
        for i in 0..size {
            for j in 0..size {
                matrix[(i, j)] = py_array[[i, j]];
            }
        }

        let vertex_ids: Vec<usize> = (0..matrix.nrows()).collect();
        let mut root = UltrametricTree::new(vertex_ids);
        root.from_matrix_recursive(&matrix);
        return root;
    }

    /// Python wrapper for [`mult`](UltrametricTree::mult).
    #[pyo3(name = "mult")]
    pub fn mult_py<'py>(
        &mut self,
        py: Python<'py>,
        py_vector: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray1<f64> {
        let size = py_vector.shape()[0];
        let py_array = py_vector.as_array();
        let mut vector = DVector::<f64>::zeros(size);
        for i in 0..size {
            vector[i] = py_array[[i]];
        }

        self.calculate_partial_product(&vector, 0.0);
        let mut product: DVector<f64> = DVector::<f64>::zeros(vector.nrows());
        self.calculate_full_product(&mut product, 0.0);

        let py_product = PyArray1::from_vec(py, product.data.as_vec().clone());
        return py_product;
    }

    /// Python wrapper for [`get_permutation_matrix`](UltrametricTree::get_permutation_matrix).
    #[pyo3(name = "get_permutation_matrix")]
    pub fn get_permutation_matrix_py<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        let size = self.partition.len();
        let permutations = self.get_permutation_matrix_recursive();
        let mut perm_mat = DMatrix::<f64>::zeros(size, size);
        for (i, &j) in permutations.iter().enumerate() {
            perm_mat[(i, j)] = 1.;
        }

        let mut py_matrix = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                py_matrix[[i, j]] = perm_mat[(i, j)];
            }
        }
        return py_matrix.into_pyarray(py);
    }

    /// Prune the `UltrametricTree` to reduce the number of vertices in the tree.
    ///
    /// The pruned tree represents the same structure of the same ultrametric matrix as the tree before. Thus, the pruned tree and the original tree are semantically equivalent, but the pruned tree contains less vertices to increase performance.
    pub fn prune_tree(&mut self) {
        let mut new_children: Vec<Box<UltrametricTree>> = Vec::new();
        let mut remove_ids: Vec<usize> = Vec::new();
        for (id, child) in self.children.iter_mut().enumerate() {
            child.prune_tree();
            if child.level == self.level {
                self.partition_leaves.extend(&child.partition_leaves);
                new_children.append(&mut child.children);
                remove_ids.push(id);
            }
        }
        remove_ids.sort();
        remove_ids.reverse();
        for &id in remove_ids.iter() {
            self.children.remove(id);
        }
        self.children.extend(new_children);
    }

    /// Displays the `UltrametricTree`.
    ///
    /// The structure of the `UltrametricTree` is printed to the terminal. This includes the current vertex and all the children of this vertex. Each vertex of the printed tree is annotated with `partition`, `partition_leaves` and `level`.
    ///
    /// # Example:
    ///
    /// ```
    /// let matrix = ultrametric_matrix_tools::na::DMatrix::from_vec(4, 4,
    ///     vec![0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// let tree = ultrametric_matrix_tools::UltrametricTree::from_matrix(&matrix);
    /// tree.print_tree();
    /// ```
    /// This prints the folowing tree:
    /// ```console
    /// [0, 1, 2, 3], [], 1
    /// ├─ [0, 2], [], 3
    /// │  ├─ [0], [0], 0
    /// │  └─ [2], [2], 5
    /// └─ [1, 3], [], 1
    ///    ├─ [1], [1], 3
    ///    └─ [3], [3], 1
    /// ```
    pub fn print_tree(&self) {
        let mut tree_root = TreeBuilder::new(format!(
            "{:?}, {:?}, {}",
            self.partition, self.partition_leaves, self.level
        ));
        for child in self.children.iter() {
            child.print_tree_recursive(&mut tree_root);
        }
        let tree = tree_root.build();
        print_tree(&tree).ok();
    }
}
