//! Usful funtions to construct and check propertyies of ultrametric matrices.

use nalgebra::DMatrix;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rand::prelude::*;

#[pymodule]
fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "is_ultrametric")]
    pub fn is_ultrametric_py(py_matrix: PyReadonlyArrayDyn<f64>) -> bool {
        let size = py_matrix.shape()[0];
        let py_array = py_matrix.as_array();
        let mut matrix = DMatrix::<f64>::zeros(size, size);
        for i in 0..size {
            for j in 0..size {
                matrix[(i, j)] = py_array[[i, j]];
            }
        }
        return self::is_ultrametric(&matrix);
    }
    #[pyfn(m)]
    #[pyo3(name = "random_ultrametric_matrix")]
    pub fn random_ultrametric_matrix_py<'py>(py: Python<'py>, size: usize) -> &'py PyArray2<f64> {
        let matrix = random_ultrametric_matrix(size);
        let mut py_matrix = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                py_matrix[[i, j]] = matrix[(i, j)];
            }
        }
        return py_matrix.into_pyarray(py);
    }
    Ok(())
}

/// Checks if a matrix is ultrametric.
///
/// To check the ultrametric property for a matrix, a similar algorithm as ```from_matrix``` for ```UltrametricTree``` is used. First, the symmetry is checked. If the matrix is symmetric, then it is recursively checked if a permutation of the matrix can be partitioned such that the off-diagonal blocks have one value and the diagonal blocks are ultrametric.
///
/// # Example:
/// ```
/// let ultrametric = ultrametric_matrix_tools::na::DMatrix::from_vec(4, 4,
///     vec![0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
/// let not_ultrametric = ultrametric_matrix_tools::na::DMatrix::from_vec(4, 4,
///     vec![0.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 2.0, 1.0, 1.0, 1.0]);
///
/// assert_eq!(ultrametric_matrix_tools::utils::is_ultrametric(&ultrametric), true);
/// assert_eq!(ultrametric_matrix_tools::utils::is_ultrametric(&not_ultrametric), false);
/// ```
pub fn is_ultrametric(matrix: &DMatrix<f64>) -> bool {
    if !is_symmetric(matrix) {
        return false;
    }
    let idx: Vec<usize> = (0..matrix.nrows()).collect();
    if !is_submatrix_ultrametric(matrix, idx, 0.0) {
        return false;
    }
    return true;
}

fn is_symmetric(matrix: &DMatrix<f64>) -> bool {
    if !matrix.is_square() {
        return false;
    }
    let size = matrix.shape().0;
    for i in 1..size {
        for j in (i + 1)..size {
            if matrix[(i, j)] != matrix[(j, i)] {
                return false;
            }
        }
    }
    return true;
}

fn is_submatrix_ultrametric(matrix: &DMatrix<f64>, idx: Vec<usize>, prev_value: f64) -> bool {
    let first_i = idx[0];
    if idx.len() == 1 {
        return true;
    }
    let mut left_partition: Vec<usize> = vec![first_i];
    let mut right_partition: Vec<usize> = Vec::new();
    let mut min = f64::MAX;
    for &i in &idx[1..] {
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
    if min < prev_value || !is_block_equal(matrix, &left_partition, &right_partition, min) {
        return false;
    }

    let left = is_submatrix_ultrametric(matrix, left_partition, min);
    let right = is_submatrix_ultrametric(matrix, right_partition, min);
    return left && right;
}

fn is_block_equal(
    matrix: &DMatrix<f64>,
    left: &Vec<usize>,
    right: &Vec<usize>,
    value: f64,
) -> bool {
    for &i in left.iter() {
        for &j in right.iter() {
            if matrix[(i, j)] != value {
                return false;
            }
        }
    }
    return true;
}

/// Constructs a random ultrametric matrix
///
/// The elements of the ultrametric matrix have an integer value between `1` and `size - 1`.
///
/// The construction is based on Theorem 1.1 in [(Fiedler, 2002)](fn.random_ultrametric_matrix.html#fn1)[^Fiedler, 2002]. The off-diagonal elements are constructed using the rules in Theorem 1.1. The diagonal elements are randomly chosen from `1` to `size - 1`.
///
/// # Example:
/// ```
/// let ultrametric_matrix = ultrametric_matrix_tools::utils::random_ultrametric_matrix(10);
///
/// assert_eq!(ultrametric_matrix_tools::utils::is_ultrametric(&ultrametric_matrix), true);
/// ```
///
/// [^Fiedler, 2002]: [Fiedler, M., 2002. Remarks on Monge matrices. Mathematica Bohemica, 127(1), pp.27-32.](https://dml.cz/bitstream/handle/10338.dmlcz/133983/MathBohem_127-2002-1_3.pdf)
pub fn random_ultrametric_matrix(size: usize) -> DMatrix<f64> {
    let mut matrix = DMatrix::<f64>::zeros(size, size);
    let mut rng = thread_rng();
    for i in 1..size {
        let elem = rng.gen_range(1..size) as f64;
        matrix[(i - 1, i)] = elem;
        matrix[(i, i - 1)] = elem;
    }
    for i in (0..(size - 2)).rev() {
        for k in (i + 2)..size {
            let elem = f64::min(matrix[(i, k - 1)], matrix[(i + 1, k)]);
            matrix[(i, k)] = elem;
            matrix[(k, i)] = elem;
        }
    }
    let diag = random_vector(size);
    for i in 0..size {
        matrix[(i, i)] = diag[i];
    }
    permutate_matrix(&mut matrix);
    return matrix;
}

/// Constructs a random special ultrametric matrix
///
/// The elements of the ultrametric matrix have an integer value between `1` and `size - 1`.
///
/// The construction is based on Theorem 1.1 in [(Fiedler, 2002)](fn.random_ultrametric_matrix.html#fn1)[^Fiedler, 2002]. The elements are constructed using the rules in Theorem 1.1.
///
/// # Example:
/// ```
/// let ultrametric_matrix = ultrametric_matrix_tools::utils::random_special_ultrametric_matrix(10);
///
/// assert_eq!(ultrametric_matrix_tools::utils::is_ultrametric(&ultrametric_matrix), true);
/// ```
///
/// [^Fiedler, 2002]: [Fiedler, M., 2002. Remarks on Monge matrices. Mathematica Bohemica, 127(1), pp.27-32.](https://dml.cz/bitstream/handle/10338.dmlcz/133983/MathBohem_127-2002-1_3.pdf)
pub fn random_special_ultrametric_matrix(size: usize) -> DMatrix<f64> {
    let mut matrix = DMatrix::<f64>::zeros(size, size);
    let mut rng = thread_rng();
    for i in 1..size {
        let elem = rng.gen_range(1..size) as f64;
        matrix[(i - 1, i)] = elem;
        matrix[(i, i - 1)] = elem;
    }
    for i in (0..(size - 2)).rev() {
        for k in (i + 2)..size {
            let elem = f64::min(matrix[(i, k - 1)], matrix[(i + 1, k)]);
            matrix[(i, k)] = elem;
            matrix[(k, i)] = elem;
        }
    }
    for i in 1..(size - 1) {
        matrix[(i, i)] = f64::max(matrix[(i - 1, i)], matrix[(i, i + 1)]);
    }
    matrix[(0, 0)] = matrix[(0, 1)];
    matrix[(size - 1, size - 1)] = matrix[(size - 2, size - 1)];
    permutate_matrix(&mut matrix);
    return matrix;
}

fn random_vector(size: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    let mut vector: Vec<f64> = Vec::new();
    for _ in 0..size {
        vector.push(rng.gen_range(1..size) as f64);
    }
    return vector;
}

fn permutate_matrix(matrix: &mut DMatrix<f64>) {
    let size = matrix.shape().0;
    let mut new_matrix = DMatrix::<f64>::zeros(size, size);
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    let mut element_vector: Vec<usize> = (0..size).collect();
    element_vector.shuffle(&mut rng);
    for (old_i, &new_i) in element_vector.iter().enumerate() {
        for (old_j, &new_j) in element_vector.iter().enumerate() {
            new_matrix[(new_i, new_j)] = matrix[(old_i, old_j)];
        }
    }
    *matrix = new_matrix;
}
