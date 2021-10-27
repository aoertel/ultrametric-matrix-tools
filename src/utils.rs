//! Usful funtions in connection with ultrametric matrices.

use nalgebra::DMatrix;

/// Checks if a matrix is ultrametric.
///
/// To check the ultrametric property for a matrix, a similar algorithm as ```from_matrix``` for ```UltrametricTree``` is used. First, the symmetry is checked. If the matrix is symmetric, then it is recursively checked if a permutation of the matrix can be partitioned such that the off-diagonal blocks have one value and the diagonal blocks are ultrametric.
///
/// # Example:
/// ```
/// let ultrametric = nalgebra::DMatrix::from_vec(4, 4,
///     vec![0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
/// let not_ultrametric = nalgebra::DMatrix::from_vec(4, 4,
///     vec![0.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 2.0, 1.0, 1.0, 1.0]);
///
/// assert_eq!(is_ultrametric(ultrametric), true);
/// assert_eq!(is_ultrametric(not_ultrametric), false);
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
