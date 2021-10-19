use nalgebra::{DMatrix, DVector};
use ultrametric_multiplication::RootedTreeVertex;

fn main() {
    let matrix = DMatrix::from_vec(
        4,
        4,
        vec![
            7.0, 1.0, 3.0, 1.0, 1.0, 5.0, 1.0, 1.0, 3.0, 1.0, 8.0, 1.0, 1.0, 1.0, 1.0, 9.0,
        ],
    );
    let mut off_diag = matrix.clone();
    let mut diag = DVector::zeros(4);
    for i in 0..4 {
        diag[i] = off_diag[(i, i)];
        off_diag[(i, i)] = 0.0;
    }
    let b = DVector::from_vec(vec![3.0, 2.0, 6.0, 7.0]);
    let mut x = DVector::zeros(4);
    let eps = 10e-12;

    let mut off_diag_tree = RootedTreeVertex::get_partition_tree(&off_diag);
    let mut full_tree = RootedTreeVertex::get_partition_tree(&matrix);
    let mut conv = (full_tree.multiply_with_tree(&x) - &b).norm() / b.norm();
    for _ in 0..100 {
        if conv <= eps {
            break;
        }
        let sigma = off_diag_tree.multiply_with_tree(&x);
        let diff = &b - sigma;
        for i in 0..4 {
            x[i] = diff[i] / diag[i];
        }
        conv = (full_tree.multiply_with_tree(&x) - &b).norm() / b.norm();
    }
    println!("Solution x to the equation system Ax=b: {}", x);
}
