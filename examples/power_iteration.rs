use nalgebra::{DMatrix, DVector};
use ultrametric_multiplication::RootedTreeVertex;

fn main() {
    let matrix = DMatrix::from_vec(
        4,
        4,
        vec![
            0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    );

    let mut b_k = DVector::from_element(4, 1.0);
    let tree = RootedTreeVertex::get_partition_tree(&matrix);
    for _ in 0..100 {
        b_k = (&tree * &b_k).normalize();
    }
    println!("{}", b_k);
}
