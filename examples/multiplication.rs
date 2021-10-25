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
    let vector = DVector::from_vec(vec![4.0, 2.0, 7.0, 5.0]);
    let tree = RootedTreeVertex::get_partition_tree(&matrix);
    let fast_product = &tree * &vector;
    println!("Product using our method: {}", fast_product);
    let normal_product = matrix * vector;
    println!("Product using normal multiplication: {}", normal_product);
}
