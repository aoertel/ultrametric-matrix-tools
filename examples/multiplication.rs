use ultrametric_matrix_tools::na::{DMatrix, DVector};
use ultrametric_matrix_tools::UltrametricTree;

fn main() {
    let matrix = DMatrix::from_vec(
        4,
        4,
        vec![
            0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    );
    let vector = DVector::from_vec(vec![4.0, 2.0, 7.0, 5.0]);
    let tree = UltrametricTree::from_matrix(&matrix);
    let fast_product = tree * &vector;
    println!("Product using our method: {}", fast_product);
    let normal_product = matrix * &vector;
    println!("Product using normal multiplication: {}", normal_product);
}
