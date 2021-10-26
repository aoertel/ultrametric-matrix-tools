use nalgebra::DMatrix;
use ultrametric_tree::utils::*;

fn main() {
    let mut matrix = DMatrix::from_vec(
        4,
        4,
        vec![
            0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    );
    println!(
        "This matrix should be ultrametric (true): {}",
        is_ultrametric(&matrix)
    );
    matrix[(0, 3)] = 2.0;
    matrix[(3, 0)] = 2.0;
    println!(
        "This matrix should not be ultrametric (false): {}",
        is_ultrametric(&matrix)
    );
}
