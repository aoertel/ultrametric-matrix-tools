use nalgebra::DMatrix;
use ultrametric_matrix_tools::UltrametricTree;

fn main() {
    let matrix = DMatrix::from_vec(
        4,
        4,
        vec![
            0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    );
    let tree = UltrametricTree::from_matrix(&matrix);
    let permutation_matrix = tree.get_permutation_matrix();
    println!(
        "Permutation matrix associated with the tree: {}",
        permutation_matrix
    );
}
