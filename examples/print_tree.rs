use ultrametric_matrix_tools::na::DMatrix;
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
    tree.print_tree();
}
