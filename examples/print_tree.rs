use nalgebra::DMatrix;
use ultrametric_multiplication::RootedTreeVertex;

fn main() {
    let matrix = DMatrix::from_vec(
        4,
        4,
        vec![
            0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    );
    let tree = RootedTreeVertex::get_partition_tree(&matrix);
    tree.print_tree();
}
