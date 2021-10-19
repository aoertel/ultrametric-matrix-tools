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
    let mut tree = RootedTreeVertex::get_partition_tree(&matrix);
    let product = tree.multiply_with_tree(&vector);
    println!("Product: {}", product);
    tree.prune_tree();
    let pruned_tree_product = tree.multiply_with_tree(&vector);
    println!("Product with pruned tree: {}", pruned_tree_product);
}
