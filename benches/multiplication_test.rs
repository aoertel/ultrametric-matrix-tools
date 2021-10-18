mod utils;

use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector};
use std::time::SystemTime;
use ultrametric_multiplication::RootedTreeVertex;

criterion_group!(benches, benchmark);
criterion_main!(benches);

fn benchmark(_c: &mut Criterion) {
    let size = 5;
    let matrix = utils::random_ultrametric_matrix(size);
    //let matrix = DMatrix::<f64>::repeat(size, size, 1.);
    //println!("{}", &matrix);
    let vector = utils::random_vector(size);
    //println!("{}", &vector);

    let start_fast = SystemTime::now();
    let mut root = RootedTreeVertex::get_partition_tree(&matrix);
    let reconstructed_matrix = root.reconstruct_matrix();
    root.prune_tree();
    let pruned_matrix = root.reconstruct_matrix();
    if !pruned_matrix.eq(&matrix) {
        let mut root = RootedTreeVertex::get_partition_tree(&matrix);
        root.print_tree();
        root.prune_tree();
        root.print_tree();
        println!("{}", &matrix);
        println!("{}", &reconstructed_matrix);
        println!("{}", &pruned_matrix);
    }
    let duration_tree_gen = start_fast.elapsed().unwrap();
    let fast_product = root.multiply_with_tree(&vector);
    let duration_fast_full = start_fast.elapsed().unwrap();

    let start_normal = SystemTime::now();
    let normal_product = calculate_normal_product(&matrix, &vector);
    let duration_normal = start_normal.elapsed().unwrap();

    //println!("Fast product: {}", &fast_product);
    //println!("Normal product: {}", &normal_product);
    if normal_product == fast_product {
        println!("Same Results!");
    } else {
        println!("Different Results!");
    }

    let perm_mat = root.get_permutation_matrix();
    let permutated_matrix = &perm_mat * &matrix * &perm_mat.transpose();
    //println!("Permuted matrix: {}", permutated_matrix);
    let permutated_vector = &perm_mat * &vector;
    //println!("Permuted vector: {}", permutated_vector);

    let start_fast_perm = SystemTime::now();
    let mut root_perm = RootedTreeVertex::get_partition_tree(&permutated_matrix);
    let duration_tree_gen_perm = start_fast_perm.elapsed().unwrap();
    let _fast_product_perm = root_perm.multiply_with_tree(&permutated_vector);
    let duration_fast_full_perm = start_fast_perm.elapsed().unwrap();

    println!("Time for tree generation: {:?}", duration_tree_gen);
    println!(
        "Time for multiplication with known tree: {:?}",
        duration_fast_full - duration_tree_gen
    );
    println!(
        "Time for full fast multiplication: {:?}",
        duration_fast_full
    );
    println!(
        "Time for tree generation with permutated matrix: {:?}",
        duration_tree_gen_perm
    );
    println!(
        "Time for multiplication with known tree with permutated matrix: {:?}",
        duration_fast_full_perm - duration_tree_gen_perm
    );
    println!(
        "Time for full fast multiplication with permutated matrix: {:?}",
        duration_fast_full_perm
    );
    println!("Time for normal multiplication: {:?}", duration_normal);
}

fn calculate_normal_product(matrix: &DMatrix<f64>, vector: &DVector<f64>) -> DVector<f64> {
    let size = vector.nrows();
    let mut product: DVector<f64> = DVector::<f64>::zeros(size);
    for i in 0..size {
        for j in 0..size {
            product[i] += matrix[(i, j)] * vector[j];
        }
    }
    return product;
}
