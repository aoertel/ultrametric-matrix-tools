mod utils;

use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::DVector;
use std::time::SystemTime;
use ultrametric_multiplication::RootedTreeVertex;

criterion_group!(benches, benchmark_jacobi);
criterion_main!(benches);

fn benchmark_jacobi(_c: &mut Criterion) {
    let size = 20000;
    let diag_elem = (size * size) as f64;
    let mut matrix = utils::generate_random_ultrametric_matrix(size);
    let mut off_diag = matrix.clone();
    let mut diag = DVector::zeros(size);
    for i in 0..size {
        diag[i] = diag_elem;
        matrix[(i, i)] = diag_elem;
        off_diag[(i, i)] = 0.;
    } 
    let b = utils::generate_random_vector(size);
    let x_start = utils::generate_random_vector(size);
    let mut x = x_start.clone();
    let mut conv = ((&matrix * &x) - &b).norm();

    let start_normal = SystemTime::now();
    while conv > 10e-4 {
        let sigma = &off_diag * &x;
        let diff = &b - sigma;
        for i in 0..size {
            x[i] = diff[i] / diag[i];
        }
        conv = ((&matrix * &x) - &b).norm();
    }
    let duration_normal = start_normal.elapsed().unwrap();

    let mut x = x_start.clone();
    conv = ((&matrix * &x) - &b).norm();

    let start_tree_gen = SystemTime::now();
    let mut tree = RootedTreeVertex::get_partition_tree(&off_diag);
    let duration_tree_gen = start_tree_gen.elapsed().unwrap();

    let start_fast = SystemTime::now();
    while conv > 10e-4 {
        let sigma = tree.multiply_with_tree(&x);
        let diff = &b - sigma;
        for i in 0..size {
            x[i] = diff[i] / diag[i];
        }
        conv = ((&matrix * &x) - &b).norm();
    }
    let duration_fast = start_fast.elapsed().unwrap();

    println!("Normal time: {:?}", duration_normal);
    println!("Tree gen time: {:?}", duration_tree_gen);
    println!("Fast Jacobi alone time: {:?}", duration_fast);
    println!("Fast Jacobi complete time: {:?}", duration_tree_gen + duration_fast);
}