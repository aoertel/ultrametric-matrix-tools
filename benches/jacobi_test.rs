mod utils;

use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector};
use std::io;
use std::time::SystemTime;
use ultrametric_multiplication::RootedTreeVertex;

criterion_group!(benches, benchmark_jacobi);
criterion_main!(benches);

fn benchmark_jacobi(_c: &mut Criterion) {
    let size = 100;
    let mut off_diag = utils::generate_random_ultrametric_matrix(size);
    let mut diag = DVector::zeros(size);
    for i in 0..size {
        diag[i] = off_diag[(i, i)];
        off_diag[(i, i)] = 0.;
    } 
    let b = utils::generate_random_vector(size);
    let mut x = utils::generate_random_vector(size);
    let mut conv = 1.;
    dbg!(conv);
    while conv > 10e-4 {
        let sigma = &off_diag * &x;
        let diff = &b - sigma;
        for i in 0..size {
            x[i] = diff[i] / diag[i];
        }
        let conv = diff.norm();
        dbg!(conv);
    }
}