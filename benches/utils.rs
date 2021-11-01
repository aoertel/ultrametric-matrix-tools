use nalgebra::{DMatrix, DVector};
use rand::prelude::*;

#[allow(unused)]
pub fn random_vector(size: usize) -> DVector<f64> {
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    let mut vector = DVector::<f64>::zeros(size);
    for i in 0..size {
        vector[i] = rng.gen_range(1..size) as f64;
    }
    return vector;
}

#[allow(unused)]
pub fn random_terrace_size_ultrametric_matrix(size: usize) -> DMatrix<f64> {
    let mut matrix = DMatrix::<f64>::zeros(size, size);
    ultrametric_matrix_recursion(&mut matrix, 0, size - 1, 1.);
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    for i in 0..size {
        matrix[(i, i)] = rng.gen_range(0..size) as f64;
    }
    return matrix;
}

fn ultrametric_matrix_recursion(matrix: &mut DMatrix<f64>, lower: usize, upper: usize, value: f64) {
    if upper >= lower + 1 {
        let mut rng: StdRng = SeedableRng::seed_from_u64(42);
        let seperator = rng.gen_range(lower..upper) + 1;
        for i in lower..seperator {
            for j in seperator..(upper + 1) {
                matrix[(i, j)] = value;
                matrix[(j, i)] = value;
            }
        }
        ultrametric_matrix_recursion(matrix, lower, seperator - 1, value + 1.);
        ultrametric_matrix_recursion(matrix, seperator, upper, value + 1.);
    }
}
