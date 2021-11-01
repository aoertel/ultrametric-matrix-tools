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

#[allow(unused)]
pub fn random_ultrametric_matrix(size: usize) -> DMatrix<f64> {
    let mut matrix = DMatrix::<f64>::zeros(size, size);
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    for i in 1..size {
        let elem = rng.gen_range(1..size) as f64;
        matrix[(i - 1, i)] = elem;
        matrix[(i, i - 1)] = elem;
    }
    for i in (0..(size - 2)).rev() {
        for k in (i + 2)..size {
            let elem = f64::min(matrix[(i, k - 1)], matrix[(i + 1, k)]);
            matrix[(i, k)] = elem;
            matrix[(k, i)] = elem;
        }
    }
    let diag = random_vector(size);
    for i in 0..size {
        matrix[(i, i)] = diag[i];
    }
    let permutation = random_permutation(size);
    matrix = &permutation * matrix * &permutation.transpose();
    return matrix;
}

fn random_permutation(size: usize) -> DMatrix<f64> {
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    let mut element_vector: Vec<usize> = (0..size).collect();
    element_vector.shuffle(&mut rng);
    let mut matrix = DMatrix::<f64>::zeros(size, size);
    for (i, &j) in element_vector.iter().enumerate() {
        matrix[(i, j)] = 1.0;
    }
    return matrix;
}

#[allow(unused)]
pub fn calculate_normal_product(matrix: &DMatrix<f64>, vector: &DVector<f64>) -> DVector<f64> {
    let size = vector.nrows();
    let mut product: DVector<f64> = DVector::<f64>::zeros(size);
    for i in 0..size {
        for j in 0..size {
            product[i] += matrix[(i, j)] * vector[j];
        }
    }
    return product;
}
