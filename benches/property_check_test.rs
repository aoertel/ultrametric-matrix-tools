mod utils;

use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector, Vector4};
use ultrametric_multiplication::{is_ultrametric, RootedTreeVertex};

criterion_group!(benches, simple_benchmark, construct_property_benchmark);
criterion_main!(benches);

fn simple_benchmark(_c: &mut Criterion) {
    let size = 100;
    let prob = 0.5;
    let mut matrix = utils::generate_random_connectivity_matrix(size, prob);
    dbg!(is_ultrametric(&matrix));
    matrix[(3, 5)] = 1.0;
    matrix[(5, 3)] = 1.0;
    dbg!(is_ultrametric(&matrix));
}

fn construct_property_benchmark(_c: &mut Criterion) {
    let mut v: Vec<Vector4<f64>> = Vec::new();
    v.push(Vector4::new(1., 5., 3., 12.));
    v.push(Vector4::new(1., 4., 2., 15.));
    v.push(Vector4::new(1., 1., 3., 13.));
    v.push(Vector4::new(0., 1., 10., 4.));
    v.push(Vector4::new(0., 5., 9., 3.));
    let mut matrix = DMatrix::<f64>::zeros(5, 5);

    for i in 0..5 {
        for j in i..5 {
            let value = v[i].dot(&v[j]) / (v[i].norm() * v[j].norm());
            matrix[(i, j)] = value;
            matrix[(j, i)] = value;
        }
    }

    println!("{}", matrix);
    let vector = DVector::from_iterator(5, [0., 1., 2., 10., 15.]);
    let mut approx_tree = RootedTreeVertex::get_approximate_partition_tree(&matrix, 0.1);
    let approx_product = approx_tree.multiply_with_tree(&vector);
    let product = matrix * vector;
    println!("Product: {}", product);
    println!("Approximated Product: {}", approx_product);
}
