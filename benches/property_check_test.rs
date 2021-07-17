mod utils;

use criterion::{criterion_group, criterion_main, Criterion};
use ultrametric_multiplication::is_ultrametric;

criterion_group!(benches, benchmark);
criterion_main!(benches);

fn benchmark(_c: &mut Criterion) {
    let size = 100;
    let prob = 0.5;
    let mut matrix = utils::generate_random_connectivity_matrix(size, prob);
    matrix[(3, 5)] = 1.0;
    matrix[(5, 3)] = 1.0;
    dbg!(is_ultrametric(&matrix));
}
