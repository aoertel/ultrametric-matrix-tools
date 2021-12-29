mod utils;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use std::io;
use std::time::SystemTime;
use ultrametric_matrix_tools::na::DVector;
use ultrametric_matrix_tools::UltrametricTree;

criterion_group!(benches, benchmark_jacobi);
criterion_main!(benches);

const MATRIX_SIZES: [usize; 7] = [
    2usize.pow(3),
    2usize.pow(5),
    2usize.pow(7),
    2usize.pow(9),
    2usize.pow(11),
    2usize.pow(13),
    2usize.pow(15),
];
const NUM_SAMPLES: u32 = 10;
const MAX_ITERATIONS: u32 = 1000000;
const TOLERANCE: f64 = 10e-10;
const HEADER_SINGLE: [&str; 16] = [
    "pos",
    "size",
    "tree_gen_mean",
    "tree_gen_std",
    "tree_algo_mean",
    "tree_algo_std",
    "complete_tree_algo_mean",
    "complete_tree_algo_std",
    "prune_tree_mean",
    "prune_tree_std",
    "pruned_tree_algo_mean",
    "pruned_tree_algo_std",
    "complete_pruned_tree_algo_mean",
    "complete_pruned_tree_algo_std",
    "normal_algo_mean",
    "normal_algo_std",
];

fn benchmark_jacobi(_c: &mut Criterion) {
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    let mut pos = 0;
    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b',')
        .from_writer(io::stdout());
    wtr.write_record(&HEADER_SINGLE).unwrap();
    for &size in MATRIX_SIZES.iter() {
        let mut tree_gen_times: Vec<f64> = Vec::new();
        let mut tree_algo_times: Vec<f64> = Vec::new();
        let mut complete_tree_times: Vec<f64> = Vec::new();
        let mut prune_tree_times: Vec<f64> = Vec::new();
        let mut pruned_tree_algo_times: Vec<f64> = Vec::new();
        let mut complete_pruned_tree_times: Vec<f64> = Vec::new();
        let mut normal_algo_times: Vec<f64> = Vec::new();
        for _ in 0..NUM_SAMPLES {
            let mut matrix = utils::random_special_ultrametric_matrix(size);
            let mut off_diag = matrix.clone();
            let mut diag = DVector::zeros(size);
            for i in 0..size {
                let mut diag_elem = 0.0;
                for j in 0..size {
                    if i != j {
                        diag_elem += matrix[(i, j)];
                    }
                }
                diag_elem = rng.gen_range((1.0 + diag_elem)..((diag_elem * diag_elem) + 1.0));
                diag[i] = diag_elem;
                matrix[(i, i)] = diag_elem;
                off_diag[(i, i)] = 0.;
            }
            let b = utils::random_vector(size);
            let b_norm = b.norm();
            let x_start = utils::random_vector(size);

            let start_tree_gen = SystemTime::now();
            let mut off_diag_tree = UltrametricTree::from_matrix(&off_diag);
            let mut full_tree = UltrametricTree::from_matrix(&matrix);
            let duration_tree_gen = start_tree_gen.elapsed().unwrap();

            let mut x = x_start.clone();
            let mut conv = (full_tree.mult(&x) - &b).norm() / b_norm;

            let start_fast = SystemTime::now();
            for _ in 0..MAX_ITERATIONS {
                if conv <= TOLERANCE {
                    break;
                }
                let sigma = off_diag_tree.mult(&x);
                let diff = &b - sigma;
                for i in 0..size {
                    x[i] = diff[i] / diag[i];
                }
                conv = (full_tree.mult(&x) - &b).norm() / b_norm;
            }
            let duration_fast = start_fast.elapsed().unwrap();
            tree_gen_times.push(duration_tree_gen.as_secs_f64());
            tree_algo_times.push(duration_fast.as_secs_f64());
            complete_tree_times.push(duration_tree_gen.as_secs_f64() + duration_fast.as_secs_f64());

            let mut x = x_start.clone();
            let mut conv = (full_tree.mult(&x) - &b).norm() / b_norm;

            let start_prune_tree = SystemTime::now();
            off_diag_tree.prune_tree();
            full_tree.prune_tree();
            let duration_prune_tree = start_prune_tree.elapsed().unwrap();

            let start_pruned_fast = SystemTime::now();
            for _ in 0..MAX_ITERATIONS {
                if conv <= TOLERANCE {
                    break;
                }
                let sigma = off_diag_tree.mult(&x);
                let diff = &b - sigma;
                for i in 0..size {
                    x[i] = diff[i] / diag[i];
                }
                conv = (full_tree.mult(&x) - &b).norm() / b_norm;
            }
            let duration_pruned_fast = start_pruned_fast.elapsed().unwrap();
            prune_tree_times.push(duration_prune_tree.as_secs_f64());
            pruned_tree_algo_times.push(duration_pruned_fast.as_secs_f64());
            complete_pruned_tree_times.push(
                duration_tree_gen.as_secs_f64()
                    + duration_prune_tree.as_secs_f64()
                    + duration_pruned_fast.as_secs_f64(),
            );

            let mut x = x_start.clone();
            let mut conv = (full_tree.mult(&x) - &b).norm() / b_norm;

            let start_normal = SystemTime::now();
            for _ in 0..MAX_ITERATIONS {
                if conv <= TOLERANCE {
                    break;
                }
                let sigma = utils::calculate_normal_product(&off_diag, &x);
                let diff = &b - sigma;
                for i in 0..size {
                    x[i] = diff[i] / diag[i];
                }
                conv = (utils::calculate_normal_product(&matrix, &x) - &b).norm() / b_norm;
            }
            let duration_normal = start_normal.elapsed().unwrap();
            normal_algo_times.push(duration_normal.as_secs_f64());
        }

        let tree_gen_mean = tree_gen_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let tree_algo_mean = tree_algo_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let complete_tree_algo_mean = complete_tree_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let prune_tree_mean = prune_tree_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let pruned_tree_algo_mean = pruned_tree_algo_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let complete_pruned_tree_mean =
            complete_pruned_tree_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let normal_algo_mean = normal_algo_times.iter().sum::<f64>() / NUM_SAMPLES as f64;

        let tree_gen_std = tree_gen_times
            .iter()
            .map(|&val| ((val - tree_gen_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let tree_algo_std = tree_algo_times
            .iter()
            .map(|&val| ((val - tree_algo_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let complete_tree_algo_std = complete_tree_times
            .iter()
            .map(|&val| ((val - complete_tree_algo_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let prune_tree_std = prune_tree_times
            .iter()
            .map(|&val| ((val - prune_tree_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let pruned_tree_algo_std = pruned_tree_algo_times
            .iter()
            .map(|&val| ((val - pruned_tree_algo_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let complete_pruned_tree_std = complete_pruned_tree_times
            .iter()
            .map(|&val| ((val - complete_pruned_tree_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let normal_algo_std = normal_algo_times
            .iter()
            .map(|&val| ((val - normal_algo_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;

        wtr.write_record(&[
            pos.to_string(),
            size.to_string(),
            tree_gen_mean.to_string(),
            tree_gen_std.to_string(),
            tree_algo_mean.to_string(),
            tree_algo_std.to_string(),
            complete_tree_algo_mean.to_string(),
            complete_tree_algo_std.to_string(),
            prune_tree_mean.to_string(),
            prune_tree_std.to_string(),
            pruned_tree_algo_mean.to_string(),
            pruned_tree_algo_std.to_string(),
            complete_pruned_tree_mean.to_string(),
            complete_pruned_tree_std.to_string(),
            normal_algo_mean.to_string(),
            normal_algo_std.to_string(),
        ])
        .unwrap();
        wtr.flush().unwrap();
        pos += 1;
    }
    wtr.flush().unwrap();
}
