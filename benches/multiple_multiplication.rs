mod utils;

use criterion::{criterion_group, criterion_main, Criterion};
use std::io;
use std::time::SystemTime;
use ultrametric_matrix_tools::UltrametricTree;

criterion_group!(benches, benchmark_multiple);
criterion_main!(benches);

const MATRIX_SIZES: [usize; 8] = [10, 100, 250, 500, 1_000, 2_500, 5_000, 10_000];
const NUM_SAMPLES: u32 = 1000;
const HEADER_MULTIPLE: [&str; 17] = [
    "pos",
    "size",
    "iterations",
    "tree_gen_mean",
    "tree_gen_std",
    "tree_mult_mean",
    "tree_mult_std",
    "complete_tree_mult_mean",
    "complete_tree_mult_std",
    "prune_tree_mean",
    "prune_tree_std",
    "pruned_tree_mult_mean",
    "pruned_tree_mult_std",
    "complete_pruned_tree_mult_mean",
    "complete_pruned_tree_mult_std",
    "normal_mult_mean",
    "normal_mult_std",
];
const NUM_ITERATIONS: usize = 1000;

#[allow(unused)]
fn benchmark_multiple(_c: &mut Criterion) {
    let mut pos = 0;
    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b',')
        .from_writer(io::stdout());
    wtr.write_record(&HEADER_MULTIPLE).unwrap();
    for &size in MATRIX_SIZES.iter() {
        let mut tree_gen_times: Vec<f64> = Vec::new();
        let mut tree_mult_times: Vec<f64> = Vec::new();
        let mut complete_tree_mult_times: Vec<f64> = Vec::new();
        let mut normal_mult_times: Vec<f64> = Vec::new();
        let mut prune_tree_times: Vec<f64> = Vec::new();
        let mut pruned_tree_mult_times: Vec<f64> = Vec::new();
        let mut complete_pruned_tree_mult_times: Vec<f64> = Vec::new();
        for i in 0..NUM_SAMPLES {
            let matrix = utils::random_ultrametric_matrix(size);
            let vector = utils::random_vector(size);

            let start_tree_gen = SystemTime::now();
            let mut root = UltrametricTree::from_matrix(&matrix);
            let duration_tree_gen = start_tree_gen.elapsed().unwrap();
            let start_tree_mult = SystemTime::now();
            for _ in 0..NUM_ITERATIONS {
                root.mult(&vector);
            }
            let duration_tree_mult = start_tree_mult.elapsed().unwrap();
            tree_gen_times.push(duration_tree_gen.as_secs_f64());
            tree_mult_times.push(duration_tree_mult.as_secs_f64());
            complete_tree_mult_times
                .push(duration_tree_gen.as_secs_f64() + duration_tree_mult.as_secs_f64());

            let start_prune_tree = SystemTime::now();
            root.prune_tree();
            let duration_prune_tree = start_prune_tree.elapsed().unwrap();
            let start_pruned_tree_mult = SystemTime::now();
            for _ in 0..NUM_ITERATIONS {
                root.mult(&vector);
            }
            let duration_pruned_tree_mult = start_pruned_tree_mult.elapsed().unwrap();
            prune_tree_times.push(duration_prune_tree.as_secs_f64());
            pruned_tree_mult_times.push(duration_pruned_tree_mult.as_secs_f64());
            complete_pruned_tree_mult_times.push(
                duration_tree_gen.as_secs_f64()
                    + duration_prune_tree.as_secs_f64()
                    + duration_pruned_tree_mult.as_secs_f64(),
            );

            let start_normal = SystemTime::now();
            for _ in 0..NUM_ITERATIONS {
                utils::calculate_normal_product(&matrix, &vector);
            }
            let duration_normal = start_normal.elapsed().unwrap();
            normal_mult_times.push(duration_normal.as_secs_f64());
        }

        let tree_gen_mean = tree_gen_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let tree_mult_mean = tree_mult_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let complete_tree_mult_mean =
            complete_tree_mult_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let prune_tree_mean = prune_tree_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let pruned_tree_mult_mean = pruned_tree_mult_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let complete_pruned_tree_mult_mean =
            complete_pruned_tree_mult_times.iter().sum::<f64>() / NUM_SAMPLES as f64;
        let normal_mult_mean = normal_mult_times.iter().sum::<f64>() / NUM_SAMPLES as f64;

        let tree_gen_std = tree_gen_times
            .iter()
            .map(|&val| ((val - tree_gen_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let tree_mult_std = tree_gen_times
            .iter()
            .map(|&val| ((val - tree_mult_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let complete_tree_mult_std = complete_tree_mult_times
            .iter()
            .map(|&val| ((val - complete_tree_mult_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let prune_tree_std = prune_tree_times
            .iter()
            .map(|&val| ((val - prune_tree_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let pruned_tree_mult_std = pruned_tree_mult_times
            .iter()
            .map(|&val| ((val - pruned_tree_mult_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let complete_pruned_tree_mult_std = complete_pruned_tree_mult_times
            .iter()
            .map(|&val| ((val - complete_pruned_tree_mult_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;
        let normal_mult_std = tree_gen_times
            .iter()
            .map(|&val| ((val - normal_mult_mean).powi(2)))
            .sum::<f64>()
            / NUM_SAMPLES as f64;

        wtr.write_record(&[
            pos.to_string(),
            size.to_string(),
            NUM_ITERATIONS.to_string(),
            tree_gen_mean.to_string(),
            tree_gen_std.to_string(),
            tree_mult_mean.to_string(),
            tree_mult_std.to_string(),
            complete_tree_mult_mean.to_string(),
            complete_tree_mult_std.to_string(),
            prune_tree_mean.to_string(),
            prune_tree_std.to_string(),
            pruned_tree_mult_mean.to_string(),
            pruned_tree_mult_std.to_string(),
            complete_pruned_tree_mult_mean.to_string(),
            complete_pruned_tree_mult_std.to_string(),
            normal_mult_mean.to_string(),
            normal_mult_std.to_string(),
        ])
        .unwrap();
        wtr.flush().unwrap();
        pos += 1;
    }
    wtr.flush().unwrap();
}
