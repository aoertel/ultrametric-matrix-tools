use criterion::{criterion_group, criterion_main, Criterion};
use graph6::*;
use nalgebra::{DMatrix, DVector};
use petgraph::algo::astar;
use petgraph::prelude::*;
use petgraph::visit::GetAdjacencyMatrix;
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use ultrametric_multiplication::RootedTreeVertex;

criterion_group!(benches, benchmark);
criterion_main!(benches);

fn benchmark(_c: &mut Criterion) {
    let size = 200;
    let matrix = generate_random_connectivity_matrix(size, 0.1);
    println!("{}", &matrix);
    let vector = generate_random_vector(size);
    println!("{}", &vector);

    let start_fast = SystemTime::now();
    let mut root = RootedTreeVertex::get_partition_tree(&matrix);
    let duration_tree_gen = start_fast.elapsed().unwrap();
    let fast_product = root.multiply_with_tree(&vector);
    let duration_fast_full = start_fast.elapsed().unwrap();

    let start_normal = SystemTime::now();
    let normal_product = calculate_normal_product(&matrix, &vector);
    let duration_normal = start_normal.elapsed().unwrap();

    println!("Fast product: {}", &fast_product);
    println!("Normal product: {}", &normal_product);
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

    //root.print_rooted_tree();

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

#[allow(unused)]
fn get_vertex_path_matrix(g: &StableUnGraph<(), ()>) -> DMatrix<f64> {
    let vertex_cnt = g.node_count();
    let mut path_matrix = DMatrix::<f64>::zeros(vertex_cnt, vertex_cnt);
    let path_matrix_mutex = Arc::new(Mutex::new(&mut path_matrix));
    (0..(vertex_cnt - 1)).into_par_iter().for_each(|i| {
        ((i + 1)..vertex_cnt).into_par_iter().for_each(|j| {
            let num_paths = get_num_vertex_disjoint_paths(&g, i as u32, j as u32);
            path_matrix_mutex.lock().unwrap()[(i, j)] = num_paths as f64;
            path_matrix_mutex.lock().unwrap()[(j, i)] = num_paths as f64;
        })
    });
    return path_matrix;
}

fn get_num_vertex_disjoint_paths(orig_g: &StableUnGraph<(), ()>, from: u32, to: u32) -> usize {
    let mut g = orig_g.clone();
    let mut num_paths = 0;
    loop {
        let shortest_path = astar(&g, from.into(), |v| v == to.into(), |_| 1, |_| 1);
        match shortest_path {
            None => return num_paths,
            Some((length, path_vertices)) => {
                num_paths += 1;
                for vertex in path_vertices {
                    if vertex == from.into() || vertex == to.into() {
                        continue;
                    }
                    g.remove_node(vertex);
                }
                if length == 1 {
                    let edge = g.find_edge(from.into(), to.into()).unwrap();
                    g.remove_edge(edge);
                }
            }
        }
    }
}

#[allow(unused)]
fn get_edge_path_matrix(g: &StableUnGraph<(), ()>) -> DMatrix<f64> {
    let vertex_cnt = g.node_count();
    let mut path_matrix = DMatrix::<f64>::zeros(vertex_cnt, vertex_cnt);
    let path_matrix_mutex = Arc::new(Mutex::new(&mut path_matrix));
    (0..(vertex_cnt - 1)).into_par_iter().for_each(|i| {
        ((i + 1)..vertex_cnt).into_par_iter().for_each(|j| {
            let num_paths = get_num_edge_disjoint_paths(&g, i as u32, j as u32);
            path_matrix_mutex.lock().unwrap()[(i, j)] = num_paths as f64;
            path_matrix_mutex.lock().unwrap()[(j, i)] = num_paths as f64;
        })
    });
    return path_matrix;
}

fn get_num_edge_disjoint_paths(orig_g: &StableUnGraph<(), ()>, from: u32, to: u32) -> usize {
    let mut g = orig_g.clone();
    let mut num_paths = 0;
    loop {
        let shortest_path = astar(&g, from.into(), |v| v == to.into(), |_| 1, |_| 1);
        match shortest_path {
            None => return num_paths,
            Some((length, path_vertices)) => {
                num_paths += 1;
                for first_id in 0..length {
                    let edge = g
                        .find_edge(
                            path_vertices[first_id].into(),
                            path_vertices[first_id + 1].into(),
                        )
                        .unwrap();
                    g.remove_edge(edge);
                }
            }
        }
    }
}

fn generate_empty_graph(vertices: usize) -> StableUnGraph<(), ()> {
    let mut g = StableUnGraph::<(), ()>::with_capacity(vertices, 0);
    for _ in 0..vertices {
        g.add_node(());
    }
    return g;
}

fn generate_random_graph(vertices: usize, edge_prob: f64) -> StableUnGraph<(), ()> {
    let mut g = generate_empty_graph(vertices);
    for from in 0..(vertices - 1) as u32 {
        for to in (from + 1)..vertices as u32 {
            let random_value: f64 = rand::thread_rng().gen();
            if random_value <= edge_prob {
                g.add_edge(from.into(), to.into(), ());
            }
        }
    }
    return g;
}

#[allow(unused)]
fn generate_example_matrix() -> DMatrix<f64> {
    let graph = generate_example_graph();
    let mut matrix = get_vertex_path_matrix(&graph);
    matrix[(2, 2)] = 4.0;
    matrix[(4, 4)] = 2.0;
    return matrix;
}

fn generate_example_graph() -> StableUnGraph<(), ()> {
    let mut g = generate_empty_graph(8);
    g.add_edge(0.into(), 4.into(), ());
    g.add_edge(0.into(), 7.into(), ());
    g.add_edge(4.into(), 7.into(), ());
    g.add_edge(5.into(), 7.into(), ());
    g.add_edge(2.into(), 3.into(), ());
    g.add_edge(1.into(), 5.into(), ());
    g.add_edge(1.into(), 3.into(), ());
    g.add_edge(1.into(), 6.into(), ());
    g.add_edge(3.into(), 5.into(), ());
    g.add_edge(3.into(), 6.into(), ());
    return g;
}

#[allow(unused)]
fn generate_random_connectivity_matrix(size: usize, edge_prob: f64) -> DMatrix<f64> {
    let mut rng = rand::thread_rng();
    let g = generate_random_graph(size, edge_prob);
    let mut matrix = get_vertex_path_matrix(&g);
    for i in 0..size {
        matrix[(i, i)] = rng.gen_range(0..size) as f64;
    }
    return matrix;
}

fn generate_random_vector(size: usize) -> DVector<f64> {
    let mut rng = rand::thread_rng();
    let mut vector = DVector::<f64>::zeros(size);
    for i in 0..size {
        vector[i] = rng.gen_range(1..size) as f64;
    }
    return vector;
}

#[allow(unused)]
fn from_graph6_string(s: &String) -> StableUnGraph<(), ()> {
    let (adj_mat, vertex_cnt) = string_to_adjacency_matrix(s);
    let mut g = generate_empty_graph(vertex_cnt);
    for from in 0..(vertex_cnt - 1) {
        for to in (from + 1)..vertex_cnt {
            if adj_mat[(from * vertex_cnt) + to] == 1. {
                g.add_edge((from as u32).into(), (to as u32).into(), ());
            }
        }
    }
    return g;
}

#[allow(unused)]
fn to_graph6_string(g: &StableUnGraph<(), ()>) -> String {
    let adj_mat = g.adjacency_matrix();
    let mut adj_vec: Vec<f32> = Vec::new();
    adj_vec.resize(adj_mat.len(), 0.0);
    for bit in adj_mat.ones() {
        adj_vec[bit] = 1.0;
    }
    return adjacency_matrix_to_string(&adj_vec, g.node_count());
}
