use graph6::*;
use nalgebra::{DMatrix, DVector};
use petgraph::algo::astar;
use petgraph::prelude::*;
use petgraph::visit::GetAdjacencyMatrix;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

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
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    for from in 0..(vertices - 1) as u32 {
        for to in (from + 1)..vertices as u32 {
            let random_value: f64 = rng.gen();
            if random_value <= edge_prob {
                g.add_edge(from.into(), to.into(), ());
            }
        }
    }
    return g;
}

#[allow(unused)]
pub fn generate_example_matrix() -> DMatrix<f64> {
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
pub fn generate_random_connectivity_matrix(size: usize, edge_prob: f64) -> DMatrix<f64> {
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    let g = generate_random_graph(size, edge_prob);
    let mut matrix = get_vertex_path_matrix(&g);
    for i in 0..size {
        matrix[(i, i)] = rng.gen_range(0..size) as f64;
    }
    return matrix;
}

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
pub fn from_graph6_string(s: &String) -> StableUnGraph<(), ()> {
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
pub fn to_graph6_string(g: &StableUnGraph<(), ()>) -> String {
    let adj_mat = g.adjacency_matrix();
    let mut adj_vec: Vec<f32> = Vec::new();
    adj_vec.resize(adj_mat.len(), 0.0);
    for bit in adj_mat.ones() {
        adj_vec[bit] = 1.0;
    }
    return adjacency_matrix_to_string(&adj_vec, g.node_count());
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

#[allow(unused)]
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
