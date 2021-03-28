extern crate nalgebra as na;

use graph6::*;
use na::{DMatrix, DVector};
use petgraph::algo::astar;
use petgraph::prelude::*;
use petgraph::visit::GetAdjacencyMatrix;
use rand::Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};

#[derive(Default, Debug, Clone)]
struct RootedTreeVertex {
    partition: Vec<usize>,
    level: f64,
    level_diff: f64,
    sum: f64,
    left_child: Option<Box<RootedTreeVertex>>,
    right_child: Option<Box<RootedTreeVertex>>,
}

impl RootedTreeVertex {
    fn new(partition: Vec<usize>) -> RootedTreeVertex {
        RootedTreeVertex {
            partition: partition,
            ..Default::default()
        }
    }
}

fn main() {
    /*
    let input = File::open("graph4c.g6");
    let input_file = match input {
        Ok(file) => file,
        Err(_) => return,
    };
    let mut lines = io::BufReader::new(input_file).lines();
    let line = lines.next();
    */

    let matrix = generate_random_connectivity_matrix(100, 0.5);
    println!("{}", &matrix);

    let vector = generate_random_vector(100);
    println!("{}", &vector);

    let mut root = get_partition_tree(&matrix);
    let fast_product = multiply_with_tree(&mut root, &vector);
    println!("Fast product: {}", &fast_product);
    let normal_product = matrix * vector;
    println!("Normal product: {}", &normal_product);
    if normal_product == fast_product {
        println!("Works!");
    } else {
        println!("Does not work");
    }
}

fn get_partition_tree(matrix: &DMatrix<f64>) -> RootedTreeVertex {
    let vertex_ids: Vec<usize> = (0..matrix.nrows()).collect();
    let mut root = RootedTreeVertex::new(vertex_ids);
    partition_tree_vertex(&matrix, &mut root, 0.0);

    return root;
}

fn partition_tree_vertex(matrix: &DMatrix<f64>, current: &mut RootedTreeVertex, parent_level: f64) {
    let first_i = current.partition[0];
    if current.partition.len() == 1 {
        current.level_diff = matrix[(first_i, first_i)];
        current.level = parent_level;
        return;
    }
    let mut left_partition: Vec<usize> = vec![first_i];
    let mut right_partition: Vec<usize> = Vec::new();
    let mut min = f64::MAX;
    for &i in &current.partition[1..] {
        if min > matrix[(first_i, i)] {
            min = matrix[(first_i, i)];
            left_partition.extend(right_partition.iter());
            right_partition.clear();
            right_partition.push(i);
        } else {
            if min == matrix[(first_i, i)] {
                right_partition.push(i);
            } else {
                left_partition.push(i);
            }
        }
    }
    current.level = min;
    current.level_diff = min - parent_level;

    let left_child = Box::new(RootedTreeVertex::new(left_partition));
    let right_child = Box::new(RootedTreeVertex::new(right_partition));
    current.left_child = Some(left_child);
    current.right_child = Some(right_child);

    partition_tree_vertex(matrix, current.left_child.as_mut().unwrap(), current.level);
    partition_tree_vertex(matrix, current.right_child.as_mut().unwrap(), current.level);
}

fn multiply_with_tree(root: &mut RootedTreeVertex, vector: &DVector<f64>) -> DVector<f64> {
    calculate_sums(root, vector);
    let mut product: DVector<f64> = DVector::<f64>::zeros(vector.nrows());
    calculate_full_product(root, &mut product, 0.0);
    return product;
}

fn calculate_sums(current: &mut RootedTreeVertex, vector: &DVector<f64>) -> f64 {
    if current.partition.len() == 1 {
        let sum = vector[current.partition[0]];
        current.sum = (current.level_diff - current.level) * sum;
        return sum;
    }

    let left_sum = calculate_sums(current.left_child.as_mut().unwrap(), vector);
    let right_sum = calculate_sums(current.right_child.as_mut().unwrap(), vector);
    let sum = left_sum + right_sum;
    current.sum = sum * current.level_diff;
    return sum;
}

fn calculate_full_product(current: &RootedTreeVertex, product: &mut DVector<f64>, prev_sum: f64) {
    let sum = prev_sum + current.sum;
    if current.partition.len() == 1 {
        product[current.partition[0]] = sum;
        return;
    }
    calculate_full_product(current.left_child.as_ref().unwrap(), product, sum);
    calculate_full_product(current.right_child.as_ref().unwrap(), product, sum);
}

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

fn to_graph6_string(g: &StableUnGraph<(), ()>) -> String {
    let adj_mat = g.adjacency_matrix();
    let mut adj_vec: Vec<f32> = Vec::new();
    adj_vec.resize(adj_mat.len(), 0.0);
    for bit in adj_mat.ones() {
        adj_vec[bit] = 1.0;
    }
    return adjacency_matrix_to_string(&adj_vec, g.node_count());
}
