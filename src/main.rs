extern crate nalgebra as na;

use graph6::*;
use na::{DMatrix, DVector};
use petgraph::algo::astar;
use petgraph::prelude::*;
use petgraph::visit::GetAdjacencyMatrix;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};

#[derive(Default, Debug, Clone)]
struct RootedTreeVertex {
    partition: Vec<usize>,
    level: f64,
    level_diff: f64,
    sum: Option<f64>,
    parent: Option<usize>,
    left_child: Option<usize>,
    right_child: Option<usize>,
}

impl RootedTreeVertex {
    fn new(partition: Vec<usize>) -> RootedTreeVertex {
        RootedTreeVertex {
            partition: partition,
            ..Default::default()
        }
    }
}

#[derive(Default, Debug)]
struct RootedTree {
    vertices: Vec<RootedTreeVertex>,
    leafs: Vec<usize>,
}

impl RootedTree {
    fn new() -> RootedTree {
        RootedTree {
            ..Default::default()
        }
    }

    fn add_vertex(&mut self, partition: Vec<usize>) -> usize {
        self.vertices.push(RootedTreeVertex::new(partition));
        return self.vertices.len() - 1;
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

    let graph = generate_example_graph();
    let mut matrix = get_vertex_path_matrix(&graph);
    matrix[(1, 1)] = 2.0;
    matrix[(4, 4)] = 4.0;
    println!("{}", &matrix);
    let vector = DVector::<f64>::from_iterator(8, vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    println!("{}", &vector);

    let mut tree = get_partition_tree(&matrix);
    dbg!(&tree);
    let product = multiply_with_tree(&mut tree, &vector);
    println!("{}", &product);
}

fn get_partition_tree(matrix: &DMatrix<f64>) -> RootedTree {
    let vertex_ids: Vec<usize> = (0..matrix.nrows()).collect();
    let mut tree = RootedTree::new();
    tree.add_vertex(vertex_ids);
    partition_tree_vertex(&matrix, &mut tree, 0);

    return tree;
}

fn partition_tree_vertex(matrix: &DMatrix<f64>, mut tree: &mut RootedTree, parent: usize) {
    let first_i = tree.vertices[parent].partition[0];
    if tree.vertices[parent].partition.len() == 1 {
        tree.leafs.push(parent);
        tree.vertices[parent].level = matrix[(first_i, first_i)];
        tree.vertices[parent].level_diff =
            -tree.vertices[tree.vertices[parent].parent.unwrap()].level;
        return;
    }
    let mut left_partition: Vec<usize> = vec![first_i];
    let mut right_partition: Vec<usize> = Vec::new();
    let mut min = f64::MAX;
    for &i in &tree.vertices[parent].partition[1..] {
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
    tree.vertices[parent].level = min;
    if parent == 0 {
        tree.vertices[parent].level_diff = min;
    } else {
        tree.vertices[parent].level_diff =
            min - tree.vertices[tree.vertices[parent].parent.unwrap()].level;
    }

    let left_child = tree.add_vertex(left_partition);
    let right_child = tree.add_vertex(right_partition);
    tree.vertices[parent].left_child = Some(left_child);
    tree.vertices[parent].right_child = Some(right_child);
    tree.vertices[left_child].parent = Some(parent);
    tree.vertices[right_child].parent = Some(parent);

    partition_tree_vertex(matrix, &mut tree, left_child);
    partition_tree_vertex(matrix, &mut tree, right_child);
}

fn multiply_with_tree(mut tree: &mut RootedTree, vector: &DVector<f64>) -> DVector<f64> {
    calculate_sums(&mut tree, 0, vector);

    return vector.clone();
}

fn calculate_sums(mut tree: &mut RootedTree, current: usize, vector: &DVector<f64>) -> f64 {
    if tree.vertices[current].partition.len() == 1 {
        return vector[tree.vertices[current].partition[0]];
    }
    let left_child = tree.vertices[current].left_child.unwrap();
    let right_child = tree.vertices[current].right_child.unwrap();
    let left_sum = calculate_sums(&mut tree, left_child, vector);
    let right_sum = calculate_sums(&mut tree, right_child, vector);
    let sum = left_sum + right_sum;
    tree.vertices[current].sum = Some(sum);
    return sum;
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
