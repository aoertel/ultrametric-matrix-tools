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
    fn new(num_leafs: usize) -> RootedTree {
        let vec: Vec<usize> = vec![0; num_leafs];
        RootedTree {
            leafs: vec,
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
    matrix[(2, 2)] = 4.0;
    matrix[(4, 4)] = 2.0;
    //let matrix = generate_random_matrix(10);
    println!("{}", &matrix);

    let vector = DVector::<f64>::from_iterator(8, vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    //let vector = generate_random_vector(10);
    println!("{}", &vector);

    let mut tree = get_partition_tree(&matrix);
    let fast_product = multiply_with_tree(&mut tree, &vector);
    println!("Fast product: {}", &fast_product);
    let normal_product = matrix * vector;
    println!("Normal product: {}", &normal_product);
    if normal_product == fast_product {
        println!("Works!");
    } else {
        println!("Does not work");
    }
}

fn get_partition_tree(matrix: &DMatrix<f64>) -> RootedTree {
    let vertex_ids: Vec<usize> = (0..matrix.nrows()).collect();
    let mut tree = RootedTree::new(matrix.nrows());
    tree.add_vertex(vertex_ids);
    partition_tree_vertex(&matrix, &mut tree, 0);

    return tree;
}

fn partition_tree_vertex(matrix: &DMatrix<f64>, tree: &mut RootedTree, parent: usize) {
    let first_i = tree.vertices[parent].partition[0];
    if tree.vertices[parent].partition.len() == 1 {
        tree.leafs[first_i] = parent;
        tree.vertices[parent].level_diff = matrix[(first_i, first_i)];
        tree.vertices[parent].level = tree.vertices[tree.vertices[parent].parent.unwrap()].level;
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

    partition_tree_vertex(matrix, tree, left_child);
    partition_tree_vertex(matrix, tree, right_child);
}

fn multiply_with_tree(tree: &mut RootedTree, vector: &DVector<f64>) -> DVector<f64> {
    calculate_sums(tree, 0, vector);
    let mut product: DVector<f64> = DVector::<f64>::zeros(vector.nrows());
    calculate_full_product(tree, 0, &mut product, 0.0);
    /*
    for i in 0..vector.nrows() {
        let leaf = tree.leafs[i];
        product[i] = calculate_single_product(&mut tree, leaf);
    }
    */
    return product;
}

fn calculate_sums(tree: &mut RootedTree, current: usize, vector: &DVector<f64>) -> f64 {
    if tree.vertices[current].partition.len() == 1 {
        let sum = vector[tree.vertices[current].partition[0]];
        tree.vertices[current].sum =
            Some((tree.vertices[current].level_diff - tree.vertices[current].level) * sum);
        return sum;
    }
    let left_child = tree.vertices[current].left_child.unwrap();
    let right_child = tree.vertices[current].right_child.unwrap();
    let left_sum = calculate_sums(tree, left_child, vector);
    let right_sum = calculate_sums(tree, right_child, vector);
    let sum = left_sum + right_sum;
    tree.vertices[current].sum = Some(sum * tree.vertices[current].level_diff);
    return sum;
}

fn calculate_full_product(
    tree: &RootedTree,
    current: usize,
    product: &mut DVector<f64>,
    prev_sum: f64,
) {
    let sum = prev_sum + tree.vertices[current].sum.unwrap();
    if tree.vertices[current].partition.len() == 1 {
        product[tree.vertices[current].partition[0]] = sum;
        return;
    }
    let left_child = tree.vertices[current].left_child.unwrap();
    let right_child = tree.vertices[current].right_child.unwrap();
    calculate_full_product(tree, left_child, product, sum);
    calculate_full_product(tree, right_child, product, sum);
}

fn calculate_single_product(mut tree: &mut RootedTree, current: usize) -> f64 {
    if current == 0 {
        return tree.vertices[0].sum.unwrap();
    } else {
        let parent = tree.vertices[current].parent.unwrap();
        return tree.vertices[current].sum.unwrap() + calculate_single_product(&mut tree, parent);
    }
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

//TODO: this function does not work, work out a way to generate a random ultrametric matrix
fn generate_random_matrix(size: usize) -> DMatrix<f64> {
    let mut rng = rand::thread_rng();
    let mut matrix = DMatrix::<f64>::zeros(size, size);
    for i in 1..size {
        matrix[(0, i)] = rng.gen_range(1..size) as f64;
    }
    for i in 1..size {
        for j in (i + 1)..size {
            let left_side = matrix[(0, j)];
            let right_side = matrix[(0, i)];
            if right_side <= left_side {
                let new_element = rng.gen_range(1..size) as f64;
                matrix[(i, j)] = new_element;
                matrix[(j, i)] = new_element;
            } else {
                let new_element = rng.gen_range(1..(left_side as usize + 1)) as f64;
                matrix[(i, j)] = left_side;
                matrix[(j, i)] = left_side;
            }
        }
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
