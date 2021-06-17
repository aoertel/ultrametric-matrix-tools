use nalgebra::DMatrix;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use petgraph::algo::astar;
use petgraph::prelude::*;
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

#[pyclass]
pub struct TestFunctions {}

#[pymethods]
impl TestFunctions {
    #[new]
    pub fn new() -> Self {
        TestFunctions {}
    }

    pub fn generate_random_connectivity_matrix<'py>(
        &self,
        py: Python<'py>,
        size: usize,
        edge_prob: f64,
    ) -> &'py PyArray2<f64> {
        let mut rng = rand::thread_rng();
        let g = generate_random_graph(size, edge_prob);
        let mut matrix = get_vertex_path_matrix(&g);
        for i in 0..size {
            matrix[(i, i)] = rng.gen_range(0..size) as f64;
        }
        let mut py_matrix = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                py_matrix[[i, j]] = matrix[(i, j)];
            }
        }
        return py_matrix.into_pyarray(py);
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
