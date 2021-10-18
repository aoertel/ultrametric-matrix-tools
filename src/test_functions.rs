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
