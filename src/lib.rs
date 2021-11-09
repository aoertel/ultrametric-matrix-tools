//! This crate is a toolbox that provides functions and data structures to generate and handle ultrametric matrices.
//!
//! The [`UltrametricTree`](ultrametric_tree::UltrametricTree) is a data structure that represents the structure of an ultrametric matrix. This tree can then be used to efficiently implement algorithms, e.g. multiplication of ultramettric matrix and vector.
//!
//! The [`utils`](utils) module provides functions to generate random ultrametric matrices and check if a matrix is ultrametric.

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod ultrametric_tree;
pub mod utils;

pub use self::ultrametric_tree::UltrametricTree;
use crate::utils::PyInit_utils;
pub extern crate nalgebra as na;

#[pymodule]
fn ultrametric_matrix_tools(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<UltrametricTree>()?;
    m.add_wrapped(wrap_pymodule!(utils))?;
    Ok(())
}
