use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod ultrametric_tree;
pub mod utils;

pub use self::ultrametric_tree::UltrametricTree;
use crate::utils::PyInit_utils;

#[pymodule]
fn ultrametric_matrix_tools(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<UltrametricTree>()?;
    m.add_wrapped(wrap_pymodule!(utils))?;
    Ok(())
}
