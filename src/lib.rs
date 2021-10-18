use pyo3::prelude::*;

pub mod rooted_tree;
pub mod utils;

pub use self::rooted_tree::*;
pub use self::utils::*;

#[pymodule]
fn ultrametric_multiplication(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RootedTreeVertex>()?;
    Ok(())
}
