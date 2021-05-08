use pyo3::prelude::*;

pub mod rooted_tree;

pub use self::rooted_tree::*;

#[pymodule(ultrametric_multiplication)]
fn mymodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RootedTreeVertex>()?;
    Ok(())
}
