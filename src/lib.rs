use pyo3::prelude::*;

pub mod rooted_tree;
pub mod test_functions;
pub mod utils;

pub use self::rooted_tree::*;
pub use self::test_functions::*;
pub use self::utils::*;

#[pymodule(ultrametric_multiplication)]
fn mymodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RootedTreeVertex>()?;
    m.add_class::<TestFunctions>()?;
    Ok(())
}
