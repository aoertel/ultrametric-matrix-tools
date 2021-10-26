use nalgebra::DMatrix;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;

pub mod ultrametric_tree;
pub mod utils;

pub use self::ultrametric_tree::*;
pub use self::utils::*;

#[pymodule]
fn ultrametric_tree(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<UltrametricTree>()?;
    #[pyfn(m)]
    #[pyo3(name = "is_ultrametric")]
    pub fn is_ultrametric_py(py_matrix: PyReadonlyArrayDyn<f64>) -> bool {
        let size = py_matrix.shape()[0];
        let py_array = py_matrix.as_array();
        let mut matrix = DMatrix::<f64>::zeros(size, size);
        for i in 0..size {
            for j in 0..size {
                matrix[(i, j)] = py_array[[i, j]];
            }
        }
        return is_ultrametric(&matrix);
    }

    Ok(())
}
