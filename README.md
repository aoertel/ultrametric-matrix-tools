

## Table of Contents <!-- omit in toc -->
- [Quickstart](#quickstart)
  - [Quickstart Rust](#quickstart-rust)
  - [Quickstart Python](#quickstart-python)
- [Build](#build)
  - [Build Rust Library](#build-rust-library)
  - [Build Python Module](#build-python-module)
- [Examples](#examples)
  - [Rust Example](#rust-example)
  - [Python Example](#python-example)
- [License](#license)

## Quickstart
### Quickstart Rust
Add the following to the ```Cargo.toml``` file:
```toml
[dependencies]
# TODO: replace the * by the latest version.
ultrametric_tree = "*"
```
An example of the construction of the ultrametric tree and multiplication with it is:
```rust
use nalgebra::{DMatrix, DVector};
use ultrametric_tree::UltrametricTree;

fn main() {
    let matrix = DMatrix::from_vec(
        4,
        4,
        vec![
            0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    );
    let vector = DVector::from_vec(vec![4.0, 2.0, 7.0, 5.0]);
    let tree = UltrametricTree::from_matrix(&matrix);
    let product = tree * vector;
}

```

### Quickstart Python
You can install the current release by running:
```console
pip install ultrametric_tree
```
An example of the construction of the ultrametric tree and multiplication with it is:
```python
from ultrametric_tree import UltrametricTree
import numpy as np

matrix = np.array([[0.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 1.0], [
                  3.0, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
vector = np.array([4.0, 2.0, 7.0, 5.0])

tree = UltrametricTree(matrix)
product = tree.mult(vector)
```

## Build
### Build Rust Library
The Rust library is build by running:
```console
cargo build --release
```
The compiled Rust library is located at ```./target/release/``` and can be copied from there.

### Build Python Module
The Python module is build from the Rust code using the [PyO3](https://github.com/PyO3/pyo3). To build the Python module, you need to [install Cargo](https://www.rust-lang.org/tools/install) and run:
```console
cargo build --release
```
The compiled Python module is located at ```./target/release/``` and can be copied from there.

## Examples
### Rust Example
You can try out the Rust examples, you need to [install Cargo](https://www.rust-lang.org/tools/install). You can try out the Python examples located in ```./examples/``` by running the following command:
```console
cargo run --release --example [example_name]
```
E.g. to run the multiplication example run:
```console
cargo run --release --example multiplication
```

### Python Example
To run the Python examples You can try out the Python examples located in ```./examples/``` by running the following command:
```console
make python_example name=[example_name]
```
E.g. to run the multiplication example run:
```console
make python_example name=multiplication
```

## License
[Apache-2.0 license](LICENSE)