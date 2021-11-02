

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
ultrametric_matrix_tools = "*"
```
An example of the construction of the ultrametric tree and multiplication with it is:
```rust
use ultrametric_matrix_tools::na::{DMatrix, DVector};
use ultrametric_matrix_tools::UltrametricTree;

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

[More examples](#rust-example) can bef found in `./examples/`.

### Quickstart Python
You can install the current release by running:
```console
pip install ultrametric_matrix_tools
```
An example of the construction of the ultrametric tree and multiplication with it is:
```python
from ultrametric_matrix_tools import UltrametricTree
import numpy as np

matrix = np.array([[0.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 1.0], [
                  3.0, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
vector = np.array([4.0, 2.0, 7.0, 5.0])

tree = UltrametricTree(matrix)
product = tree.mult(vector)
```

[More examples](#python-example) can bef found in `./examples/`.

## Build
### Build Rust Library
The Rust library is build by running:
```console
cargo build --release
```
The compiled Rust library is located at `./target/release/` and can be copied from there.

### Build Python Module
The Python module is build from the Rust code using the [PyO3](https://github.com/PyO3/pyo3). To build the Python module, you need to [install Cargo](https://www.rust-lang.org/tools/install) and run:
```console
cargo build --release
```
The compiled Python module is located at `./target/release/` and can be copied from there.

To export the Python wheels from a Linux system run the following commands:

Linux (requires docker):
```console
docker run --rm -v $(pwd):/io konstin2/maturin build --release
```

Windows (requires mingw32-python and mingw64-python):
```console
make python_package_windows
```

Currently, cross-compiling to macOS is not supported.

## Examples
### Rust Example
You can try out the Rust examples, you need to [install Cargo](https://www.rust-lang.org/tools/install). You can try out the Python examples located in `./examples/` by running the following command:
```console
cargo run --release --example [example_name]
```
E.g. to run the multiplication example run:
```console
cargo run --release --example multiplication
```

### Python Example
To run the Python examples You can try out the Python examples located in `./examples/` by running the following command:
```console
make python_example name=[example_name]
```
E.g. to run the multiplication example run:
```console
make python_example name=multiplication
```

## License
[Apache-2.0 license](LICENSE)