

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

### Quickstart Python

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

### Python Example
To run the Python examples You can try out the Python examples located in ```./examples/``` by running the following command:
```console
make python_example name=[example_name]
```

## License
[Apache-2.0 license](LICENSE)