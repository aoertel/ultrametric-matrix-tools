name?=multiplication

build:
	cargo build --release

copy_python: build
	cp target/release/libultrametric_matrix_tools.so examples/ultrametric_matrix_tools.so

bench:
	cargo criterion

bench_single:
	cargo criterion --bench single_multiplication

bench_jacobi:
	cargo criterion --bench jacobi_method

bench_multiple:
	cargo criterion --bench multiple_multiplication

python_example: copy_python
	python3 examples/$(name).py

python_package_windows:
	export PYO3_CROSS_LIB_DIR="/usr/x86_64-w64-mingw32/"
	maturin build --target x86_64-pc-windows-gnu --release
	export PYO3_CROSS_LIB_DIR="/usr/i686-w64-mingw32/"
	maturin build --target i686-pc-windows-gnu --release

python_package_macos:
	@echo Currently macOS is not supported!