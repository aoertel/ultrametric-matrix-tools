name?=multiplication

build:
	cargo build --release

copy_python: build
	cp target/release/libultrametric_multiplication.so examples/ultrametric_multiplication.so

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