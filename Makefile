name?=multiplication

build:
	cargo build --release

copy_python: build
	cp target/release/libultrametric_multiplication.so examples/ultrametric_multiplication.so

bench:
	cargo criterion

bench_performance:
	cargo criterion --bench performance

bench_jacobi:
	cargo criterion --bench jacobi_test

bench_test:
	cargo criterion --bench multiplication_test

bench_performance_multiple:
	cargo criterion --bench performance_multiple

python_example: copy_python
	python3 examples/$(name).py