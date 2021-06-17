build:
	cargo build --release

copy_python: build
	cp target/release/libultrametric_multiplication.so ultrametric_multiplication.so

test_python: copy_python
	python3 test.py

bench:
	cargo criterion