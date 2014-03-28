default: build_dev

build_dev:
	- python setup.py build_ext --inplace

clean:
	- rm -rf build
	- find . -name '*.so' -delete
	- find . -name '*.c' -delete
