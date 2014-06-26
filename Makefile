default: build_dev

.PHONY: dist

dist:
	python setup.py sdist upload

build_dev:
	- python setup.py build_ext --inplace

clean:
	- rm -rf build
	- find . -name '*.so' -delete
	- find . -name '*.c' -delete
