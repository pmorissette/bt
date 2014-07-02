TMPREPO=/tmp/docs/bt

default: build_dev

.PHONY: dist upload docs pages serve klink

dist:
	python setup.py sdist

upload: dist
	twine upload dist/*

docs: 
	$(MAKE) -C docs/ clean
	$(MAKE) -C docs/ html

pages: 
	- rm -rf $(TMPREPO)
	git clone -b gh-pages git@github.com:pmorissette/bt.git $(TMPREPO)
	rm -rf $(TMPREPO)/*
	cp -r docs/build/html/* $(TMPREPO)
	cd $(TMPREPO); \
	git add -A ; \
	git commit -a -m 'auto-updating docs' ; \
	git push

serve:
	cd docs/build/html; \
	python -m SimpleHTTPServer

build_dev:
	- python setup.py build_ext --inplace

clean:
	- rm -rf build
	- rm -rf dist
	- rm -rf bt.egg-info
	- find . -name '*.so' -delete
	- find . -name '*.c' -delete

klink:
	git subtree pull --prefix=docs/source/_themes/klink --squash klink master
