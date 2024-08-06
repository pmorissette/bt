TMPREPO=/tmp/docs/bt

default: build_dev

.PHONY: dist upload docs pages serve klink notebooks test lint fix develop

develop:
	python -m pip install -e .[dev]

test:
	python -m pytest -vvv tests --cov=bt --junitxml=python_junit.xml --cov-report=xml --cov-branch --cov-report term

lint:
	python -m ruff check bt setup.py docs/source/conf.py
	python -m ruff format --check bt setup.py docs/source/conf.py

fix:
	python -m ruff check --fix bt setup.py docs/source/conf.py
	python -m ruff format bt setup.py docs/source/conf.py

dist:
	python setup.py sdist
	python -m twine check dist/*

upload: dist
	python -m twine upload dist/* --skip-existing

docs:
	$(MAKE) -C docs/ clean
	$(MAKE) -C docs/ html

pages:
	rm -rf $(TMPREPO)
	git clone -b gh-pages git@github.com:pmorissette/bt.git $(TMPREPO)
	rm -rf $(TMPREPO)/*
	cp -r docs/build/html/* $(TMPREPO)
	cd $(TMPREPO);\
	git add -A ;\
	git commit -a -m 'auto-updating docs' ;\
	git push

serve:
	cd docs/build/html; \
	python -m http.server 9087

build_dev:
	python setup.py build_ext --inplace

clean:
	rm -rf build
	rm -rf dist
	rm -rf bt.egg-info
	find . -name '*.so' -delete
	find . -name '*.c' -delete

klink:
	git subtree pull --prefix=docs/source/_themes/klink --squash klink master

notebooks:
	cd docs/source; \
	jupyter notebook --no-browser --ip=*
