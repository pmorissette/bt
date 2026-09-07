# Develop bt

Use Python 3.11 for the development and documentation tools. bt's runtime still supports Python 3.9 and later. Install [uv](https://docs.astral.sh/uv/) and a C compiler for the Cython extension, then create an environment:

```bash
uv venv --python 3.11
source .venv/bin/activate
make develop
make lint
make checks
make coverage
make build
make test-dist
```

On Windows, activate with `.venv\Scripts\activate` instead. Run `make help` for available targets. `make test` runs tests without coverage; `make benchmark` runs the separate backtest benchmarks. Type checking (`make check-types`) is advisory and not a CI gate.

Coverage measures the Python modules. The compiled `bt.core` extension is still tested, but excluded from coverage because it is built without Cython line tracing.

## Build the documentation

```bash
make docs-develop
make docs
make serve
```

Open <http://localhost:9087>. Yardang reads `[tool.yardang]` in `pyproject.toml`, generates the homepage from `README.md`, and uses the repository root as the Sphinx source directory. Output goes into `docs/html`, using the installed Klink theme. The old Sphinx Makefiles, hand-maintained `conf.py`, vendored theme, and manual `make pages` deployment are no longer used.

`docs/requirements.txt` pins released Yardang and Klink packages; no unreleased Yardang changes are required. `docs/build.py` uses Yardang's configuration generator and Sphinx's public API to register Klink's theme path, resolve static assets, configure ffn cross-references, and treat warnings as errors. These small compatibility overrides can be removed as upstream support becomes available.

Edit the landing page in `README.md` and API reference in `docs/source/overview.md`. Other guides and notebook examples still use their checked-in reStructuredText sources and images; documentation builds do not execute notebooks or fetch market data. After editing notebooks, install Pandoc and regenerate the exports with Klink's conversion helper:

```bash
uv pip install nbconvert
cd docs/source
python -c 'import klink; klink.convert_notebooks()'
```

This converts saved notebook outputs without executing cells, copies images into `_static`, and updates image paths and Klink CSS classes. Review and commit the changed notebooks, exports, and images together.

Pull requests build documentation and upload an HTML artifact. Successful pushes to `master` also publish to the existing `gh-pages` branch. Local builds never publish.

## Continue the Markdown migration

The first increment adopts Yardang's README landing page and moves `bt.rst` to `overview.md`. Autodoc directives remain inside MyST `eval-rst` fences so existing RST docstrings and API anchors render correctly, as described in the [MyST autodoc guide](https://myst-parser.readthedocs.io/en/stable/faq/index.html#use-sphinx-ext-autodoc-in-markdown-files). The remaining conversion is separate from the build migration:

1. Convert the hand-written installation, algorithm, and tree guides to Markdown, preserving Sphinx cross-references and heading anchors. Keep tutorials, task guides, and API reference distinct.
1. Convert raw reStructuredText cells in `intro.ipynb` and `examples-nb.ipynb` to Markdown before exporting notebooks. Use saved outputs, retain image paths and Klink styling, and replace each RST export with its Markdown equivalent in the same change. Do not leave both formats with the same basename.
1. Convert the examples index and its RST includes to MyST equivalents. Remove the excluded legacy `docs/source/index.rst` once any useful material has been incorporated into the landing page or guides.

For each conversion, run `make docs` and check navigation, images, API links, and heading links in a browser. Preserve page basenames where possible; maintain `[tool.yardang.redirects]` when paths change. The current redirects preserve legacy page URLs, including `bt.html`, and forward their fragments; renamed headings need explicit compatibility anchors.

## Update the Copier template

`.copier-answers.yaml` records the Cython variant of [python-project-templates/base](https://github.com/python-project-templates/base) and the exact template revision. From a clean branch, update with:

```bash
copier update --answers-file .copier-answers.yaml --trust
```

Review the resulting diff and resolve conflicts before running the checks above. The Python Templates Copier Update GitHub App can propose updates automatically; installing that app is separate from this repository change.

bt intentionally retains its MIT license, runtime Python floor, package metadata and version, Cython build hook, native wheel matrix, top-level `tests` directory, benchmarks, and Ruff line length. Documentation adopts the template's Markdown homepage while retaining Klink styling and redirecting legacy page URLs. The docs workflow builds from source because bt's CI produces platform-specific wheel artifacts.

The Cython template supplies shared Python/compiler setup, native-wheel CI, and distribution smoke tests. CI builds wheels from clean sdists instead of deleting compiled files from the checkout. `make test-dist` checks installed native extensions outside the source tree and rebuilds sdists into wheels for the same check. bt retains its full source-tree test suite; cibuildwheel verifies that each installed wheel imports the compiled `bt.core` extension.

The build workflow follows the template's Python 3.11–3.14 matrix on Linux x86_64/ARM64, macOS ARM64, and Windows x86_64. The separate release workflow retains its existing Python versions and macOS architecture selection. The runtime minimum remains unchanged.
