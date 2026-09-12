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
make develop
make docs
make serve
```

Open <http://localhost:9087>. Yardang reads `[tool.yardang]` in `pyproject.toml`, generates the homepage from `README.md`, and uses the repository root as the Sphinx source directory. Output goes into `docs/html`, using the installed Klink theme. The old Sphinx Makefiles, hand-maintained `conf.py`, vendored theme, and manual `make pages` deployment are no longer used.

Documentation dependencies live in `pyproject.toml` under the `develop` extra, alongside the other development tools. Both local setup and documentation CI install that extra, including `klink>=0.1.13` for the sidebar and footnote fixes. There is no separate documentation requirements file.

`make docs` runs `yardang build --warning-is-error`, then copies `docs/source/_static` into the built site's `_static` directory. ffn cross-references are configured in `[tool.yardang.intersphinx-mapping]`; Yardang 0.10.0 or later supports this setting. No custom Python build wrapper is needed.

Edit the landing page in `README.md`, guides in `docs/source/*.md`, and API reference in `docs/source/overview.md`. Keep tutorials, task guides, and API reference distinct. Autodoc directives remain inside MyST `eval-rst` fences so existing RST docstrings and API anchors render correctly, as described in the [MyST autodoc guide](https://myst-parser.readthedocs.io/en/stable/faq/index.html#use-sphinx-ext-autodoc-in-markdown-files).

Notebook examples use checked-in Markdown exports and images; documentation builds do not execute notebooks or fetch market data. After editing notebooks, regenerate the exports from their saved outputs:

```bash
uv pip install nbconvert
cd docs/source
jupyter nbconvert --to markdown --template-file=../notebook.md.j2 --NbConvertApp.output_files_dir=_static *.ipynb
python -m mdformat *.md
```

The export template preserves image paths, retina dimensions, and Klink CSS classes. Pandoc is not required. Review and commit the changed notebooks, exports, and images together.

Pull requests build documentation and upload an HTML artifact. Successful pushes to `master` also publish to the existing `gh-pages` branch. Local builds never publish.

Run `make docs` after documentation changes and check navigation, images, API links, and heading links in a browser. Preserve page basenames where possible; maintain `[tool.yardang.redirects]` when paths change. The redirects preserve legacy page URLs, including `bt.html`, and forward their fragments; renamed headings need explicit compatibility anchors. `docs/source/examples.md` includes the individual Markdown examples, which also retain their standalone URLs.

## Update the Copier template

`.copier-answers.yaml` records the Cython variant of [python-project-templates/base](https://github.com/python-project-templates/base) and the exact template revision. From a clean branch, update with:

```bash
copier update --answers-file .copier-answers.yaml --trust
```

Review the resulting diff and resolve conflicts before running the checks above. The Python Templates Copier Update GitHub App can propose updates automatically; installing that app is separate from this repository change.

bt intentionally retains its MIT license, runtime Python floor, package metadata and version, Cython build hook, native wheel matrix, top-level `tests` directory, benchmarks, and Ruff line length. Documentation adopts the template's Markdown homepage while retaining Klink styling and redirecting legacy page URLs. The docs workflow builds from source because bt's CI produces platform-specific wheel artifacts.

The Cython template supplies shared Python/compiler setup, native-wheel CI, and distribution smoke tests. CI builds wheels from clean sdists instead of deleting compiled files from the checkout. `make test-dist` checks installed native extensions outside the source tree and rebuilds sdists into wheels for the same check. bt retains its full source-tree test suite; cibuildwheel verifies that each installed wheel imports the compiled `bt.core` extension.

The build workflow follows the template's Python 3.11–3.14 matrix on Linux x86_64/ARM64, macOS ARM64, and Windows x86_64. The separate release workflow retains its existing Python versions and macOS architecture selection. The runtime minimum remains unchanged.
