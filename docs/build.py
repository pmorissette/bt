"""Build the documentation with released Yardang and Klink packages."""

from pathlib import Path

from sphinx.application import Sphinx
from yardang.build import generate_docs_configuration


def main():
    with generate_docs_configuration() as config_dir:
        app = Sphinx(
            srcdir=".",
            confdir=config_dir,
            outdir="docs/html",
            doctreedir="docs/html/.doctrees",
            buildername="html",
            confoverrides={
                "html_static_path": [str(Path("docs/source/_static").resolve())],
                "html_title": "bt — Flexible Backtesting for Python",
                "intersphinx_mapping": {"ffn": ("https://pmorissette.github.io/ffn/", None)},
            },
            warningiserror=True,
        )
        app.build()
        return app.statuscode


if __name__ == "__main__":
    raise SystemExit(main())
