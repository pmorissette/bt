import os, shutil, string, glob, io
from sphinx.util.compat import Directive
from docutils.parsers.rst import directives
from IPython.nbconvert import html, python
from IPython.nbformat import current
from runipy.notebook_runner import NotebookRunner
from jinja2 import FileSystemLoader
from notebook_sphinxext import \
    notebook_node, nb_to_html, nb_to_python, \
    visit_notebook_node, depart_notebook_node, \
    evaluate_notebook

class NotebookCellDirective(Directive):
    """Insert an evaluated notebook cell into a document

    This uses runipy and nbconvert to transform an inline python
    script into html suitable for embedding in a Sphinx document.
    """
    required_arguments = 0
    optional_arguments = 1
    has_content = True
    option_spec = {'skip_exceptions' : directives.flag}

    def run(self):
        # check if raw html is supported
        if not self.state.document.settings.raw_enabled:
            raise self.warning('"%s" directive disabled.' % self.name)

        # Construct notebook from cell content
        content = "\n".join(self.content)
        with open("temp.py", "w") as f:
            f.write(content)

        convert_to_ipynb('temp.py', 'temp.ipynb')

        skip_exceptions = 'skip_exceptions' in self.options

        evaluated_text = evaluate_notebook('temp.ipynb', skip_exceptions=skip_exceptions)

        # create notebook node
        attributes = {'format': 'html', 'source': 'nb_path'}
        nb_node = notebook_node('', evaluated_text, **attributes)
        (nb_node.source, nb_node.line) = \
            self.state_machine.get_source_and_line(self.lineno)

        # clean up
        files = glob.glob("*.png") + ['temp.py', 'temp.ipynb']
        for file in files:
            os.remove(file)

        return [nb_node]

def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    app.add_node(notebook_node,
                 html=(visit_notebook_node, depart_notebook_node))

    app.add_directive('notebook-cell', NotebookCellDirective)

def convert_to_ipynb(py_file, ipynb_file):
    with io.open(py_file, 'r', encoding='utf-8') as f:
        notebook = current.reads(f.read(), format='py')
    with io.open(ipynb_file, 'w', encoding='utf-8') as f:
        current.write(notebook, f, format='ipynb')
