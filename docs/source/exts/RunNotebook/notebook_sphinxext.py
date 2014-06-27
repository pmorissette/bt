import os, shutil, string, glob, re
from sphinx.util.compat import Directive
from docutils import nodes
from docutils.parsers.rst import directives
from IPython.nbconvert import html, python
from IPython.nbformat.current import read, write
from runipy.notebook_runner import NotebookRunner, NotebookError

class NotebookDirective(Directive):
    """Insert an evaluated notebook into a document

    This uses runipy and nbconvert to transform a path to an unevaluated notebook
    into html suitable for embedding in a Sphinx document.
    """
    required_arguments = 1
    optional_arguments = 1
    option_spec = {'skip_exceptions' : directives.flag}
    final_argument_whitespace = True

    def run(self): # check if there are spaces in the notebook name
        nb_path = self.arguments[0]
        if ' ' in nb_path: raise ValueError(
            "Due to issues with docutils stripping spaces from links, white "
            "space is not allowed in notebook filenames '{0}'".format(nb_path))
        # check if raw html is supported
        if not self.state.document.settings.raw_enabled:
            raise self.warning('"%s" directive disabled.' % self.name)

        # get path to notebook
        source_dir = os.path.dirname(
            os.path.abspath(self.state.document.current_source))
        nb_filename = self.arguments[0]
        nb_basename = os.path.basename(nb_filename)
        rst_file = self.state_machine.document.attributes['source']
        rst_dir = os.path.abspath(os.path.dirname(rst_file))
        nb_abs_path = os.path.abspath(os.path.join(rst_dir, nb_filename))

        # Move files around.
        rel_dir = os.path.relpath(rst_dir, setup.confdir)
        rel_path = os.path.join(rel_dir, nb_basename)
        dest_dir = os.path.join(setup.app.builder.outdir, rel_dir)
        dest_path = os.path.join(dest_dir, nb_basename)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Copy unevaluated script
        try:
            shutil.copyfile(nb_abs_path, dest_path)
        except IOError:
            raise RuntimeError("Unable to copy notebook to build destination.")

        dest_path_eval = string.replace(dest_path, '.ipynb', '_evaluated.ipynb')
        dest_path_script = string.replace(dest_path, '.ipynb', '.py')
        rel_path_eval = string.replace(nb_basename, '.ipynb', '_evaluated.ipynb')
        rel_path_script = string.replace(nb_basename, '.ipynb', '.py')

        # Create python script vesion
        unevaluated_text = nb_to_html(nb_abs_path)
        script_text = nb_to_python(nb_abs_path)
        f = open(dest_path_script, 'w')
        f.write(script_text.encode('utf8'))
        f.close()

        skip_exceptions = 'skip_exceptions' in self.options

        evaluated_text = evaluate_notebook(nb_abs_path, dest_path_eval,
                                           skip_exceptions=skip_exceptions)

        # Create link to notebook and script files
        link_rst = "(" + \
                   formatted_link(nb_basename) + "; " + \
                   formatted_link(rel_path_eval) + "; " + \
                   formatted_link(rel_path_script) + \
                   ")"

        self.state_machine.insert_input([link_rst], rst_file)

        # create notebook node
        attributes = {'format': 'html', 'source': 'nb_path'}
        nb_node = notebook_node('', evaluated_text, **attributes)
        (nb_node.source, nb_node.line) = \
            self.state_machine.get_source_and_line(self.lineno)

        # add dependency
        self.state.document.settings.record_dependencies.add(nb_abs_path)

        # clean up png files left behind by notebooks.
        png_files = glob.glob("*.png")
        fits_files = glob.glob("*.fits")
        h5_files = glob.glob("*.h5")
        for file in png_files:
            os.remove(file)

        return [nb_node]


class notebook_node(nodes.raw):
    pass

def nb_to_python(nb_path):
    """convert notebook to python script"""
    exporter = python.PythonExporter()
    output, resources = exporter.from_filename(nb_path)
    return output

def nb_to_html(nb_path):
    """convert notebook to html"""
    exporter = html.HTMLExporter(template_file='full')
    output, resources = exporter.from_filename(nb_path)
    header = output.split('<head>', 1)[1].split('</head>',1)[0]
    body = output.split('<body>', 1)[1].split('</body>',1)[0]

    # http://imgur.com/eR9bMRH
    header = header.replace('<style', '<style scoped="scoped"')
    header = header.replace('body {\n  overflow: visible;\n  padding: 8px;\n}\n', '')
    header = header.replace("code,pre{", "code{")

    # Filter out styles that conflict with the sphinx theme.
    filter_strings = [
        'navbar',
        'body{',
        'alert{',
        'uneditable-input{',
        'collapse{',
    ]
    filter_strings.extend(['h%s{' % (i+1) for i in range(6)])

    line_begin_strings = [
        'pre{',
        'p{margin'
        ]

    header_lines = filter(
        lambda x: not any([s in x for s in filter_strings]), header.split('\n'))
    header_lines = filter(
        lambda x: not any([x.startswith(s) for s in line_begin_strings]), header_lines)

    header = '\n'.join(header_lines)

    # concatenate raw html lines
    lines = ['<div class="ipynotebook">']
    lines.append(header)
    lines.append(body)
    lines.append('</div>')
    return '\n'.join(lines)

def evaluate_notebook(nb_path, dest_path=None, skip_exceptions=False):
    # Create evaluated version and save it to the dest path.
    # Always use --pylab so figures appear inline
    # perhaps this is questionable?
    notebook = read(open(nb_path), 'json')
    nb_runner = NotebookRunner(notebook, pylab=False)
    try:
        nb_runner.run_notebook(skip_exceptions=skip_exceptions)
    except NotebookError as e:
        print ''
        print e
        # Return the traceback, filtering out ANSI color codes.
        # http://stackoverflow.com/questions/13506033/filtering-out-ansi-escape-sequences
        return 'Notebook conversion failed with the following traceback: \n%s' % \
            re.sub(r'\\033[\[\]]([0-9]{1,2}([;@][0-9]{0,2})*)*[mKP]?', '', str(e))
    if dest_path is None:
        dest_path = 'temp_evaluated.ipynb'
    write(nb_runner.nb, open(dest_path, 'w'), 'json')
    ret = nb_to_html(dest_path)
    if dest_path is 'temp_evaluated.ipynb':
        os.remove(dest_path)
    return ret

def formatted_link(path):
    return "`%s <%s>`__" % (os.path.basename(path), path)

def visit_notebook_node(self, node):
    self.visit_raw(node)

def depart_notebook_node(self, node):
    self.depart_raw(node)

def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    app.add_node(notebook_node,
                 html=(visit_notebook_node, depart_notebook_node))

    app.add_directive('notebook', NotebookDirective)
