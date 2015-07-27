import os
from subprocess import call
import shutil
import re


def convert_notebooks():
    """
    Converts IPython Notebooks to proper .rst files and moves static
    content to the _static directory.
    """
    convert_status = call(['ipython', 'nbconvert', '--to', 'rst', '*.ipynb'])
    if convert_status != 0:
        raise SystemError('Conversion failed! Status was %s' % convert_status)

    notebooks = [x for x in os.listdir('.') if '.ipynb'
                 in x and os.path.isfile(x)]
    names = [os.path.splitext(x)[0] for x in notebooks]

    for i in range(len(notebooks)):
        name = names[i]
        notebook = notebooks[i]

        print('processing %s (%s)' % (name, notebook))

        # move static files
        sdir = '%s_files' % name
        statics = os.listdir(sdir)
        statics = [os.path.join(sdir, x) for x in statics]
        [shutil.copy(x, '_static/') for x in statics]
        shutil.rmtree(sdir)

        # rename static dir in rst file
        rst_file = '%s.rst' % name
        print('REsT file is %s' % rst_file)
        data = None
        with open(rst_file, 'r') as f:
            data = f.read()

        if data is not None:
            with open(rst_file, 'w') as f:
                data = re.sub('%s' % sdir, '_static', data)
                f.write(data)

        # add special tags
        lines = None
        with open(rst_file, 'r') as f:
            lines = f.readlines()

        if lines is not None:
            n = len(lines)
            i = 0
            rawWatch = False

            while i < n:
                line = lines[i]
                # add class tags to images for css formatting
                if 'image::' in line:
                    lines.insert(i + 1, '    :class: pynb\n')
                    n += 1
                elif 'parsed-literal::' in line:
                    lines.insert(i + 1, '    :class: pynb-result\n')
                    n += 1
                elif 'raw:: html' in line:
                    rawWatch = True

                if rawWatch:
                    if '<div' in line:
                        line = line.replace('<div', '<div class="pynb-result"')
                        lines[i] = line
                        rawWatch = False

                i += 1

            with open(rst_file, 'w') as f:
                f.writelines(lines)


def get_html_theme_path():
    """Returns list of HTML theme paths."""
    cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    return cur_dir


VERSION = (0, 1, 8)
__version__ = '.'.join(str(v) for v in VERSION)
__version_full__ = __version__
