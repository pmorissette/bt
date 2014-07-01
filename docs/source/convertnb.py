#!/usr/bin/env python
import os
from subprocess import call
import shutil
import re


if __name__ == '__main__':
    convert_status = call(['ipython', 'nbconvert', '--to', 'rst', '*.ipynb'])
    if convert_status != 0:
        raise SystemError('Conversion failed! Status was %s' % convert_status)

    notebooks = [x for x in os.listdir('.') if '.ipynb'
                 in x and os.path.isfile(x)]
    names = [os.path.splitext(x)[0] for x in notebooks]

    for i in range(len(notebooks)):
        name = names[i]
        notebook = notebooks[i]

        print 'processing %s (%s)' % (name, notebook)

        # move static files
        sdir = '%s_files' % name
        statics = os.listdir(sdir)
        statics = [os.path.join(sdir, x) for x in statics]
        [shutil.copy(x, '_static/') for x in statics]
        shutil.rmtree(sdir)

        # rename static dir in rst file
        rst_file = '%s.rst' % name
        print 'REsT file is %s' % rst_file
        data = None
        with open(rst_file, 'r') as f:
            data = f.read()

        if data is not None:
            with open(rst_file, 'w') as f:
                data = re.sub('%s' % sdir, '_static', data)
                f.write(data)

        # add class tags images for css formatting
        lines = None
        with open(rst_file, 'r') as f:
            lines = f.readlines()

        if lines is not None:
            for i in range(len(lines)):
                line = lines[i]
                if 'image::' in line:
                    lines.insert(i + 1, '\t:class: pynb\n')

            with open(rst_file, 'w') as f:
                f.writelines(lines)
