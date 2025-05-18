#!/usr/bin/env python3

"""Convert all example notebooks (.Rmd) to simple scripts (.py)."""

import os

# Get the directory name safely
base_dir = (os.path.dirname(os.path.abspath(__file__))
    if '__file__' in locals() else os.getcwd())

for path, folders, files in os.walk(base_dir):
    # Skip hidden folders
    folders[:] = [name for name in folders if not name.startswith('.')]

    for name in files:
        if name.startswith('.'):
            continue
        if not name.endswith('.Rmd'):
            continue

        rmd = os.path.join(path, name)
        py = rmd[:-3] + 'py'

        with open(rmd) as lines, open(py, 'w') as text:
            text.write('#!/usr/bin/env python3\n')

            first_magic_comment = True
            in_yaml = False

            for line in lines:
                # Skip YAML header
                if line.startswith('---'):
                    in_yaml = not in_yaml
                    continue
                if in_yaml:
                    continue

                # Skip code fence
                if line.startswith('```'):
                    continue

                # Skip lines marked as "Jupyter only"
                if 'Jupyter only' in line:
                    continue

                # Remove IPython magics
                if '%matplotlib widget' in line or '%matplotlib inline' in line:
                    continue

                # Replace first Jupyter magic with Qt5Agg backend
                if line.replace(' ', '').startswith('#%'):
                    if first_magic_comment:
                        text.write('import matplotlib as mpl\n')
                        text.write("mpl.use('Qt5Agg')\n")
                        first_magic_comment = False
                    continue

                # Otherwise write the line
                text.write(line)