import os

first_magic_comment = True  # Flag to check the first magic comment

# Get the directory name safely
base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() \
else os.getcwd()

for path, folders, files in os.walk(base_dir):
    folders[:] = [name for name in folders if not name.startswith('.')]

    for name in files:
        if name.startswith('.'):
            pass
        elif name.endswith('.Rmd'):
            rmd = os.path.join(path, name)
            py = rmd[:-3] + 'py'

            with open(rmd) as lines, open(py, 'w') as text:
                text.write('#!/usr/bin/env python3\n')

                for line in lines:
                    if line.startswith('---'):
                        for line in lines:
                            if line.startswith('---'):
                                break
                    elif line.startswith('```'):
                        pass
                    elif 'Jupyter only' in line:
                        pass
                    elif line.replace(' ', '').startswith('#%'):
                        if first_magic_comment:

                            text.write("import matplotlib as mpl \
                                       \nmpl.use('Qt5Agg')\n")
                            first_magic_comment = False 
                    else:
                        text.write(line)