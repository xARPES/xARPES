#!/usr/bin/env python3

"""Convert all example notebooks (.Rmd) to simple scripts (.py)."""

import os

for path, folders, files in os.walk(os.path.relpath(os.path.dirname(__file__))):
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
                    elif line.replace(' ', '').startswith('#%'): # magic comment
                        pass
                    else:
                        text.write(line)
