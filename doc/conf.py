# Configuration file for the Sphinx documentation builder.

# https://www.sphinx-doc.org/en/master/usage/configuration.html

import xarpes

project = 'xARPES'
copyright = '2024 xARPES Developers'
version = xarpes.__version__
release = xarpes.__version__

html_logo = '../logo/xarpes_small.svg'
html_theme_options = {'logo_only': True}

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

exclude_patterns = ['README.md']

mathjax_path = ('https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js'
    '?config=TeX-AMS-MML_HTMLorMML')

rst_epilog = '''
.. include:: <isogrk1.txt>
'''

html_theme = 'sphinx_rtd_theme'
