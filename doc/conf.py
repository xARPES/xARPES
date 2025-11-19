# Configuration file for the Sphinx documentation builder.

# https://www.sphinx-doc.org/en/master/usage/configuration.html

try:
    import xarpes
    version = release = xarpes.__version__
except Exception:
    version = release = "0.1.0"

project = 'xARPES'
copyright = '2025 xARPES Developers'

# --- HTML logo + static files ---
html_static_path = ["_static"]
html_logo = "_static/xarpes.svg"
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
}

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'numpydoc',
    'myst_parser',
]

exclude_patterns = ['README.md']

mathjax_path = ('https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js'
    '?config=TeX-AMS-MML_HTMLorMML')

rst_epilog = '''
.. include:: <isogrk1.txt>
'''

html_theme = 'sphinx_rtd_theme'

# The following setting specifies the order in which members are documented
autodoc_member_order = 'bysource'

numpydoc_show_class_members = False
