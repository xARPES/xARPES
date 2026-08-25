# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

try:
    import xarpes
    version = release = xarpes.__version__
except Exception:
    version = release = "0.1.0"

project = "xARPES"
copyright = "2025-2026 xARPES Developers"

# --- HTML logo + static files ---
html_static_path = ["_static"]
html_logo = "_static/xarpes_small.svg"
html_theme = "sphinx_rtd_theme"
html_theme_options = {"logo_only": True}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
    "nbsphinx",
]

exclude_patterns = ["README.md"]

# MyST: helpful for .md pages (not required for notebooks, but fine)
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# nbsphinx settings
nbsphinx_execute = "never"
nbsphinx_use_pandoc = False

# MathJax v3 configuration (Sphinx uses v3 by default)
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "tags": "ams",
    }
}

rst_epilog = """
.. include:: <isogrk1.txt>
"""

autodoc_member_order = "bysource"
numpydoc_show_class_members = False
