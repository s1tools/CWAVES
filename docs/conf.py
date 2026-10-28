# Configuration file for the Sphinx documentation.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "CWAVES"
author = "Antoine Grouazel / IFREMER"
copyright = "2026, Antoine Grouazel / IFREMER"

# The full version, including alpha/beta/rc tags.
try:
    import cwaves

    release = cwaves.__version__
except Exception:  # pragma: no cover - fallback when the package is not installed
    release = "0.0.0"

version = release.split("+")[0].split(".")[0]

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "generate_*.py"]

napoleon_google_docstring = True
autodoc_typehints = "both"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo_ifremer.jpg"
html_logo_width = "220px"
