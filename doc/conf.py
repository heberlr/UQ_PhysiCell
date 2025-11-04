# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import version from the package
try:
    from uq_physicell.VERSION import __version__
    release = __version__
    version = __version__
except ImportError:
    # Fallback version if import fails
    raise ValueError("Could not import version from uq_physicell.VERSION")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UQ-PhysiCell'
copyright = '2025, Heber L. Rocha'
author = 'Heber L. Rocha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',      # Generate summary tables
    'sphinx.ext.todo',             # Support for TODO items
    'myst_nb',                     # Support for Jupyter Notebooks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'UQ_PhysiCell_logo.png'

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True  # Better formatting for examples
napoleon_use_admonition_for_notes = True     # Better formatting for notes
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True             # Process type annotations

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',  # Keep source order
    'special-members': '__init__',
    'exclude-members': '__weakref__'
}

# Mock imports for missing dependencies
autodoc_mock_imports = [
    'botorch',
    'gpytorch', 
    'torch',
    'matplotlib',
    'scipy',
    'seaborn'
]

# Autosummary settings
autosummary_generate = True

# TODO extension settings
todo_include_todos = True

# MyST settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Notebook execution mode for MyST-NB (replaces deprecated jupyter_execute_notebooks)
# Options: 'off' (never execute), 'auto' (execute when outputs are missing), 'force' (always execute)
# See: https://myst-nb.readthedocs.io/en/latest/config.html#nb-execution
nb_execution_mode = "off"   # 'off' | 'auto' | 'force'

# Backwards-compat note: older versions of myst-nb used the jupyter_execute_notebooks
# config name which is now deprecated; use nb_execution_mode instead.

# Add custom CSS to make DataFrame outputs responsive (see _static/custom.css)
html_css_files = [
    'custom.css',
]
