# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Diropt'
copyright = '2024, Ronak Mehta'
author = 'Ronak Mehta'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'sphinx.ext.doctest',
  'sphinx.ext.duration',
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.intersphinx',
  'sphinx.ext.inheritance_diagram',
  'sphinx.ext.napoleon',
  'myst_nb',  # This is used for the .ipynb notebooks
]


myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'show_toc_level': 2,
    'navigation_with_keys': False,
}
html_static_path = ['_static']
html_title = "Diropt"
