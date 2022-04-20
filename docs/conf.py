# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'Metabu'
copyright = '2022, Rakotoarison, Milijaona, Rasoanaivon, Sebag, Schoenauer'
author = 'Herilalaina Rakotoarison, Louisot Milijaona, Andry Rasoanaivon, Michele Sebag, and Marc Schoenauer'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    # "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    # "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    # "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "numpydoc",
    'sphinxcontrib.bibtex',
    # "sphinx.ext.linkcode",
]

numpydoc_show_class_members = False
autosummary_generate = True

# numpydoc_show_class_members = False


# generate autosummary even if no references
# autosummary_generate = False
# autosummary_imported_members = False


# Biblio
bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_typehints = "signature"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_readable_theme
html_theme = 'readable'
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]


html_theme_options = {
    # "display_github": True,
    # 'github_user': 'luxusg1',
    # 'github_repo': 'Metabu',
    # 'github_button': False,
    # 'github_banner': True,
    #'nosidebar': True,
    #'page_width': '1000px'
    # "sticky_navigation": True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_title = "Learning Meta-features for AutoML"

# add custom css
# def setup(app):
#   app.add_css_file('custom.css')  # give a filename you created.