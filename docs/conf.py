# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import re
from rodeo import __version__, __author__

# -- Project information -----------------------------------------------------

project = 'rodeo'
author = __author__
copyright = '2019-, ' + author

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    "myst_nb"
]

# Remove method table from numpydoc to get rid of warnings
numpydoc_show_class_members = False

# napoleon options
# napoleon_google_docstring = False
# napoleon_use_param = True
# napoleon_use_ivar = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    'examples/notebooks',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'rodeodoc'


# -- Options for LaTeX output ------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'rodeo.tex', 'rodeo Documentation',
     'Mohan Wu, Martin Lysy', 'manual'),
]


# -- Options for myst-nb -----------------------------------------------------

nb_custom_formats = {
    '.md': ['jupytext.reads', {'fmt': 'md'}]
}

nb_execution_mode = 'cache'

nb_execution_timeout = -1

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_image',
]

myst_title_to_header = True

myst_heading_anchors = 3

# convert latexdefs.tex to mathjax format
mathjax3_config = {'tex': {'macros': {}}}
with open('examples/latexdefs.tex', 'r') as f:
    for line in f:
        # newcommand macros
        macros = re.findall(
            r'\\(newcommand){\\(.*?)}(\[(\d)\])?{(.+)}', line)
        for macro in macros:
            if len(macro[2]) == 0:
                mathjax3_config['tex']['macros'][macro[1]] = '{'+macro[4]+'}'
            else:
                mathjax3_config['tex']['macros'][macro[1]] = [
                    '{'+macro[4]+'}', int(macro[3])
                ]
        # DeclarMathOperator macros
        macros = re.findall(r'\\(DeclareMathOperator\*?){\\(.*?)}{(.+)}', line)
        for macro in macros:
            mathjax3_config['tex']['macros'][macro[1]] = \
                '{\\operatorname{'+macro[2]+'}}'
