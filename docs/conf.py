# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Rau'
copyright = '2025, Brian DuSell'
author = 'Brian DuSell'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.bibtex',
    'sphinx_multiversion'
]

templates_path = ['_templates']
exclude_patterns = ['dist']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_theme_options = {
    'github_user' : 'bdusell',
    'github_repo' : 'rau',
    'github_button' : False,
    'github_banner' : True,
    'show_related' : True,
    'show_relbar_bottom' : True
}
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
        # Add a section in the sidebar for different versions.
        'versioning.html',
    ]
}

# For the ordering of autodoc-generated documentation.
autodoc_default_options = {
    'member-order' : 'bysource',
    'members' : True,
    'imported-members' : True,
    'undoc-members' : False,
    'show-inheritance' : True,
    'inherited-members' : False
}
autodoc_class_signature = 'separated'
autodoc_inherit_docstrings = False

# Configure links to external documentation.
intersphinx_mapping = {
    'python' : ('https://docs.python.org/3', None),
    'numpy' : ('https://numpy.org/doc/stable', None),
    'pytorch' : ('https://docs.pytorch.org/docs/stable', None)
}

# Bibtex.
bibtex_bibfiles = ['bibliography.bib']
bibtex_reference_style = 'author_year'

# Multiversioning.
smv_released_pattern = r'^refs/tags/v.+$'