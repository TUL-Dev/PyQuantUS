# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'QuantUS'
copyright = '2024, David Spector'
author = 'David Spector'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration', 'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages', 'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', 'sphinx_rtd_theme']
# extensions.append("sphinx_wagtail_theme")

autodoc_default_options = {
    'members': True,          # Include class and function members
    'undoc-members': False,   # Exclude members without docstrings
    'show-inheritance': True, # Show class inheritance
}

templates_path = ['_templates']

# html_sidebars = {"**": [
#     "searchbox.html",
#     "globaltoc.html"
# ]}
exclude_patterns = []
# html_baseurl = 'https://tul-dev.github.io/PyQuantUS/'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = False
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "TUL-Dev", # Username
    "github_repo": "PyQuantUS", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/source/", # Path in the checkout to the docs root
}
# html_theme_options = dict( # Wagtail
#     project_name = "QuantUS",
#     logo = "../_images/transducer.png",
#     github_url = "https://github.com/TUL-Dev/PyQuantUS/blob/main/docs/source/",
#     logo_alt = "QuantUS",
#     logo_height = 59,
#     logo_url = "/PyQuantUS/",
#     logo_width = 45,
#     # header_links = "Top 1|http://example.com/one, Top 2|http://example.com/two",
#     footer_links = ",".join([
#         # "About Us|http://example.com/",
#         # "Contact|http://example.com/contact",
#         # "Legal|http://example.com/dev/null",
#     ]),
# )

html_logo = "logo.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_last_updated_fmt = "%b %d, %Y"
html_show_sphinx = False
