# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QuantUS'
copyright = '2024, David Spector'
author = 'David Spector'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration', 'sphinx.ext.intersphinx']
extensions.append("sphinx_wagtail_theme")

templates_path = ['_templates']

html_sidebars = {"**": [
    "searchbox.html",
    "globaltoc.html"
]}
exclude_patterns = []
html_baseurl = 'https://tul-dev.github.io/QuantUS/'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_wagtail_theme'
html_theme_options = dict( # Wagtail
    project_name = "QuantUS",
    logo = "../_images/transducer.png",
    github_url = "https://github.com/TUL-Dev/QuantUS/blob/main/docs/source/",
    logo_alt = "QuantUS",
    logo_height = 59,
    logo_url = "/QuantUS/",
    logo_width = 45,
    # header_links = "Top 1|http://example.com/one, Top 2|http://example.com/two",
    footer_links = ",".join([
        # "About Us|http://example.com/",
        # "Contact|http://example.com/contact",
        # "Legal|http://example.com/dev/null",
    ]),
)

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_last_updated_fmt = "%b %d, %Y"
html_show_sphinx = False
