# Configuration file for the Sphinx documentation builder.
from datetime import datetime
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path
import tomllib

# -- Project information
def _load_project_info() -> tuple[str, str, str]:
    """Load (name, author, version) from installed metadata or pyproject.toml."""
    try:
        info = metadata("splisosm")
        return info["Name"], info["Author"], info["Version"]
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject = tomllib.load(f)
        project_cfg = pyproject.get("project", {})
        name = project_cfg.get("name", "splisosm")
        version = project_cfg.get("version", "0.0.0")
        authors = project_cfg.get("authors", [])
        author = (
            authors[0].get("name", "")
            if authors and isinstance(authors[0], dict)
            else ""
        )
        return name, author, version


project_name, author, version = _load_project_info()
project = project_name
copyright = f"{datetime.now():%Y}, {author}"
release = version

# -- General configuration

extensions = [
    'myst_nb',
    'myst_sphinx_gallery',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinx.ext.mathjax',
]

# MyST configuration
nb_execution_mode = "off"

# Citation bibtex file
bibtex_bibfiles = ["refs.bib"]

# AutoAPI configuration
autoapi_dirs = ['../../splisosm']
autoapi_add_toctree_entry = False
autoapi_python_class_content = 'both'
autoapi_ignore = ['**/.ipynb_checkpoints/*', '**/*-checkpoint.py']
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance', 
    'show-module-summary', 
]
autoapi_member_order = 'groupwise'

# Napoleon configuration
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# Autodoc options
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'show-inheritance': True,
}
# Treat single-backtick interpreted text as literal to avoid accidental
# ambiguous cross-references from docstring tokens like `n` or `n_factors`.
default_role = 'literal'
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# Autosectionlabel configuration
autosectionlabel_prefix_document = True

# Silence duplicate-label warnings that are expected in the changelog —
# every version block repeats `New features`, `Bug fixes`, `Testing`,
# and autosectionlabel emits a warning for each.  Real broken xrefs
# still surface as `ref.ref` warnings.
suppress_warnings = ["autosectionlabel.changelog", "autosectionlabel.CHANGELOG"]

# Intersphinx mapping (for cross-references)
intersphinx_mapping = {
    # 'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
    'spatialdata': ('https://spatialdata.scverse.org/en/stable/', None),
}
intersphinx_disabled_domains = ['std']

# LaTeX options for math rendering (myst-nb uses MathJax 3)
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
mathjax3_config = {
    'tex': {
        'inlinemath': [['\\(', '\\)']],
        'displaymath': [['\\[', '\\]']],
    }
}

# -- Options for HTML output
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'logo': {
        'text': 'SPLISOSM',
    },
    'search_bar_text': 'Search...',
    'show_toc_level': 4,
    'navigation_depth': 4,
    # Optional: Adds a GitHub link to the top right
    'repository_url': 'https://github.com/JiayuSuPKU/splisosm',
    'use_repository_button': True,
}

# Custom CSS files
html_static_path = ['_static']

# -- Options for EPUB output
epub_show_urls = 'footnote'