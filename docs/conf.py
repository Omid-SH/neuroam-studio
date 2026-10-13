"""Sphinx configuration for the public NeuroAM Studio documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import neuroam  # noqa: E402  (needs the sys.path insert above first)

project = "NeuroAM Studio"
author = "NeuroAM Studio contributors"
copyright = f"{author}"
release = neuroam.__version__
version = release
language = "en"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
# a plain ```mermaid fence is just a highlighted code block to MyST by
# default; this is what actually hands it to sphinxcontrib-mermaid's
# `mermaid` directive instead, so ARCHITECTURE.md's diagram renders as a
# real diagram rather than a wall of arrow syntax.
myst_fence_as_directive = ["mermaid"]

# auto-generates a #slug anchor for every heading (GitHub's own slug rules),
# up to this depth -- without it, this project's own cross-doc links (e.g.
# ARCHITECTURE.md linking to DESIGN.md#7-anatomical-registration) don't
# resolve, because nothing defines that anchor otherwise.
myst_heading_anchors = 4

master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Retained in the repository for provenance, but not published as part of
    # the maintained user documentation.
    "PLAN.md",
    "PAPER_PLAN.md",
    "NEUROINFORMATICS_PUBLISHING_REVIEW.md",
    "AM_VERIFICATION.md",
    "STIMULATION_PARAMETER_STUDY.md",
    "CLINICAL_ELECTRODES.md",
]

# -- autodoc -----------------------------------------------------------------
autodoc_mock_imports = ["neuron", "pyvista", "plotly", "skimage", "pyamg"]
"""Modules the ``neuroam`` package imports lazily/optionally at call time,
never at module import time -- mocked here purely so autodoc can *inspect*
a function signature that mentions one in a type hint without needing it
actually installed in the docs build environment."""

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_title = f"{project} {release}"
html_static_path = ["_static"]

# `on_rtd` mirrors Read the Docs' own convention for detecting its build
# environment, in case any future extension needs to branch on it.
on_rtd = os.environ.get("READTHEDOCS") == "True"
