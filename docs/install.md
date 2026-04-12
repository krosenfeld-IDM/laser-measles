# Installation

This guide shows you how to install laser-measles and set up your environment for use or development.

## Prerequisites

- Python 3.10 or later

## Install laser-measles

Install the latest stable release from PyPI:

```bash
pip install laser-measles
```

**Recommended stable release:** For new projects, pin to the 0.10.x release line, which is validated and supported for most use cases:

```bash
pip install "laser-measles>=0.10,<1.0"
```

To install the latest in-development version directly from GitHub:

```bash
pip install git+https://github.com/InstituteforDiseaseModeling/laser-measles.git@main
```

## Install optional dependencies

The package includes optional dependency groups for specific workflows. Install them using bracket notation:

```bash
# Development dependencies (testing, linting)
pip install "laser-measles[dev]"

# Documentation dependencies (MkDocs, mkdocstrings)
pip install "laser-measles[docs]"

# Example and tutorial dependencies (Jupyter, notebooks, plotting)
pip install "laser-measles[examples]"

# All optional dependencies
pip install "laser-measles[full]"
```

### What each group includes

**dev** — Testing and code quality tools

- `pytest`: Testing framework
- `pytest-order`: Ordered test execution

**docs** — Documentation building tools

- `mkdocs-material`: MkDocs theme
- `mkdocstrings-python`: API reference generation from docstrings
- `mkdocs-jupyter`: Jupyter notebook rendering
- `mkdocs-gen-files`, `mkdocs-literate-nav`: Auto-generated API navigation

**examples** — Tools for running examples and tutorials

- `jupytext`: Jupyter notebook text conversion
- `notebook`: Jupyter notebook interface
- `seaborn`: Statistical data visualization
- `ipykernel`: Jupyter kernel support

**full** — All of the above groups combined

## Set up a development environment

If you are contributing to laser-measles, use the GitHub Codespace for a pre-configured environment:

<a href='https://codespaces.new/InstituteforDiseaseModeling/laser-measles'><img src='https://github.com/codespaces/badge.svg' alt='Open in GitHub Codespaces' style='max-width: 100%;'></a>

### Run the tests

```bash
tox
```

To combine coverage data across all tox environments (useful when running tox with multiple Python versions):

| Platform | Command |
|----------|---------|
| Windows  | `set PYTEST_ADDOPTS=--cov-append` then `tox` |
| Other    | `PYTEST_ADDOPTS=--cov-append tox` |

### Build the documentation

```bash
tox -e docs
```

### Check version bumping

To verify the version bump configuration without making any changes, run:

```bash
uvx bump-my-version bump minor --dry-run -vv
```
