# Installation

This page covers how to install `laser-measles` for use or development.

## Prerequisites

- Python 3.10 or later

## Install from PyPI

Install the latest stable release:

```bash
pip install laser-measles
```

## Install the development version

To install the latest unreleased code directly from the `main` branch:

```bash
pip install git+https://github.com/InstituteforDiseaseModeling/laser-measles.git@main
```

!!! warning
    Development versions may contain breaking changes or unstable features. Use the stable release for production work.

## Optional dependencies

`laser-measles` provides optional dependency groups for specific use cases:

```bash
# Development tools (testing, linting, type checking)
pip install laser-measles[dev]

# Documentation build tools (MkDocs and plugins)
pip install laser-measles[docs]

# Examples and notebook support
pip install laser-measles[examples]

# All optional dependencies
pip install laser-measles[full]
```

### What each group includes

**`dev`** — Tools for testing, linting, and code quality:

| Package | Purpose |
|---------|---------|
| `ruff` | Linting and formatting |
| `pytest` | Testing framework |
| `pytest-order` | Ordered test execution |
| `pyright` | Static type checking |
| `mypy` | Static type checking |
| `bump-my-version` | Version management |

**`docs`** — Tools for building the documentation site:

| Package | Purpose |
|---------|---------|
| `mkdocs-material` | MkDocs theme |
| `mkdocstrings-python` | API reference generation from docstrings |
| `mkdocs-jupyter` | Jupyter notebook rendering |
| `mkdocs-gen-files`, `mkdocs-literate-nav` | Auto-generated API navigation |
| `mkdocs-include-markdown-plugin` | Include external Markdown files |
| `mkdocs-autorefs`, `mkdocs-api-autonav` | API cross-referencing and navigation |
| `mkdocs-table-reader-plugin`, `mkdocs-exclude` | Table reading and file exclusion |

**`examples`** — Tools for running examples and tutorials:

| Package | Purpose |
|---------|---------|
| `jupytext` | Jupyter notebook text conversion |
| `notebook` | Jupyter notebook interface |
| `seaborn` | Statistical data visualization |
| `ipykernel` | Jupyter kernel support |
| `optuna` | Hyperparameter optimization |
| `plotly` | Interactive plotting |

**`full`** — All optional dependencies combined (includes all packages from `dev`, `docs`, and `examples`).

## Setting up for development

### Using GitHub Codespaces

For a pre-configured cloud development environment, open the repository in GitHub Codespaces:

<a href='https://codespaces.new/InstituteforDiseaseModeling/laser-measles'><img src='https://github.com/codespaces/badge.svg' alt='Open in GitHub Codespaces' style='max-width: 100%;'></a>

### Local development setup

1. Clone the repository and install with development dependencies:

    ```bash
    git clone https://github.com/InstituteforDiseaseModeling/laser-measles.git
    cd laser-measles
    pip install -e ".[dev]"
    ```

2. Run the full test suite:

    ```bash
    tox
    ```

3. Build the documentation:

    ```bash
    tox -e docs
    ```

To combine coverage data across all tox environments:

| Platform | Command |
|----------|---------|
| Windows | `set PYTEST_ADDOPTS=--cov-append` then `tox` |
| Other | `PYTEST_ADDOPTS=--cov-append tox` |

To verify version bumping before releasing:

```bash
uvx bump-my-version bump minor --dry-run -vv
```

## Next steps

After installing, see the [Quick Start tutorial](tutorials/tut_quickstart_hello_world.ipynb) to run your first model.
