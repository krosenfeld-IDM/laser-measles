"""
Laser Measles - Agent-based measles simulation framework.

This package provides tools for simulating measles transmission dynamics
using agent-based models with various spatial and temporal configurations.
"""

__version__ = "0.6.3"

# --- Exports ---
from difflib import get_close_matches

from .api import *  # noqa: F403
from .api import __all__


def __getattr__(name: str) -> None:
    # Get all available names in the module
    available_names = __all__

    # Find close matches
    suggestions = get_close_matches(name, available_names, n=3, cutoff=0.6)

    if suggestions:
        suggestion_text = f"Did you mean: {', '.join(suggestions)}?"
    else:
        suggestion_text = f"Available attributes: {', '.join(available_names)}"

    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"{suggestion_text}"
    )
