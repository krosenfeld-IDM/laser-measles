"""
Basic classes
"""

from matplotlib.figure import Figure


class BaseComponent:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose
        self.initialized = True

    def plot(self, fig: Figure = None):
        """
        Placeholder for plotting method.
        """
        yield None
