import inspect

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from laser_measles.base import BasePhase
from laser_measles.base import BaseLaserModel


class StateTracker(BasePhase):
    """
    Component for tracking the number in each state for each time tick.

    This class maintains a time series of state counts across all nodes in the model.
    The states are dynamically generated as properties based on model.params.states
    (e.g., "S", "I", "R"). Each state can be accessed as a property that returns
    a numpy array of shape (nticks,) containing the time series for that state.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        self.state_tracker = np.zeros(
            (len(model.params.states), model.params.num_ticks), dtype=model.patches.states.dtype
        )  # (num_states e.g., SIR, nticks)

        # Dynamically create properties for each state
        for i, state in enumerate(model.params.states):
            setattr(self.__class__, state, property(lambda self, idx=i: self.state_tracker[idx]))

    def __call__(self, model, tick: int) -> None:
        # model.nodes.states is (num_states, num_nodes), we sum over the nodes
        self.state_tracker[:, tick] = model.patches.states.sum(axis=1)

    def initialize(self, model: BaseLaserModel) -> None:
        pass

    def plot(self, fig: Figure = None):
        """
        Plots the time series of state counts across all nodes using subplots.

        This function creates a separate subplot for each state, showing how the number of individuals
        in each state changes over time. Each state gets its own subplot for better visibility.

        Parameters:
            fig (Figure, optional): A matplotlib Figure object. If None, a new figure will be created.

        Yields:
            None: This function uses a generator to yield control back to the caller.
            If used directly (not as a generator), it will show the plot immediately.

        Example:
            # Use as a generator (for model.visualize()):
            for _ in tracker.plot():
                plt.show()
        """
        n_states = len(self.model.params.states)
        fig = plt.figure(figsize=(12, 4 * n_states), dpi=128) if fig is None else fig
        fig.suptitle("State Counts Over Time")

        time = np.arange(self.model.params.num_ticks)
        for i, state in enumerate(self.model.params.states):
            ax = plt.subplot(n_states, 1, i + 1)
            ax.plot(time, self.state_tracker[i], label=state)
            ax.set_ylabel("Number of Individuals")
            ax.grid(True)
            ax.legend()
            
            # Only add xlabel to the bottom subplot
            if i == n_states - 1:
                ax.set_xlabel("Time")

        plt.tight_layout()

        # Check if the function is being used as a generator
        frame = inspect.currentframe()
        try:
            yield
        finally:
            if frame:
                del frame
