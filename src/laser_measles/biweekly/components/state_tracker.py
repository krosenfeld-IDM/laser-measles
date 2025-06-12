import inspect

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from laser_measles.biweekly.base import BaseComponent


class StateTracker(BaseComponent):
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
            (len(model.params.states), model.params.nticks), dtype=model.nodes.states.dtype
        )  # (num_states e.g., SIR, nticks)

        # Dynamically create properties for each state
        for i, state in enumerate(model.params.states):
            setattr(self.__class__, state, property(lambda self, idx=i: self.state_tracker[idx]))

    def __call__(self, model, tick: int) -> None:
        # model.nodes.states is (num_states, num_nodes), we sum over the nodes
        self.state_tracker[:, tick] = model.nodes.states.sum(axis=1)

    def plot(self, fig: Figure = None):
        """
        Plots the time series of state counts across all nodes.

        This function creates a line plot showing how the number of individuals in each state
        changes over time. Each state is represented by a different colored line.

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
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("State Counts Over Time")

        time = np.arange(self.model.params.nticks)
        for i, state in enumerate(self.model.params.states):
            plt.plot(time, self.state_tracker[i], label=state)

        plt.xlabel("Time")
        plt.ylabel("Number of Individuals")
        plt.legend()
        plt.grid(True)

        # Check if the function is being used as a generator
        frame = inspect.currentframe()
        try:
            yield
        finally:
            if frame:
                del frame
