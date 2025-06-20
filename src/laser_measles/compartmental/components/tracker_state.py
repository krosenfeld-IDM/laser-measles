import inspect

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from laser_measles.base import BasePhase
from laser_measles.base import BaseLaserModel


class StateTracker(BasePhase):
    """
    Component for tracking the number in each SEIR state for each time tick.

    This class maintains a time series of state counts across all nodes in the model.
    The states are dynamically generated as properties based on model.params.states
    (e.g., "S", "E", "I", "R"). Each state can be accessed as a property that returns
    a numpy array of shape (num_ticks,) containing the time series for that state.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        self.state_tracker = np.zeros(
            (len(model.params.states), model.params.num_ticks), dtype=model.patches.states.dtype
        )  # (num_states e.g., SEIR, num_ticks)

        # Dynamically create properties for each state
        for i, state in enumerate(model.params.states):
            setattr(self.__class__, state, property(lambda self, idx=i: self.state_tracker[idx]))

    def __call__(self, model, tick: int) -> None:
        # model.patches.states is (num_states, num_patches), we sum over the patches
        self.state_tracker[:, tick] = model.patches.states.sum(axis=1)

    def initialize(self, model: BaseLaserModel) -> None:
        pass

    def plot(self, fig: Figure = None):
        """
        Plots the time series of SEIR state counts across all nodes using subplots.

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
        fig = plt.figure(figsize=(12, 3 * n_states), dpi=128) if fig is None else fig
        fig.suptitle("SEIR State Counts Over Time")

        time = np.arange(self.model.params.num_ticks)
        colors = ['blue', 'orange', 'red', 'green']  # S, E, I, R
        
        for i, state in enumerate(self.model.params.states):
            ax = plt.subplot(n_states, 1, i + 1)
            color = colors[i] if i < len(colors) else 'black'
            ax.plot(time, self.state_tracker[i], label=f'{state} (Total)', color=color, linewidth=2)
            ax.set_ylabel(f"Number in {state}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format y-axis with scientific notation for large numbers
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # Only add xlabel to the bottom subplot
            if i == n_states - 1:
                ax.set_xlabel("Time (days)")

        plt.tight_layout()

        # Check if the function is being used as a generator
        frame = inspect.currentframe()
        try:
            yield
        finally:
            if frame:
                del frame
                
    def plot_combined(self, fig: Figure = None):
        """
        Plots all SEIR states on a single plot for easy comparison.
        
        Parameters:
            fig (Figure, optional): A matplotlib Figure object. If None, a new figure will be created.
            
        Yields:
            None: This function uses a generator to yield control back to the caller.
        """
        fig = plt.figure(figsize=(12, 6), dpi=128) if fig is None else fig
        
        time = np.arange(self.model.params.num_ticks)
        colors = ['blue', 'orange', 'red', 'green']  # S, E, I, R
        linestyles = ['-', '--', '-.', ':']
        
        for i, state in enumerate(self.model.params.states):
            color = colors[i] if i < len(colors) else 'black'
            linestyle = linestyles[i] if i < len(linestyles) else '-'
            plt.plot(time, self.state_tracker[i], 
                    label=f'{state}', color=color, linestyle=linestyle, linewidth=2)
        
        plt.xlabel("Time (days)")
        plt.ylabel("Number of Individuals")
        plt.title("SEIR Model Dynamics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        plt.tight_layout()
        
        # Check if the function is being used as a generator
        frame = inspect.currentframe()
        try:
            yield
        finally:
            if frame:
                del frame