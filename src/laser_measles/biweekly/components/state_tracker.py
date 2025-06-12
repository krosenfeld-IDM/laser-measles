import numpy as np

from laser_measles.biweekly.base import BaseComponent


class StateTracker(BaseComponent):
    """
    Component for tracking the number in each state for each time tick.

    This class maintains a time series of state counts across all nodes in the model.
    The states are dynamically generated as properties based on model.params.states
    (e.g., "S", "I", "R"). Each state can be accessed as a property that returns
    a numpy array of shape (nticks,) containing the time series for that state.

    Example:
        >>> tracker = StateTracker(model)
        >>> susceptible = tracker.S  # Get time series of susceptible individuals
        >>> infected = tracker.I     # Get time series of infected individuals
        >>> recovered = tracker.R    # Get time series of recovered individuals
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
