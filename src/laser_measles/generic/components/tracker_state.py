import numpy as np

from laser_measles.base import BaseComponent


class StatesTracker(BaseComponent):
    """
    Tracks the total number of agents in each state at each time tick.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        self.state_tracker = np.zeros((len(model.params.states), model.params.nticks))

    def __call__(self, model, tick: int) -> None:
        counts = np.bincount(model.population.state, minlength=len(model.params.states))
        self.state_tracker[:, tick] = counts
