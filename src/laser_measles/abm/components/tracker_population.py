import numpy as np

from laser_measles.base import BaseComponent


class PopulationTracker(BaseComponent):
    """
    Tracks the total population size at each time tick.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        self.population_tracker = np.zeros((model.patches.count, model.params.nticks), dtype=model.patches.populations.dtype)

    def __call__(self, model, tick: int) -> None:
        self.population_tracker[:,tick] = model.patches.populations