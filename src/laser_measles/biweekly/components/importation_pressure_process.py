import numpy as np

from laser_measles.biweekly.base import BaseComponent

def cast_type(a, dtype):
    return a.astype(dtype) if a.dtype != dtype else a


class ImportationPressureProcess(BaseComponent):
    """
    Component for simulating the importation pressure in the model.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        self.crude_importation_rate = 1 # N per year per 1k
        self.importation_start = 0
        self.importation_end = 4 # bi weeks

    def __call__(self, model, tick: int) -> None:
        if tick < self.importation_start or tick > self.importation_end:
            return

        # state counts
        states = model.nodes.states

        # population
        population = states.sum(axis=0)

        # importation pressure
        importation_pressure = population * (self.crude_importation_rate / 26.0 / 1000.0)

        # ensure importation pressure is not greater than the susceptible population
        importation_pressure = cast_type(importation_pressure, states.dtype)
        np.minimum(importation_pressure, states[0], out=importation_pressure)

        # update states
        states[0] -= importation_pressure
        states[1] += importation_pressure
