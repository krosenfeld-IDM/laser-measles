import numpy as np


from laser_measles.biweekly.base import BaseComponent


class ImportationPressureProcess(BaseComponent):
    """
    Component for simulating the importation pressure in the model.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        self.crude_importation_rate = 0.001 # N per year per 1k

    def __call__(self, model, tick: int) -> None:
        # state counts
        states = model.nodes.states

        # model parameters
        params = model.params

        # importation pressure
        importation_pressure = states * (self.params.crude_importation_rate / 26.0 / 1000.0)

        # ensure importation pressure is not greater than the susceptible population
        np.minimum(importation_pressure, states[0], out=importation_pressure)

        # update states
        states[0] -= importation_pressure
        states[1] += importation_pressure