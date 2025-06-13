import numpy as np

from laser_measles.biweekly.base import BaseComponent


def cast_type(a, dtype):
    return a.astype(dtype) if a.dtype != dtype else a


class VitalDynamicsProcess(BaseComponent):
    """
    Component for simulating the vital dynamics in the model.

    This class handles the simulation of births and deaths in the population model.
    It processes:
    - Births: Both vaccinated and unvaccinated births based on crude birth rate
    - Deaths: Based on crude death rate
    - Population updates: Adds births to appropriate compartments and removes deaths

    Parameters
    ----------
    model : object
        The simulation model containing nodes, states, and parameters
    verbose : bool, default=False
        Whether to print verbose output during simulation

    Notes
    -----
    - Birth rates are calculated per biweek (26 periods per year)
    - Births are split between vaccinated (MCV1) and unvaccinated compartments
    - Deaths are removed from all compartments proportionally
    - All state counts are ensured to be non-negative
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)

    def __call__(self, model, tick: int) -> None:
        # state counts
        states = model.nodes.states

        # model parameters
        params = model.params

        # Vital dynamics
        population = states.sum(axis=0)
        biweek_avg_births = population * (params.crude_birth_rate / 26.0 / 1000.0)
        vaccinated_births = cast_type(np.random.poisson(biweek_avg_births*np.array(model.scenario['mcv1'])), states.dtype)
        unvaccinated_births = cast_type(np.random.poisson(biweek_avg_births*(1-np.array(model.scenario['mcv1']))), states.dtype)
        # births = cast_type(np.random.poisson(biweek_avg_births)*np.array(model.scenario['mcv1']), states.dtype)

        biweek_avg_deaths = states * (params.crude_death_rate / 26.0 / 1000.0)
        deaths = cast_type(np.random.poisson(biweek_avg_deaths), states.dtype)  # number of deaths

        states[0] += unvaccinated_births  # add births to S
        states[2] += vaccinated_births  # add births to R
        states -= deaths  # remove deaths from each compartment

        # make sure that all states >= 0
        np.maximum(states, 0, out=states)
