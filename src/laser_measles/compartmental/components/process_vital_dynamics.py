import numpy as np
from pydantic import BaseModel, Field

from laser_measles.base import BasePhase
from laser_measles.base import BaseLaserModel
from laser_measles.utils import cast_type


class VitalDynamicsParams(BaseModel):
    """Parameters specific to vital dynamics."""

    crude_birth_rate: float = Field(
        20.0,
        description="Annual crude birth rate per 1000 population",
        gt=0.0
    )
    crude_death_rate: float = Field(
        8.0,
        description="Annual crude death rate per 1000 population",
        gt=0.0
    )


class VitalDynamicsProcess(BasePhase):
    """
    Phase for simulating the vital dynamics in the model.

    This phase handles the simulation of births and deaths in the population model.
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
    params : VitalDynamicsParams | None, default=None
        Component-specific parameters. If None, will use default parameters

    Notes
    -----
    - Birth rates are calculated per biweek (26 periods per year)
    - Births are split between vaccinated (MCV1) and unvaccinated compartments
    - Deaths are removed from all compartments proportionally
    - All state counts are ensured to be non-negative
    """

    def __init__(self, model, verbose: bool = False, params: VitalDynamicsParams | None = None) -> None:
        super().__init__(model, verbose)
        if params is None:
            params = VitalDynamicsParams()
        self.params = params

    def __call__(self, model, tick: int) -> None:
        # state counts
        states = model.patches.states # num_compartments x num_patches

        # Vital dynamics
        population = states.sum(axis=0)
        avg_births = population * (self.params.crude_birth_rate / 365.0 / 1000.0)
        vaccinated_births = cast_type(np.random.poisson(avg_births*np.array(model.scenario['mcv1'])), states.dtype)
        unvaccinated_births = cast_type(np.random.poisson(avg_births*(1-np.array(model.scenario['mcv1']))), states.dtype)

        avg_deaths = states * (self.params.crude_death_rate / 365.0 / 1000.0)
        deaths = cast_type(np.random.poisson(avg_deaths), states.dtype)  # number of deaths

        states.S += unvaccinated_births  # add births to S
        states.R += vaccinated_births  # add births to R
        states -= deaths  # remove deaths from each compartment

        # make sure that all states >= 0
        np.maximum(states, 0, out=states)

    def initialize(self, model: BaseLaserModel) -> None:
        pass
