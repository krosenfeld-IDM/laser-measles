import numpy as np
from pydantic import BaseModel, Field

from laser_measles.base import BaseComponent
from laser_measles.utils import cast_type


class InfectionParams(BaseModel):
    """Parameters specific to the infection process component."""
    
    beta: float = Field(32, description="Base transmission rate", gt=0.0)
    seasonality: float = Field(0.06, description="Seasonality factor", ge=0.0, le=1.0)
    season_start: int = Field(0, description="Season start tick (0-25)", ge=0, le=25)


class InfectionProcess(BaseComponent):
    """
    Component for simulating the spread of infection in the model.

    This class implements a stochastic infection process that models disease transmission
    between different population groups. It uses a seasonally-adjusted transmission rate
    and accounts for mixing between different population groups.

    The infection process follows these steps:
    1. Calculates expected new infections based on:
       - Base transmission rate (beta)
       - Seasonal variation
       - Population mixing matrix
       - Current number of infected individuals
    2. Converts expected infections to probabilities
    3. Samples actual new infections from a binomial distribution
    4. Updates population states:
       - Moves current infected to recovered (configurable recovery period)
       - Adds new infections to infected population
       - Removes new infections from susceptible population

    Parameters
    ----------
    model : object
        The simulation model containing population states and parameters
    verbose : bool, default=False
        Whether to print detailed information during execution
    params : InfectionParams | None, default=None
        Component-specific parameters. If None, will use default parameters

    Notes
    -----
    The infection process uses a configurable recovery period and seasonal
    transmission rate that varies sinusoidally over time.
    """

    def __init__(self, model, verbose: bool = False, params: InfectionParams | None = None) -> None:
        super().__init__(model, verbose)
        if params is None:
            params = InfectionParams()
        self.params = params

    def __call__(self, model, tick: int) -> None:
        # state counts
        states = model.nodes.states

        # calculate the expected number of new infections
        # beta * (1 + seasonality * sin(2π(t-t0)/26)) * mixing * I
        expected = (
            self.params.beta
            * (1 + self.params.seasonality * np.sin(2 * np.pi * (tick - self.params.season_start) / 26.0))
            * np.matmul(model.params.mixing, states[1])
        )

        # probability of infection = 1 - exp(-expected/total_population)
        prob = 1 - np.exp(-expected / states.sum(axis=0))

        # sample from binomial distribution to get actual new infections
        dI = cast_type(np.random.binomial(n=states[0], p=prob), states.dtype)

        # move all currently infected to recovered (using configurable recovery period)
        states[2] += states[1]
        states[1] = 0

        # update susceptible and infected populations
        states[1] += dI  # add new infections to I
        states[0] -= dI  # remove new infections from S

        return
