from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel
from pydantic import Field

from laser_measles.base import BaseComponent
from laser_measles.utils import cast_type


class BirthsConstantPopParams(BaseModel):
    """Parameters specific to the births process component."""

    cbr: float = Field(default=20, description="Crude birth rate per 1000 people per year", gt=0.0)


class BirthsConstantPopProcess(BaseComponent):
    """
    A component to handle the birth events in a model with constant population - that is, births == deaths.

    Attributes:

        model: The model instance containing population and parameters.
        verbose (bool): Flag to enable verbose output. Default is False.
        initializers (list): List of initializers to be called on birth events.
        metrics (DataFrame): DataFrame to holding timing metrics for initializers.
    """

    def __init__(self, model, verbose: bool = False, params: BirthsConstantPopParams | None = None):
        """
        Initialize the Births component.

        Parameters:

            model (object): The model object which must have a `population` attribute.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
            params (BirthsParams, optional): Component parameters. If None, uses model.params.

        Raises:

            AssertionError: If the model does not have a `population` attribute.
            AssertionError: If the model's population does not have a `dob` attribute.
        """

        assert getattr(model, "population", None) is not None, "Births requires the model to have a `population` attribute"

        super().__init__(model, verbose)
        self.params = params if params is not None else BirthsConstantPopParams()

        model.population.add_scalar_property("dob", dtype=np.int32)
        # Simple initializer for ages where birth rate = mortality rate:
        daily_mortality_rate = (1 + self.params.cbr / 1000) ** (1 / 365) - 1
        model.population.dob[0 : model.population.count] = -1 * model.prng.exponential(
            1 / daily_mortality_rate, model.population.count
        ).astype(np.int32)

        model.patches.add_scalar_property("births", dtype=np.uint32)
        mu = (1 + self.params.cbr / 1000) ** (1 / 365) - 1
        model.patches.births = model.prng.poisson(lam=model.patches.populations * mu, size=model.patches.births.shape)
        self._initializers = []
        self._metrics = []

        return

    @property
    def initializers(self):
        """
        Returns the initializers to call on new agent births.

        This method retrieves the initializers that are used to set up the
        initial state or configuration for agents at birth.

        Returns:

            list: A list of initializers - instances of objects with an `on_birth` method.
        """

        return self._initializers

    @property
    def metrics(self):
        """
        Returns the timing metrics for the births initializers.

        This method retrieves the timing metrics for the births initializers.

        Returns:

            DataFrame: A Pandas DataFrame of timing metrics for the births initializers.
        """

        return pd.DataFrame(self._metrics, columns=["tick"] + [type(initializer).__name__ for initializer in self._initializers])

    def __call__(self, model, tick) -> None:
        """
        Adds new agents to each patch based on expected daily births calculated from CBR. Calls each of the registered initializers for the newborns.

        Args:

            model: The simulation model containing patches, population, and parameters.
            tick: The current time step in the simulation.

        Returns:

            None

        This method performs the following steps:

            1. Draw a random set of indices, or size size "number of births"  from the population,
        """

        # When we get to having birth rate per node, will need to be more clever here, but with constant birth rate across nodes,
        # random selection will be population proportional.  If node id is not contiguous, could be tricky?
        indices = model.prng.choice(model.patches.populations.sum(), size=model.patches.births.sum(), replace=False)

        # Births, set date of birth and state to 0 (susceptible)
        if hasattr(model.population, "dob"):
            model.population.dob[indices] = tick  # set to current tick
        model.population.state[indices] = 0

        # Deaths
        model.patches.populations -= cast_type(
            np.bincount(
                model.population.nodeid[indices], weights=model.population.dob[indices] < tick, minlength=len(model.patches.populations)
            ),
            model.patches.populations.dtype,
        )

        timing = [tick]
        for initializer in self._initializers:
            tstart = datetime.now(tz=None)  # noqa: DTZ005
            initializer.on_birth(model, tick, indices, None)
            tfinish = datetime.now(tz=None)  # noqa: DTZ005
            delta = tfinish - tstart
            timing.append(int(delta.total_seconds() * 1_000_000))
        self._metrics.append(timing)

        return

    def plot(self, fig: Figure = None):
        """
        Plots the births in the top 5 most populous patches and a pie chart of birth initializer times.

        Parameters:

            fig (Figure, optional): A matplotlib Figure object. If None, a new figure will be created. Defaults to None.

        Yields:

            None: This function yields twice to allow for intermediate plotting steps.
        """

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Births in Top 5 Most Populous Patches")

        indices = self.model.patches.populations[0, :].argsort()[-5:]
        ax1 = plt.gca()
        ticks = list(range(0, self.params.nticks, 365))
        for index in indices:
            ax1.plot(self.model.patches.populations[ticks, index], marker="x", markersize=4)

        ax2 = ax1.twinx()
        for index in indices:
            ax2.plot(self.model.patches.births[:, index], marker="+", markersize=4)

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Births in Top 5 Most Populous Patches")

        metrics = pd.DataFrame(self._metrics, columns=["tick"] + [type(initializer).__name__ for initializer in self._initializers])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()

        plt.pie(sum_columns, labels=sum_columns.index, autopct="%1.1f%%", startangle=140)
        plt.title("On Birth Initializer Times")

        yield

        return
