import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Births:
    def __init__(self, model, verbose: bool = False):
        assert getattr(model, "population", None) is not None, "Births requires the model to have a `population` attribute"
        assert getattr(model.population, "dob", None) is not None, "Births requires the model population to have a `dob` attribute"

        self.__name__ = "births"
        self.model = model

        nyears = (model.params.nticks + 364) // 365
        model.patches.add_vector_property("births", length=nyears, dtype=np.uint32)

        self._initializers = []

        return

    @property
    def initializers(self):
        return self._initializers

    def __call__(self, model, tick) -> None:
        doy = tick % 365 + 1  # day of year 1…365
        year = tick // 365

        if doy == 1:
            model.patches.births[year, :] = model.prng.poisson(model.patches.populations[tick, :] * model.params.cbr / 1000)

        annual_births = model.patches.births[year, :]
        todays_births = (annual_births * doy // 365) - (annual_births * (doy - 1) // 365)
        count_births = todays_births.sum()
        istart, iend = model.population.add(count_births)

        model.population.dob[istart:iend] = tick  # set to current tick

        # set the nodeids for the newborns in case subsequent initializers need them (properties varying by patch)
        index = istart
        nodeids = model.population.nodeid
        for nodeid, births in enumerate(todays_births):
            nodeids[index : index + births] = nodeid
            index += births

        for initializer in self._initializers:
            initializer(model, tick, istart, iend)

        model.patches.populations[tick + 1, :] += todays_births

        return

    def plot(self, fig: Figure = None) -> None:
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Births in Top 5 Most Populous Patches")

        # indices = [np.where(self._model.counties.names == county)[0][0] for county in counties]
        indices = self.model.patches.populations[0, :].argsort()[-5:]
        ax1 = plt.gca()
        ticks = list(range(0, self.model.params.nticks, 365))
        for index in indices:
            ax1.plot(self.model.patches.populations[ticks, index], marker="x", markersize=4)

        ax2 = ax1.twinx()
        for index in indices:
            ax2.plot(self.model.patches.births[:, index], marker="+", markersize=4)

        return
