import click
import numpy as np
from laser_core.demographics import KaplanMeierEstimator
from laser_core.sortedqueue import SortedQueue
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm


class NonDiseaseDeaths:
    def __init__(self, model, verbose: bool = False):
        self.__name__ = "non_disease_deaths"
        self.model = model

        model.population.add_scalar_property("alive", dtype=bool, default=True)
        # model.population.alive[0 : model.population.count] = True

        model.population.add_scalar_property("dod", dtype=np.uint16)  # Up to 65535 days in the future
        cumulative_deaths = np.loadtxt(model.params.mortality_file)
        model.estimator = KaplanMeierEstimator(cumulative_deaths)
        dods = model.population.dod[0 : model.population.count]
        dobs = model.population.dob[0 : model.population.count]
        # Use -dobs to get the current age of the agent (in days)
        dods[:] = model.estimator.predict_age_at_death(-dobs, max_year=100)

        dods -= dobs.astype(dods.dtype)  # renormalize to be relative to _now_ (t = 0)

        # add non-disease mortality to the model
        model.nddq = SortedQueue(model.population.capacity, model.population.dod)
        print("Adding agents to the non-disease death queue…")
        for i in tqdm(np.nonzero(dods[0 : model.population.count] < model.params.nticks)[0]):
            model.nddq.push(i)

        # +364 to account for something other than integral numbers of years (in nticks)
        # model.patches.add_vector_property("deaths", (model.params.nticks + 364) // 365)
        model.patches.add_vector_property("deaths", length=model.params.nticks)

        return

    def on_birth(self, model, tick, istart, iend):
        # newborns are alive and have a predicted date of death
        model.population.alive[istart:iend] = True
        model.population.dod[istart:iend] = 0  # temporarily set to 0 for the next line
        model.population.dod[istart:iend] = tick + model.estimator.predict_age_at_death(model.population.dod[istart:iend], max_year=100)

        max_tick = model.params.nticks
        dods = model.population.dod[0 : model.population.count]
        q = model.nddq
        for agent in range(istart, iend):
            if dods[agent] < max_tick:
                q.push(agent)

        return

    def __call__(self, model, tick):
        nodeids = model.population.nodeid[0 : model.population.count]
        node_population = model.patches.populations[tick, :]
        node_deaths = model.patches.deaths[tick, :]
        alive = model.population.alive[0 : model.population.count]
        # susceptibility = model.population.susceptibility[0 : model.population.count]
        # ma_timers = model.population.ma_timers[0 : model.population.count]
        # ri_timers = model.population.ri_timers[0 : model.population.count]
        # etimers = model.population.etimers[0 : model.population.count]
        # itimers = model.population.itimers[0 : model.population.count]

        pq = model.nddq
        while (len(pq) > 0) and (pq.peekv() <= tick):
            iagent = pq.popi()
            nodeid = nodeids[iagent]
            node_population[nodeid] -= 1
            node_deaths[nodeid] += 1
            alive[iagent] = False
            # susceptibility[iagent] = 0
            # ma_timers[iagent] = 0
            # ri_timers[iagent] = 0
            # etimers[iagent] = 0
            # itimers[iagent] = 0

        return

    def plot(self, fig: Figure = None):
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Cumulative Non-Disease Deaths for Year 0 Population")

        dobs = self.model.population.dob[0 : self.model.population.count]
        dods = self.model.population.dod[0 : self.model.population.count]
        individuals = np.nonzero((0 < dobs) & (dobs < 365))[0]
        if len(individuals) > 0:
            ages_at_death = (dods[individuals] - dobs[individuals]) // 365
            aad_max = ages_at_death.max()
            counts = np.zeros(aad_max + 1, dtype=np.int32)
            np.add.at(counts, ages_at_death, 1)  # [individuals], 1)
            cumulative = counts.cumsum()
            plt.plot(range(aad_max + 1), cumulative, marker="x", markersize=4, color="red")

            percentage = self.model.estimator.cumulative_deaths / self.model.estimator.cumulative_deaths[-1]
            expected_deaths = np.round(len(individuals) * percentage).astype(np.uint32)

            plt.plot(range(aad_max + 1), expected_deaths, marker="+", markersize=4, color="blue")
            plt.xlabel("Years Since Birth")
            yield
        else:
            click.echo("Found no individuals born in the first year.")

        return
