import click
import numpy as np
from laser_core.demographics import AliasedDistribution
from laser_core.demographics import load_pyramid_csv
from laser_core.laserframe import LaserFrame
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm


class InitialPopulation:
    def __init__(self, model, verbose: bool = False):
        self.model = model

        capacity = calc_capacity(model.patches.populations[0, :].sum(), model.params.nticks, model.params.cbr, verbose)
        model.population = LaserFrame(capacity)

        model.population.add_scalar_property("nodeid", dtype=np.uint16)
        for nodeid, count in enumerate(model.patches.populations[0, :]):
            first, last = model.population.add(count)
            model.population.nodeid[first:last] = nodeid

        # Initialize population ages

        pyramid_file = model.params.pyramid_file
        age_distribution = load_pyramid_csv(pyramid_file)
        both = age_distribution[:, 2] + age_distribution[:, 3]  # males + females
        sampler = AliasedDistribution(both)
        bin_min_age_days = age_distribution[:, 0] * 365  # minimum age for bin, in days (include this value)
        bin_max_age_days = (age_distribution[:, 1] + 1) * 365  # maximum age for bin, in days (exclude this value)
        initial_pop = model.population.count
        samples = sampler.sample(initial_pop)  # sample for bins from pyramid
        model.population.add_scalar_property("dob", dtype=np.int32)
        mask = np.zeros(initial_pop, dtype=bool)
        dobs = model.population.dob[0:initial_pop]
        click.echo("Assigning day of year of birth to agents…")
        for i in tqdm(range(len(age_distribution))):  # for each possible bin value...
            mask[:] = samples == i  # ...find the agents that belong to this bin
            # ...and assign a random age, in days, within the bin
            dobs[mask] = model.prng.integers(bin_min_age_days[i], bin_max_age_days[i], mask.sum())

        dobs *= -1  # convert ages to date of birth prior to _now_ (t = 0) ∴ negative

        return

    def plot(self, fig: Figure = None) -> None:
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        fig.suptitle("Distribution of Day of Birth for Initial Population")

        count = self.model.patches.populations[0, :].sum()  # just the initial population
        dobs = self.model.population.dob[0:count]
        plt.hist(dobs, bins=100)
        plt.xlabel("Day of Birth")

        return


def calc_capacity(population: np.uint32, nticks: np.uint32, cbr: np.float32, verbose: bool = False) -> np.uint32:
    # We assume a constant birth rate (CBR) for the population growth
    # The formula is: P(t) = P(0) * (1 + CBR)^t
    # where P(t) is the population at time t, P(0) is the initial population, and t is the number of ticks
    # We need to allocate space for the population data for each tick
    # We will use the maximum population growth to estimate the capacity
    # We will use the maximum population growth to estimate the capacity
    daily_rate = (cbr / 1000) / 365.0  # CBR is per 1000 people per year
    capacity = np.uint32(population * (1 + daily_rate) ** nticks)

    if verbose:
        click.echo(f"Population growth: {population:,} … {capacity:,}")
        alternate = np.uint32(population * (1 + cbr / 1000) ** (nticks / 365))
        click.echo(f"Alternate growth:  {population:,} … {alternate:,}")

    return capacity
