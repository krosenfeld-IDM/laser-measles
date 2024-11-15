from datetime import datetime

import click
import numpy as np
import pandas as pd
from laser_core.demographics import AliasedDistribution
from laser_core.demographics import load_pyramid_csv
from laser_core.laserframe import LaserFrame
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from laser_core.propertyset import PropertySet
from laser_core.random import seed as seed_prng
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

from laser_measles.measles_births import Births
from laser_measles.utils import calc_capacity
from laser_measles.utils import calc_distances


class Model:
    def __init__(self, scenario: pd.DataFrame, parameters: PropertySet, name: str = "measles") -> None:
        self.tinit = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tinit}: Creating the {name} model…")
        self.scenario = scenario
        self.params = parameters
        self.name = name

        self.prng = seed_prng(parameters.seed if parameters.seed is not None else self.tinit.microsecond)

        click.echo(f"Initializing the {name} model with {len(scenario)} patches…")

        if parameters.verbose:
            click.echo(f"Counties: {scenario.name.values[0:4]}...")
            click.echo(f"Populations: {scenario.population.values[0:4]}...")
            click.echo(f"Lat/longs: {list(zip(scenario.latitude.values, scenario.longitude.values))[0:4]}...")

        # We need some patches with population data ...
        npatches = len(scenario)
        self.patches = LaserFrame(npatches)

        # "activate" all the patches (count == capacity)
        self.patches.add(npatches)
        self.patches.add_vector_property("populations", length=parameters.nticks + 1)
        # set patch populations at t = 0 to initial populations
        self.patches.populations[0, :] = scenario.population

        # ... and connectivity data
        distances = calc_distances(scenario.latitude.values, scenario.longitude.values, parameters.verbose)
        network = gravity(
            scenario.population.values,
            distances,
            parameters.k,
            parameters.a,
            parameters.b,
            parameters.c,
        )
        network = row_normalizer(network, parameters.max_frac)
        self.patches.add_vector_property("network", length=npatches, dtype=np.float32)
        self.patches.network[:, :] = network

        # Initialize the model population

        capacity = calc_capacity(self.patches.populations[0, :].sum(), parameters.nticks, parameters.cbr, parameters.verbose)
        self.population = LaserFrame(capacity)

        self.population.add_scalar_property("nodeid", dtype=np.uint16)
        for nodeid, count in enumerate(self.patches.populations[0, :]):
            first, last = self.population.add(count)
            self.population.nodeid[first:last] = nodeid

        # Initialize population ages

        pyramid_file = parameters.pyramid_file
        age_distribution = load_pyramid_csv(pyramid_file)
        both = age_distribution[:, 2] + age_distribution[:, 3]  # males + females
        sampler = AliasedDistribution(both)
        bin_min_age_days = age_distribution[:, 0] * 365  # minimum age for bin, in days (include this value)
        bin_max_age_days = (age_distribution[:, 1] + 1) * 365  # maximum age for bin, in days (exclude this value)
        initial_pop = self.population.count
        samples = sampler.sample(initial_pop)  # sample for bins from pyramid
        self.population.add_scalar_property("dob", dtype=np.int32)
        mask = np.zeros(initial_pop, dtype=bool)
        dobs = self.population.dob[0:initial_pop]
        click.echo("Assigning day of year of birth to agents…")
        for i in tqdm(range(len(age_distribution))):  # for each possible bin value...
            mask[:] = samples == i  # ...find the agents that belong to this bin
            # ...and assign a random age, in days, within the bin
            dobs[mask] = self.prng.integers(bin_min_age_days[i], bin_max_age_days[i], mask.sum())

        dobs *= -1  # convert ages to date of birth prior to _now_ (t = 0) ∴ negative

        return

    @property
    def components(self) -> list:
        return self._components

    @components.setter
    def components(self, components: list) -> None:
        self._components = components
        self.instances = [self]  # instantiated instances of components
        self.phases = [self]  # callable phases of the model
        for component in components:
            instance = component(self, self.params.verbose)
            self.instances.append(instance)
            if "__call__" in dir(instance):
                self.phases.append(instance)

        # TODO - integrate this above
        births = next(filter(lambda object: isinstance(object, Births), self.instances))
        for instance in self.instances:
            if "on_birth" in dir(instance):
                births.initializers.append(instance)
        return

    def __call__(self, model, tick: int) -> None:
        model.patches.populations[tick + 1, :] = model.patches.populations[tick, :]
        return

    def run(self) -> None:
        self.tstart = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tstart}: Running the {self.name} model for {self.params.nticks} ticks…")

        self.metrics = []
        for tick in tqdm(range(self.params.nticks)):
            timing = [tick]
            for phase in self.phases:
                tstart = datetime.now(tz=None)  # noqa: DTZ005
                phase(self, tick)
                tfinish = datetime.now(tz=None)  # noqa: DTZ005
                delta = tfinish - tstart
                timing.append(delta.seconds * 1_000_000 + delta.microseconds)
            self.metrics.append(timing)

        self.tfinish = datetime.now(tz=None)  # noqa: DTZ005
        print(f"Completed the {self.name} model at {self.tfinish}…")

        if self.params.verbose:
            metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
            plot_columns = metrics.columns[1:]
            sum_columns = metrics[plot_columns].sum()
            width = max(map(len, sum_columns.index))
            for key in sum_columns.index:
                print(f"{key:{width}}: {sum_columns[key]:13,} µs")
            print("=" * (width + 2 + 13 + 3))
            print(f"{'Total:':{width+1}} {sum_columns.sum():13,} microseconds")

        return

    def visualize(self, pdf: bool = True) -> None:
        if not pdf:
            for instance in self.instances:
                for _plot in instance.plot():
                    plt.show()

        else:
            click.echo("Generating PDF output…")
            pdf_filename = f"{self.name} {self.tstart:%Y-%m-%d %H%M%S}.pdf"
            with PdfPages(pdf_filename) as pdf:
                for instance in self.instances:
                    for _plot in instance.plot():
                        pdf.savefig()
                        plt.close()

            click.echo(f"PDF output saved to '{pdf_filename}'.")

        return

    def plot(self, fig: Figure = None):
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Scenario Patches and Populations")
        if "geometry" in self.scenario.columns:
            ax = plt.gca()
            self.scenario.plot(ax=ax)
        scatter = plt.scatter(
            self.scenario.longitude,
            self.scenario.latitude,
            s=self.scenario.population / 1000,
            c=self.scenario.population,
            cmap="inferno",
        )
        plt.colorbar(scatter, label="Population")

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Distribution of Day of Birth for Initial Population")

        count = self.patches.populations[0, :].sum()  # just the initial population
        dobs = self.population.dob[0:count]
        plt.hist(dobs, bins=100)
        plt.xlabel("Day of Birth")

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()

        plt.pie(
            sum_columns,
            labels=[name if not name.startswith("do_") else name[3:] for name in sum_columns.index],
            autopct="%1.1f%%",
            startangle=140,
        )
        plt.title("Update Phase Times")

        yield
        return
