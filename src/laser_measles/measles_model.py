"""Base Measles Model"""

from datetime import datetime

import click
import pandas as pd
from laser_core.random import seed as seed_prng
from tqdm import tqdm

from laser_measles.measles_births import setup_births
from laser_measles.measles_init import setup_initial_population
from laser_measles.measles_metapop import setup_meta_population
from laser_measles.measles_nddeaths import setup_nd_deaths
from laser_measles.measles_params import get_parameters
from laser_measles.measles_susceptibility import setup_susceptibility


class Model:
    """Tabula rasa for the measles model"""


@click.command()
@click.option("--nticks", default=365, help="Number of ticks to run the simulation")
@click.option("--validate", is_flag=True, help="Validate the model")
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--params", default=None, help="JSON file with parameters")
@click.option("--output", default=None, help="Output file for results")
@click.option("--seed", default=20241107, help="Random seed")
def run(nticks, seed, verbose, validate, **kwargs):
    """Run the measles model"""
    click.echo(f"{datetime.now(tz=None)}: Running the measles model for {nticks} ticks…")  # noqa: DTZ005

    model = Model()
    model.prng = seed_prng(seed if seed is not None else datetime.now().microsecond)  # noqa: DTZ005

    model.params = get_parameters(nticks, verbose, kwargs)

    metapop = setup_meta_population(model, verbose)  # patch populations and network

    initpop = setup_initial_population(model, verbose)  # initialize population ages

    births = setup_births(model, verbose)  # vital dynamics setup - births

    nd_deaths = setup_nd_deaths(model, verbose)  # vital dynamics setup - deaths
    births.initializers.append(nd_deaths.on_birth)

    # immune system setup
    susceptibility = setup_susceptibility(model, verbose)
    births.initializers.append(susceptibility.on_birth)

    # maternal antibody setup
    # routine immunization setup
    # disease dynamics - incubation progression
    # disease dynamics - infection progression
    # disease dynamics - transmission

    model.phases = [
        metapop,
        births,
        nd_deaths,
        # susceptibility, # no-op
    ]

    model.metrics = []
    for tick in tqdm(range(nticks)):
        timing = [tick]
        for phase in model.phases:
            tstart = datetime.now(tz=None)  # noqa: DTZ005
            phase(model, tick)
            tfinish = datetime.now(tz=None)  # noqa: DTZ005
            delta = tfinish - tstart
            timing.append(delta.seconds * 1_000_000 + delta.microseconds)
        model.metrics.append(timing)

    if True:
        metrics = pd.DataFrame(model.metrics, columns=["tick"] + [phase.__name__ for phase in model.phases])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()
        print(sum_columns)
        print("=" * 36)
        print(f"Total: {sum_columns.sum():29,} microseconds")

    if validate:
        click.echo("Validating the model…")
        # metapop.plot()
        # initpop.plot()
        # births.plot()
        # nd_deaths.plot()
        susceptibility.plot()

    return


if __name__ == "__main__":
    ctx = click.Context(run)
    ctx.invoke(run, nticks=365, seed=20241107, verbose=True, validate=True)
