"""Base Measles Model"""

from datetime import datetime

import click
import pandas as pd
from laser_core.random import seed as seed_prng
from tqdm import tqdm

from laser_measles.measles_births import setup_births
from laser_measles.measles_incubation import Incubation
from laser_measles.measles_infection import Infection
from laser_measles.measles_init import setup_initial_population
from laser_measles.measles_maternalabs import setup_maternal_antibodies
from laser_measles.measles_metapop import setup_meta_population
from laser_measles.measles_nddeaths import setup_nd_deaths
from laser_measles.measles_params import get_parameters
from laser_measles.measles_ri import setup_routine_immunization
from laser_measles.measles_susceptibility import setup_susceptibility
from laser_measles.measles_transmission import Transmission


class Model:
    """Tabula rasa for the measles model"""


@click.command()
@click.option("--nticks", default=365, help="Number of ticks to run the simulation")
@click.option("--viz", is_flag=True, help="Display visualizations  to help validate the model")
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--params", default=None, help="JSON file with parameters")
@click.option("--output", default=None, help="Output file for results")
@click.option("--seed", default=20241107, help="Random seed")
def run(nticks, seed, verbose, viz, **kwargs):
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

    susceptibility = setup_susceptibility(model, verbose)
    births.initializers.append(susceptibility.on_birth)

    maternal_antibodies = setup_maternal_antibodies(model, verbose)
    births.initializers.append(maternal_antibodies.on_birth)

    # routine immunization setup
    routine_immunization = setup_routine_immunization(model, verbose)
    births.initializers.append(routine_immunization.on_birth)

    # disease dynamics - infection progression
    # infection dynamics come _before_ incubation dynamics so newly set itimers
    # don't immediately expire
    infection = Infection(model, verbose)
    births.initializers.append(infection.on_birth)

    # disease dynamics - incubation progression
    incubation = Incubation(model, verbose)
    births.initializers.append(incubation.on_birth)

    # disease dynamics - transmission
    transmission = Transmission(model, verbose)

    model.phases = [
        metapop,
        births,
        nd_deaths,
        # susceptibility, # no-op
        maternal_antibodies,
        routine_immunization,
        infection,
        incubation,
        transmission,
    ]

    ninitial = 100
    # Seed initial infections in random locations at the start of the simulation
    # cinitial = 0
    # while cinitial < ninitial:
    #     index = model.prng.integers(0, model.population.count)
    #     if model.population.susceptibility[index] > 0:
    #         model.population.itimer[index] = model.params.inf_mean
    #         cinitial += 1

    # Seed initial infections in Node 13 (King County) at the start of the simulation
    # Pierce County is Node 18, Snohomish County is Node 14, Yakima County is 19
    cinitial = 0
    COUNTY = 13
    istart, iend = model.patches.populations[:].cumsum()[COUNTY - 1 : COUNTY + 1]
    while cinitial < ninitial:
        index = model.prng.integers(istart, iend)
        if model.population.susceptibility[index] > 0:
            model.population.itimer[index] = model.params.inf_mean
            cinitial += 1

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

    if viz:
        click.echo("Validating the model…")
        metapop.plot()
        initpop.plot()
        births.plot()
        nd_deaths.plot()
        susceptibility.plot()
        maternal_antibodies.plot()
        routine_immunization.plot()
        incubation.plot()
        infection.plot()
        transmission.plot()

    return


if __name__ == "__main__":
    ctx = click.Context(run)
    ctx.invoke(run, nticks=365, seed=20241107, verbose=True, viz=True)
