"""Nothern Nigeria Measles Model"""

import click
import numpy as np

from laser_measles.measles_births import Births
from laser_measles.measles_incubation import Incubation
from laser_measles.measles_infection import Infection
from laser_measles.measles_maternalabs import MaternalAntibodies
from laser_measles.measles_nddeaths import NonDiseaseDeaths
from laser_measles.measles_ri import RoutineImmunization
from laser_measles.measles_susceptibility import Susceptibility
from laser_measles.measles_transmission import Transmission
from laser_measles.model import Model
from laser_measles.nn_metapop import get_scenario
from laser_measles.nn_params import get_parameters
from laser_measles.utils import seed_infections_in_patch


@click.command()
@click.option("--nticks", default=365, help="Number of ticks to run the simulation")
@click.option("--seed", default=20241107, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--viz", is_flag=True, help="Display visualizations  to help validate the model")
@click.option("--pdf", is_flag=True, help="Output visualization results as a PDF")
@click.option("--output", default=None, help="Output file for results")
@click.option("--params", default=None, help="JSON file with parameters")
@click.option("--param", "-p", multiple=True, help="Additional parameter overrides (param:value or param=value)")
def run(**kwargs):
    parameters = get_parameters(kwargs)
    scenario = get_scenario(parameters, parameters["verbose"])
    model = Model(scenario, parameters, name="northen nigeria measles")

    # infection dynamics come _before_ incubation dynamics so newly set itimers
    # don't immediately expire
    model.components = [
        Births,
        NonDiseaseDeaths,
        Susceptibility,
        MaternalAntibodies,
        RoutineImmunization,
        Infection,
        Incubation,
        Transmission,
    ]

    # seed_infections_randomly(model, ninfections=100)
    # Seed initial infections in most populous patch at the start of the simulation
    ipatch = np.argsort(model.patches.populations[0, :])[-1]
    seed_infections_in_patch(model, ipatch=ipatch, ninfections=100)

    model.run()

    if parameters["viz"]:
        model.visualize(pdf=parameters["pdf"])

    return


if __name__ == "__main__":
    ctx = click.Context(run)
    ctx.invoke(run, nticks=365, seed=20241107, verbose=True, viz=True, pdf=False)
