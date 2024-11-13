"""Base Measles Model"""

from datetime import datetime

import click
import pandas as pd
from laser_core.random import seed as seed_prng
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from laser_measles.measles_births import Births
from laser_measles.measles_incubation import Incubation
from laser_measles.measles_infection import Infection
from laser_measles.measles_init import InitialPopulation
from laser_measles.measles_maternalabs import MaternalAntibodies
from laser_measles.measles_metapop import MetaPopulation
from laser_measles.measles_nddeaths import NonDiseaseDeaths
from laser_measles.measles_params import get_parameters
from laser_measles.measles_ri import RoutineImmunization
from laser_measles.measles_susceptibility import Susceptibility
from laser_measles.measles_transmission import Transmission
from laser_measles.utils import seed_infections_in_patch


class Model:
    """Tabula rasa for the measles model"""


@click.command()
@click.option("--nticks", default=365, help="Number of ticks to run the simulation")
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--params", default=None, help="JSON file with parameters")
@click.option("--output", default=None, help="Output file for results")
@click.option("--seed", default=20241107, help="Random seed")
@click.option("--viz", is_flag=True, help="Display visualizations  to help validate the model")
@click.option("--pdf", is_flag=True, help="Output visualization results as a PDF")
def run(nticks, seed, verbose, viz, pdf, **kwargs):
    """Run the measles model"""
    model = Model()
    model.tstart = datetime.now(tz=None)  # noqa: DTZ005
    click.echo(f"{model.tstart}: Running the measles model for {nticks} ticks…")

    model.prng = seed_prng(seed if seed is not None else model.tstart.microsecond)

    model.params = get_parameters(nticks, verbose, kwargs)

    # infection dynamics come _before_ incubation dynamics so newly set itimers
    # don't immediately expire
    components = [
        MetaPopulation,
        InitialPopulation,
        Births,
        NonDiseaseDeaths,
        Susceptibility,
        MaternalAntibodies,
        RoutineImmunization,
        Infection,
        Incubation,
        Transmission,
    ]
    instances = []
    model.phases = []
    for component in components:
        instance = component(model, verbose)
        instances.append(instance)
        if "__call__" in dir(instance):
            model.phases.append(instance)

    # TODO - integrate this above
    births = next(filter(lambda object: isinstance(object, Births), instances))
    for instance in instances:
        if "on_birth" in dir(instance):
            births.initializers.append(instance.on_birth)

    # seed_infections_randomly(model, ninfections=100)
    # Seed initial infections in Node 13 (King County) at the start of the simulation
    # Pierce County is Node 18, Snohomish County is Node 14, Yakima County is 19
    seed_infections_in_patch(model, ipatch=13, ninfections=100)

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

    model.tfinish = datetime.now(tz=None)  # noqa: DTZ005
    print(f"Completed the measles model at {model.tfinish}…")

    if verbose:
        metrics = pd.DataFrame(model.metrics, columns=["tick"] + [phase.__name__ for phase in model.phases])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()
        width = max(map(len, sum_columns.index))
        for key in sum_columns.index:
            print(f"{key:{width}}: {sum_columns[key]:13,} µs")
        print("=" * (width + 2 + 13 + 3))
        print(f"{'Total:':{width+1}} {sum_columns.sum():13,} microseconds")

    if viz:
        click.echo("Validating the model…")
        if not pdf:
            for instance in instances:
                instance.plot()
                plt.show()

            plt.pie(
                sum_columns,
                labels=[name if not name.startswith("do_") else name[3:] for name in sum_columns.index],
                autopct="%1.1f%%",
                startangle=140,
            )
            plt.title("Update Phase Times")
            plt.show()
        else:
            click.echo("Generating PDF output…")
            pdf_filename = f"measles {model.tstart:%Y-%m-%d %H%M%S}.pdf"
            with PdfPages(pdf_filename) as pdf:
                for instance in instances:
                    instance.plot()
                    pdf.savefig()
                    plt.close()

                plt.pie(
                    sum_columns,
                    labels=[name if not name.startswith("do_") else name[3:] for name in sum_columns.index],
                    autopct="%1.1f%%",
                    startangle=140,
                )
                plt.title("Update Phase Times")
                pdf.savefig()
                plt.close()
            click.echo(f"PDF output saved to '{pdf_filename}'.")

    return


if __name__ == "__main__":
    ctx = click.Context(run)
    ctx.invoke(run, nticks=365, seed=20241107, verbose=True, viz=True, pdf=False)
