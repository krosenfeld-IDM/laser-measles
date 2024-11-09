"""Northern Nigeria Measles Model (NNMM)"""

from datetime import datetime
from pathlib import Path

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from laser_core.demographics.kmestimator import KaplanMeierEstimator
from laser_core.random import seed as seed_prng
from laser_core.sortedqueue import SortedQueue
from tqdm import tqdm

from laser_measles.nn_births import setup_births
from laser_measles.nn_initpop import setup_initial_population
from laser_measles.nn_metapop import setup_meta_population
from laser_measles.nn_params import get_parameters


class Model:
    """Tabula rasa for the Northern Nigeria Measles Model (NNMM)"""


@click.command()
@click.option("--nticks", type=int, default=365)
@click.option("--seed", type=int, default=20241028)
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--params", type=click.Path(exists=True), default=None, help="Path to a JSON file with model parameters")
@click.option("--param", "-p", type=str, multiple=True, help="Override a parameter from the command line")
def run(nticks: int, seed: int, verbose: bool, **kwargs) -> None:
    """Run the Northern Nigeria Measles Model (NNMM)"""
    print(f"Running the Northern Nigeria Measles Model (NNMM) at {datetime.now()}…")  # noqa: DTZ005

    # Note, we are using the laser_core PRNG here which includes seeding the threads used by Numba
    # in compiled functions. Runs on hardware with a different number of cores or setting the
    # environment variable NUMBA_NUM_THREADS/OMP_NUM_THREADS will affect the results even with the
    # same initial seed.

    model = Model()
    model.prng = seed_prng(seed if seed is not None else datetime.now().microsecond)  # noqa: DTZ005

    # get parameters
    model.params = get_parameters(nticks, verbose, kwargs)

    # meta population setup
    setup_meta_population(model, verbose)

    # initial population demographics
    setup_initial_population(model, verbose)

    # vital dynamics setup - births
    setup_births(model, verbose)

    # vital dynamics setup - deaths

    # immune system setup

    # maternal antibody setup

    # routine immunization setup

    # disease dynamics - incubation progression

    # disease dynamics - infection progression

    # disease dynamics - transmission

    # Predict the age at death for each agent
    model.population.add_scalar_property("dod", dtype=np.uint16)  # Up to 65535 days in the future
    mortality_file = Path(__file__).parent / "nigeria_mortality.csv"
    cumulative_deaths = np.loadtxt(mortality_file)
    estimator = KaplanMeierEstimator(cumulative_deaths)
    dods = model.population.dod[0:initial_pop]
    dods[:] = estimator.predict_age_at_death(dobs, max_year=100)

    dods -= dobs.astype(dods.dtype)  # renormalize to be relative to _now_ (t = 0)
    dobs *= -1  # convert ages to date of birth prior to _now_ (t = 0) ∴ negative

    # add non-disease mortality to the model
    model.nddq = SortedQueue(capacity, model.population.dod)
    print("Adding agents to the non-disease death queue…")
    for i in tqdm(np.nonzero(dods[0:initial_pop] < model.params.nticks)[0]):
        model.nddq.push(i)

    # +364 to account for something other than integral numbers of years (in nticks)
    # model.patches.add_vector_property("deaths", (model.params.nticks + 364) // 365)
    model.patches.add_vector_property("deaths", length=model.params.nticks)

    model.population.add_scalar_property("etimer", dtype=np.uint8)  # Only need up to ~20 days
    model.population.add_scalar_property("itimer", dtype=np.uint8)  # Only need up to ~20 days

    @nb.njit((nb.uint32, nb.uint8[:]), parallel=True)
    def nb_infection_update(count, itimers):
        for i in nb.prange(count):
            if itimers[i] > 0:
                itimers[i] -= 1

        return

    @nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:], nb.float32, nb.float32), parallel=True)
    def nb_exposure_update(count, etimers, itimers, inf_mean, inf_std):
        for i in nb.prange(count):
            if etimers[i] > 0:
                etimers[i] -= 1
                if etimers[i] == 0:
                    itimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std))))

        return

    model.population.add_scalar_property("susceptibility", dtype=np.uint8)

    # initialize susceptibility based on age
    @nb.njit((nb.uint32, nb.int32[:], nb.uint8[:]), parallel=True)
    def initialize_susceptibility(count, dob, susceptibility):
        five_years = -5 * 365
        for i in nb.prange(count):
            if dob[i] >= five_years:
                susceptibility[i] = nb.uint8(1)

        return

    initialize_susceptibility(model.population.count, dobs, model.population.susceptibility)

    # Routine Immunization (RI)

    model.patches.add_scalar_property("ri_coverage", dtype=np.float32)
    model.patches.ri_coverage[:] = model.prng.poisson(model.params.ri_coverage * 100, model.patches.count) / 100
    model.population.add_scalar_property("mcv", dtype=np.uint8)

    GET_MCV1 = 1
    GET_MCV2 = 2
    GET_NONE = 0

    def set_mcv_status(model, istart, iend):
        mcv1_cutoff = model.patches.ri_coverage * model.params.probability_mcv1_take  # probability of (MCV1 vaccination) _and_ (MCV1 take)
        mcv2_cutoff = (
            mcv1_cutoff + model.patches.ri_coverage * (1.0 - model.params.probability_mcv1_take) * model.params.probability_mcv2_take
        )  # probability of (MCV1 vaccination) _and_ (not MCV1 take) and (MCV2 take)

        draws = model.prng.random(size=(iend - istart))
        nodeids = model.population.nodeid[istart:iend]
        get_mcv1 = draws <= mcv1_cutoff[nodeids]
        get_mcv2 = (draws > mcv1_cutoff[nodeids]) & (draws <= mcv2_cutoff[nodeids])
        # get_none = (draws > mcv2_cutoff[nodeids]) # "get_none" is the default
        mcv = model.population.mcv[istart:iend]
        mcv[get_mcv1] = GET_MCV1
        mcv[get_mcv2] = GET_MCV2

        return

    model.population.add_scalar_property("ri_timer", np.uint16)  # Use uint16 for timer since 15 months = 450 days > 2^8

    def set_mcv_timers(model, istart, iend):
        count = iend - istart
        ri_timer_values_mcv1 = model.prng.integers(model.params.mcv1_start, model.params.mcv1_end, count).astype(
            model.population.ri_timer.dtype
        )
        ri_timer_values_mcv2 = model.prng.integers(model.params.mcv2_start, model.params.mcv2_end, count).astype(
            model.population.ri_timer.dtype
        )

        mcv = model.population.mcv[istart:iend]

        mask_mcv1 = mcv == GET_MCV1
        mask_mcv2 = mcv == GET_MCV2
        mask_none = mcv == GET_NONE  # for validation
        if mask_mcv1.sum() == 0:
            raise ValueError("Didn't find anyone with mcv set to GET_MCV1")
        if mask_mcv2.sum() == 0:
            raise ValueError("Didn't find anyone with mcv set to GET_MCV2")
        if mask_none.sum() == 0:
            raise ValueError("Didn't find anyone with mcv set to GET_NONE")

        timers = model.population.ri_timer[istart:iend]
        timers[mask_mcv1] = ri_timer_values_mcv1[mask_mcv1]
        timers[mask_mcv2] = ri_timer_values_mcv2[mask_mcv2]

        return

    @nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True)
    def nb_ri_update(count, ri_timer, susceptibility):
        for i in nb.prange(count):
            timer = ri_timer[i]
            if timer > 0:
                timer -= 1
                ri_timer[i] = timer
                if timer == 0:
                    susceptibility[i] = 1

        return

    # maternal immunity
    model.population.add_scalar_property("ma_timer", np.uint8)  # Use uint8 for timer since 6 months ~ 180 days < 2^8

    def init_ma_timers(model, istart, iend):
        model.population.susceptibility[istart:iend] = 0
        model.population.ma_timer[istart:iend] = int(6 * 365 / 12)

        return

    # Note this interacts well with RI since RI _before_ maternal antibody
    # waning will set susceptibility to 0 (with no effect, because it is
    # already 0) and when the ma_timer elapses, susceptibility will be set to 1.
    # ∴ RI _before_ maternal antibody waning will have no effect.
    # RI _after_ maternal antibody waning will correctly set susceptibility to 1

    @nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:]), parallel=True)
    def nb_update_ma_timer(count, ma_timer, susceptibility):
        for i in nb.prange(count):
            timer = ma_timer[i]
            if timer > 0:
                timer -= 1
                ma_timer[i] = timer
                if timer == 0:
                    susceptibility[i] = 1

        return

    # initial infections
    prevalence = 2.5 / 100  # 2.5% prevalence
    initial_infections = model.prng.poisson(prevalence * model.patches.populations[0, :]).astype(np.uint32)

    @nb.njit((nb.uint32, nb.uint32[:], nb.uint16[:], nb.uint8[:], nb.uint8[:], nb.float32, nb.float32), parallel=True)
    def initialize_infections(count, infections, nodeids, susceptibility, etimers, exp_shape, exp_scale):
        for i in nb.prange(count):
            if susceptibility[i] > 0:
                ninfs = infections[nodeids[i]]
                if ninfs > 0:
                    ninfs -= 1
                    infections[nodeids[i]] = ninfs
                    etimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.gamma(exp_shape, exp_scale))))

        return

    initialize_infections(
        model.population.count,
        initial_infections,
        model.population.nodeid,
        model.population.susceptibility,
        model.population.etimer,
        model.params.exp_shape,
        model.params.exp_scale,
    )

    # transmission

    model.patches.add_vector_property("cases", model.params.nticks, dtype=np.uint32)
    model.patches.add_scalar_property("forces", dtype=np.float32)
    model.patches.add_vector_property("incidence", model.params.nticks, dtype=np.uint32)

    @nb.njit(
        (nb.uint8[:], nb.uint16[:], nb.float32[:], nb.uint8[:], nb.uint32, nb.float32, nb.float32, nb.uint32[:]),
        parallel=True,
        nogil=True,
        cache=True,
    )
    def nb_transmission_update(susceptibilities, nodeids, forces, etimers, count, exp_shape, exp_scale, incidence):
        for i in nb.prange(count):
            susceptibility = susceptibilities[i]
            if susceptibility > 0:
                nodeid = nodeids[i]
                force = susceptibility * forces[nodeid]  # force of infection attenuated by personal susceptibility
                if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                    susceptibilities[i] = 0.0  # no longer susceptible
                    # set exposure timer for newly infected individuals to a draw from a gamma distribution, must be at least 1 day
                    etimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.gamma(exp_shape, exp_scale))))

                    incidence[nodeid] += 1

        return

    model.patches.add_vector_property("susceptible", model.params.nticks, dtype=np.uint32)

    @nb.njit((nb.uint32, nb.uint8[:], nb.uint16[:], nb.uint32[:]), parallel=True)
    def nb_survey(count, susceptible, nodeids, journal):
        for i in nb.prange(count):
            if susceptible[i] > 0:
                journal[nodeids[i]] += 1

        return

    ########## model step phases ##########

    # propagate current population to the next tick
    def propagate_population(model: Model, tick: int) -> None:
        model.patches.populations[tick + 1, :] = model.patches.populations[tick, :]

        return

    def do_non_disease_deaths(model: Model, tick: int) -> None:
        nodeids = model.population.nodeid[
            0 : model.population.count
        ]  # could leave [0:count] out, but this should catch out of range accesses
        node_population = model.patches.populations[tick + 1, :]
        # year = tick // 365

        # node_deaths = model.patches.deaths[year, :]
        node_deaths = model.patches.deaths[tick, :]
        alive = model.population.alive[0 : model.population.count]  # could leave [0:count] out, but this should catch out of range accesses

        susceptibility = model.population.susceptibility
        ma_timers = model.population.ma_timer
        ri_timers = model.population.ri_timer
        etimers = model.population.etimer
        itimers = model.population.itimer

        pq = model.nddq
        while (len(pq) > 0) and (pq.peekv() <= tick):
            iagent = pq.popi()
            nodeid = nodeids[iagent]
            node_population[nodeid] -= 1
            node_deaths[nodeid] += 1
            alive[iagent] = False
            susceptibility[iagent] = 0
            ma_timers[iagent] = 0
            ri_timers[iagent] = 0
            etimers[iagent] = 0
            itimers[iagent] = 0

        return

    # +364 to account for something other than integral numbers of years (in nticks)
    model.patches.add_vector_property("births", length=(model.params.nticks + 364) // 365)

    def do_births(model: Model, tick: int) -> None:
        doy = tick % 365 + 1  # day of year 1…365
        year = tick // 365

        if doy == 1:
            model.patches.births[year, :] = model.prng.poisson(model.patches.populations[tick, :] * model.params.cbr / 1000)

        annual_births = model.patches.births[year, :]
        todays_births = (annual_births * doy // 365) - (annual_births * (doy - 1) // 365)
        count_births = todays_births.sum()
        istart, iend = model.population.add(count_births)

        model.population.alive[istart:iend] = True

        model.population.dob[istart:iend] = 0  # temporarily set to 0 for the next line
        model.population.dod[istart:iend] = tick + estimator.predict_age_at_death(model.population.dob[istart:iend], max_year=100)
        model.population.dob[istart:iend] = tick  # set to current tick

        index = istart
        nodeids = model.population.nodeid
        dods = model.population.dod
        max_tick = model.params.nticks
        for nodeid, births in enumerate(todays_births):
            nodeids[index : index + births] = nodeid
            for agent in range(index, index + births):
                if dods[agent] < max_tick:
                    model.nddq.push(agent)
            index += births
        model.patches.populations[tick + 1, :] += todays_births

        set_mcv_status(model, istart, iend)
        set_mcv_timers(model, istart, iend)
        init_ma_timers(model, istart, iend)

        return

    def do_infection_update(model, tick):
        nb_infection_update(np.uint32(model.population.count), model.population.itimer)
        return

    def do_exposure_update(model, tick):
        nb_exposure_update(
            np.uint32(model.population.count),
            model.population.etimer,
            model.population.itimer,
            np.float32(model.params.inf_mean),
            np.float32(model.params.inf_std),
        )
        return

    def do_ri_update(model, tick):
        nb_ri_update(np.uint32(model.population.count), model.population.ri_timer, model.population.susceptibility)
        return

    def do_ma_update(model, tick):
        nb_update_ma_timer(np.uint32(model.population.count), model.population.ma_timer, model.population.susceptibility)
        return

    def do_transmission_update(model, tick) -> None:
        patches = model.patches
        population = model.population

        contagion = patches.cases[tick, :]  # we will accumulate current infections into this view into the cases array
        nodeids = population.nodeid[0 : population.count]  # just look at the active agent indices
        itimers = population.itimer[0 : population.count]  # just look at the active agent indices
        np.add.at(contagion, nodeids[itimers > 0], 1)  # increment by the number of active agents with non-zero itimer

        network = patches.network
        transfer = (contagion * network).round().astype(np.uint32)
        contagion += transfer.sum(axis=1)  # increment by incoming "migration"
        contagion -= transfer.sum(axis=0)  # decrement by outgoing "migration"

        forces = patches.forces
        beta_effective = model.params.beta + model.params.seasonality_factor * np.sin(
            2 * np.pi * (tick - model.params.seasonality_phase) / 365
        )
        np.multiply(contagion, beta_effective, out=forces)
        np.divide(forces, model.patches.populations[tick, :], out=forces)  # per agent force of infection

        nb_transmission_update(
            population.susceptibility,
            population.nodeid,
            forces,
            population.etimer,
            population.count,
            model.params.exp_shape,
            model.params.exp_scale,
            model.patches.incidence[tick, :],
        )

        return

    def do_survey(model, tick):
        # susceptible = model.population.susceptibility[0 : model.population.count]
        # nodeids = model.population.nodeid[0 : model.population.count]
        # np.add.at(model.patches.susceptible[tick, :], nodeids[susceptible > 0], 1)
        nb_survey(
            np.uint32(model.population.count), model.population.susceptibility, model.population.nodeid, model.patches.susceptible[tick, :]
        )

        return

    model.phases = [
        propagate_population,
        do_births,
        do_non_disease_deaths,
        do_infection_update,
        do_exposure_update,
        do_ri_update,
        do_ma_update,
        do_transmission_update,
        do_survey,
    ]

    model.metrics = []

    print(f"Running the model for {model.params.nticks} ticks…")
    for tick in tqdm(range(model.params.nticks)):
        metrics = [tick]
        for phase in model.phases:
            tstart = datetime.now(tz=None)  # noqa: DTZ005
            phase(model, tick)
            tfinish = datetime.now(tz=None)  # noqa: DTZ005
            delta = tfinish - tstart
            metrics.append(delta.seconds * 1_000_000 + delta.microseconds)
        model.metrics.append(metrics)

    print(f"Completed the Northern Nigeria Measles Model (NNMM) at {datetime.now()}…")  # noqa: DTZ005

    metrics = pd.DataFrame(model.metrics, columns=["tick"] + [phase.__name__ for phase in model.phases])
    plot_columns = metrics.columns[1:]
    sum_columns = metrics[plot_columns].sum()
    print(sum_columns)
    print("=" * 36)
    print(f"Total: {sum_columns.sum():29,} microseconds")

    # Page One
    nrows = 2
    ncols = 2

    fig = plt.figure(figsize=(16, 9))  # , dpi=200)

    # Upper Left
    fig.add_subplot(nrows, ncols, 1)
    plt.title("Node 0")
    ax1 = plt.gca()
    ax1.plot(model.patches.populations[:, 0], label="Population")
    ax1.set_ylabel("Population")
    # ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(model.patches.deaths[:, 0], color="red", label="Deaths")
    ax2.set_ylabel("Deaths")
    # ax2.legend(loc="upper left")
    plt.legend(loc="upper right")

    # Upper Right
    fig.add_subplot(nrows, ncols, 2)
    plt.title("Population DOB")
    plt.hist(model.population.dob[0 : model.population.count], bins=100, alpha=0.5)  # , label="DOB")

    # Lower Left
    fig.add_subplot(nrows, ncols, 3)
    plt.title("Deaths by year for newborns (age < 1 year)")

    # expected deaths at age 0
    percentage = cumulative_deaths / cumulative_deaths[-1]
    expected_deaths = model.patches.births[-1, :].sum() * percentage

    dobs = model.population.dob[0 : model.population.count]
    dods = model.population.dod[0 : model.population.count]
    ages_in_years = (model.params.nticks - dobs) // 365
    ages_at_death = (dods - dobs) // 365
    aad_max = ages_at_death.max()
    age = 0
    individuals = np.nonzero(ages_in_years == age)[0]
    counts = np.zeros(aad_max + 1, dtype=np.int32)
    np.add.at(counts, ages_at_death[individuals], 1)
    cumulative = counts.cumsum()
    plt.plot(range(aad_max + 1), cumulative, "x", label=f"{age}")
    plt.plot(range(aad_max + 1), expected_deaths, marker="o", markersize=6, label="Expected")

    # Lower Right
    fig.add_subplot(nrows, ncols, 4)
    plt.pie(
        sum_columns,
        labels=[name if not name.startswith("do_") else name[3:] for name in sum_columns.index],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Update Phase Times")

    fig.legend()
    mgr = plt.get_current_fig_manager()
    mgr.full_screen_toggle()
    plt.show()

    # Page Two
    nrows = 2
    ncols = 2

    fig = plt.figure(figsize=(16, 9))  # , dpi=200)

    # Upper Left
    fig.add_subplot(nrows, ncols, 1)
    plt.title("Incidence")

    # We will plot the incidence for the largest N and smallest N nodes
    N = 3
    indices = np.argsort(model.patches.populations[0, :])

    ax1 = plt.gca()
    ax1.plot(model.patches.cases.sum(axis=1), color="red", label="Cases")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()

    for i in range(N):
        inode = indices[i]
        ax2.plot(model.patches.incidence[:, inode], marker="o", label=f"Node {inode}({i}) Incidence")

    for i in range(-N, 0):
        inode = indices[i]
        ax2.plot(model.patches.incidence[:, inode], marker="+", label=f"Node {inode}({i}) Incidence")

    ax2.legend(loc="center right")

    # Upper Right

    fig.add_subplot(nrows, ncols, 2)
    inode = indices[-1]
    plt.title(f"S-I Space Trajectory for Node {inode}")
    s_fraction = model.patches.susceptible[:, inode] / model.patches.populations[1:, inode]
    i_fraction = model.patches.cases[:, inode] / model.patches.populations[1:, inode]
    plt.xlabel("Susceptible Fraction")
    plt.ylabel("Infected Fraction")
    plt.plot(s_fraction, i_fraction, "x")

    # Lower Left

    nn_latitudes = [node[1][1] for node in nn_nodes.values()]
    nn_longitudes = [node[1][0] for node in nn_nodes.values()]
    nn_populations = [node[0][0] for node in nn_nodes.values()]
    nn_sizes = 0.15 * np.sqrt(nn_populations)

    #####

    shapefile_path = Path("~/Downloads/nga_adm_osgof_20190417/nga_admbnda_adm2_osgof_20190417.shp").expanduser()
    shapefile = gpd.read_file(shapefile_path)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(111)
    shapefile[shapefile.placetype == "region"].plot(ax=ax)
    plt.scatter(nn_longitudes, nn_latitudes, s=nn_sizes, c=nn_populations, norm=LogNorm(), cmap="inferno")
    plt.xlim(-6, 2)
    plt.ylim(49, 57)
    plt.colorbar(label="Population")

    mgr = plt.get_current_fig_manager()
    mgr.full_screen_toggle()
    plt.show()

    return


if __name__ == "__main__":
    ctx = click.Context(run)  # Create a click context
    ctx.invoke(run, nticks=365, seed=20241029, verbose=True)  # Call the command with nticks, seed, and verbose
