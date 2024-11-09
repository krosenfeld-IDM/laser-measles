import numpy as np
from laser_core.laserframe import LaserFrame


def setup_births(model, verbose: bool = False):
    # We need to estimate the population growth over the course of the simulation to allocate space
    populations = model.patches.populations[0, :]
    initial_pop = populations.sum()
    capacity = calc_capacity(initial_pop, model.params.nticks, model.params.cbr, verbose)
    model.population = LaserFrame(capacity)

    return births_phase


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
        print(f"Population growth: {population:,} … {capacity:,}")
        alternate = np.uint32(population * (1 + cbr / 1000) ** (nticks / 365))
        print(f"Alternate growth:  {population:,} … {alternate:,}")

    return capacity
