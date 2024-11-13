import click
import numpy as np
from laser_core.migration import distance


def calc_distances(latitudes: np.ndarray, longitudes: np.ndarray, verbose: bool = False) -> np.ndarray:
    assert latitudes.ndim == 1, "Latitude array must be one-dimensional"
    assert longitudes.shape == latitudes.shape, "Latitude and longitude arrays must have the same shape"
    npatches = len(latitudes)
    distances = np.zeros((npatches, npatches), dtype=np.float32)
    for i, (lat, long) in enumerate(zip(latitudes, longitudes)):
        distances[i, :] = distance(lat, long, latitudes, longitudes)

    if verbose:
        click.echo(f"Upper left corner of distance matrix:\n{distances[0:4, 0:4]}")

    return distances


def seed_infections_randomly(model, ninfections: int = 100) -> None:
    # Seed initial infections in random locations at the start of the simulation
    cinfections = 0
    while cinfections < ninfections:
        index = model.prng.integers(0, model.population.count)
        if model.population.susceptibility[index] > 0:
            model.population.itimer[index] = model.params.inf_mean
            cinfections += 1

    return


def seed_infections_in_patch(model, ipatch: int, ninfections: int = 100) -> None:
    # Seed initial infections in a specific location at the start of the simulation
    cinfections = 0
    while cinfections < ninfections:
        index = model.prng.integers(0, model.population.count)
        if model.population.susceptibility[index] > 0 and model.population.nodeid[index] == ipatch:
            model.population.itimer[index] = model.params.inf_mean
            cinfections += 1

    return
