import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import row_normalizer

from laser_measles.nigeria import lgas


def setup_meta_population(model, verbose: bool = False) -> None:
    # We need some patches with population data ...
    populations, latitudes, longitudes = initialize_patches(verbose)
    model.patches = LaserFrame(len(populations))
    model.patches.add(len(populations))  # "activate" all the patches (count == capacity)
    model.patches.add_vector_property("populations", length=model.params.nticks + 1)
    model.patches.populations[0, :] = populations  # set patch populations at t = 0 to initial populations

    # ... and connectivity data
    distances = calc_distances(latitudes, longitudes, verbose)
    network = gravity(populations, distances, model.params.k, model.params.a, model.params.b, model.params.c)
    network = row_normalizer(network, model.params.max_frac)
    model.patches.add_vector_property("network", length=model.patches.count, dtype=np.float32)
    model.patches.network[:, :] = network

    return


def initialize_patches(verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    admin2 = {k: v for k, v in lgas.items() if len(k.split(":")) == 5}
    print(f"Processing {len(admin2)} admin2 areas in Nigeria…")
    nn_nodes = {k: v for k, v in admin2.items() if k.split(":")[2].startswith("NORTH_")}
    print(f"Loading population and location data for {len(nn_nodes)} admin2 areas in Northern Nigeria…")
    # Values in nigeria.lgas are tuples: ((population, year), (longitude, latitude), area_km2)
    populations = np.array([v[0][0] for v in nn_nodes.values()])
    print(f"Total initial population: {populations.sum():,}")
    latitudes = np.array([v[1][1] for v in nn_nodes.values()])
    longitudes = np.array([v[1][0] for v in nn_nodes.values()])

    if verbose:
        print(f"Populations: {populations[0:4]}")
        print(f"Lat/longs: {list(zip(latitudes, longitudes))[0:4]}")

    return populations, latitudes, longitudes


def calc_distances(latitudes: np.ndarray, longitudes: np.ndarray, verbose: bool = False) -> np.ndarray:
    assert latitudes.ndim == 1, "Latitude array must be one-dimensional"
    assert longitudes.shape == latitudes.shape, "Latitude and longitude arrays must have the same shape"
    distances = np.zeros((len(latitudes), len(latitudes)), dtype=np.float32)
    for i, (lat, long) in enumerate(zip(latitudes, longitudes)):
        distances[i, :] = distance(lat, long, latitudes, longitudes)

    if verbose:
        print(f"Upper left corner of distance matrix:\n{distances[0:4, 0:4]}")

    return distances
