import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from matplotlib.figure import Figure

from laser_measles.nigeria import lgas
from laser_measles.utils import calc_distances


class MetaPopulation:
    def __init__(self, model, verbose: bool = False):
        self.__name__ = "propagate_population"
        self.model = model

        # We need some patches with population data ...
        names, populations, latitudes, longitudes = initialize_patches(verbose)
        model.patches = LaserFrame(len(populations))
        model.patches.add(len(populations))  # "activate" all the patches (count == capacity)
        model.patches.add_vector_property("populations", length=model.params.nticks + 1)
        model.patches.populations[0, :] = populations  # set patch populations at t = 0 to initial populations

        self._names = names
        self._populations = populations
        self._latitudes = latitudes
        self._longitudes = longitudes

        # ... and connectivity data
        distances = calc_distances(latitudes, longitudes, verbose)
        network = gravity(populations, distances, model.params.k, model.params.a, model.params.b, model.params.c)
        network = row_normalizer(network, model.params.max_frac)
        model.patches.add_vector_property("network", length=model.patches.count, dtype=np.float32)
        model.patches.network[:, :] = network

        return

    @property
    def names(self):
        return self._names

    @property
    def populations(self):
        return self._populations

    @property
    def latitudes(self):
        return self._latitudes

    @property
    def longitudes(self):
        return self._longitudes

    def __call__(self, model, tick):
        model.patches.populations[tick + 1, :] = model.patches.populations[tick, :]
        return

    def plot(self, fig: Figure = None) -> None:
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("[Northern] Nigeria Admin2 Patches and Populations")
        gpdf = gpd.read_file(self.model.params.shape_file)
        ax = plt.gca()
        gpdf.plot(ax=ax)
        scatter = plt.scatter(
            self.longitudes,
            self.latitudes,
            s=self.populations / 10_000,
            c=self.populations,
            cmap="inferno",
        )
        plt.colorbar(scatter, label="Population")

        return


def initialize_patches(verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    admin2 = {k: v for k, v in lgas.items() if len(k.split(":")) == 5}
    print(f"Processing {len(admin2)} admin2 areas in Nigeria…")
    nn_nodes = {k: v for k, v in admin2.items() if k.split(":")[2].startswith("NORTH_")}
    print(f"Loading population and location data for {len(nn_nodes)} admin2 areas in Northern Nigeria…")
    # Values in nigeria.lgas are tuples: ((population, year), (longitude, latitude), area_km2)
    names = np.array([k.split(":")[4] for k in nn_nodes.keys()])
    populations = np.array([v[0][0] for v in nn_nodes.values()])
    print(f"Total initial population: {populations.sum():,}")
    latitudes = np.array([v[1][1] for v in nn_nodes.values()])
    longitudes = np.array([v[1][0] for v in nn_nodes.values()])

    if verbose:
        print(f"Populations: {populations[0:4]}")
        print(f"Lat/longs: {list(zip(latitudes, longitudes))[0:4]}")

    return names, populations, latitudes, longitudes
