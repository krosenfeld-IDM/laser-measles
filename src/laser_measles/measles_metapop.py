import click
import geopandas as gpd
import numpy as np
import pandas as pd
from laser_core.laserframe import LaserFrame
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class MetaPopulation:
    def __init__(self, model, verbose: bool = False):
        self.__name__ = "propagate_population"

        pops = pd.read_csv(model.params.population_file)
        pops.set_index("county", inplace=True)
        gpdf = gpd.read_file(model.params.shape_file)
        gpdf.drop(
            columns=[
                "EDIT_DATE",
                "EDIT_STATU",
                "EDIT_WHO",
                "GLOBALID",
                "JURISDICT_",
                "JURISDIC_1",
                "JURISDIC_3",
                "JURISDIC_4",
                "JURISDIC_5",
                "JURISDIC_6",
                "OBJECTID",
            ],
            inplace=True,
        )
        gpdf.rename(columns={"JURISDIC_2": "county"}, inplace=True)
        gpdf.set_index("county", inplace=True)

        gpdf = gpdf.join(pops)
        centroids = gpdf.centroid.to_crs(epsg=4326)  # convert from meters to degrees
        gpdf["latitude"] = centroids.y
        gpdf["longitude"] = centroids.x
        gpdf.to_crs(epsg=4326, inplace=True)

        click.echo(f"Using {len(gpdf)} counties in Washington…")

        if verbose:
            click.echo(f"Counties: {gpdf.index.values[0:4]}...")
            click.echo(f"Populations: {gpdf.population.values[0:4]}...")
            click.echo(f"Lat/longs: {list(zip(gpdf.latitude.values, gpdf.longitude.values))[0:4]}...")

        self._gpdf = gpdf
        self.count = len(gpdf)

        npatches = self.count
        model.patches = LaserFrame(npatches)

        # "activate" all the patches (count == capacity)
        model.patches.add(npatches)
        model.patches.add_vector_property("populations", length=model.params.nticks + 1)
        # set patch populations at t = 0 to initial populations
        model.patches.populations[0, :] = gpdf.population

        # ... and connectivity data
        distances = calc_distances(gpdf.latitude.values, gpdf.longitude.values, model.params.verbose)
        network = gravity(
            gpdf.population.values,
            distances,
            model.params.k,
            model.params.a,
            model.params.b,
            model.params.c,
        )
        network = row_normalizer(network, model.params.max_frac)
        model.patches.add_vector_property("network", length=npatches, dtype=np.float32)
        model.patches.network[:, :] = network

        return

    @property
    def names(self):
        return self._names.to_numpy(copy=False)  # copy=False is just a suggestion

    @property
    def populations(self):
        return self._populations.to_numpy(copy=False)  # copy=False is just a suggestion

    @property
    def latitudes(self):
        return self._latitudes.to_numpy(copy=False)  # copy=False is just a suggestion

    @property
    def longitudes(self):
        return self._longitudes.to_numpy(copy=False)  # copy=False is just a suggestion

    def __call__(self, model, tick):
        model.patches.populations[tick + 1, :] = model.patches.populations[tick, :]
        return

    def plot(self, fig: Figure = None) -> None:
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Washington State Counties and Populations")
        ax = plt.gca()
        self._gpdf.plot(ax=ax)
        scatter = plt.scatter(
            self._gpdf.longitude,
            self._gpdf.latitude,
            s=self._gpdf.population / 1000,
            c=self._gpdf.population,
            cmap="inferno",
        )
        plt.colorbar(scatter, label="Population")

        return


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
