import click
import geopandas as gpd

# import numpy as np
import pandas as pd

# from laser_measles.nigeria import lgas


def get_scenario(params, verbose: bool = False) -> pd.DataFrame:
    # We need some patches with population data ...
    # names, populations, latitudes, longitudes = initialize_patches(verbose)
    if verbose:
        click.echo(f"Loading population and location data from '{params.shape_file}'…")
    gpdf = gpd.read_file(params.shape_file)
    if verbose:
        click.echo(f"Loaded {len(gpdf):,} patches (total population {gpdf.population.sum():,}) from '{params.shape_file}'.")

    return gpdf


# deprecated
# def initialize_patches(verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     admin2 = {k: v for k, v in lgas.items() if len(k.split(":")) == 5}
#     print(f"Processing {len(admin2)} admin2 areas in Nigeria…")
#     nn_nodes = {k: v for k, v in admin2.items() if k.split(":")[2].startswith("NORTH_")}
#     print(f"Loading population and location data for {len(nn_nodes)} admin2 areas in Northern Nigeria…")
#     # Values in nigeria.lgas are tuples: ((population, year), (longitude, latitude), area_km2)
#     names = np.array([k.split(":")[4] for k in nn_nodes.keys()])
#     populations = np.array([v[0][0] for v in nn_nodes.values()])
#     print(f"Total initial population: {populations.sum():,}")
#     latitudes = np.array([v[1][1] for v in nn_nodes.values()])
#     longitudes = np.array([v[1][0] for v in nn_nodes.values()])

#     if verbose:
#         print(f"Populations: {populations[0:4]}")
#         print(f"Lat/longs: {list(zip(latitudes, longitudes))[0:4]}")

#     return names, populations, latitudes, longitudes
