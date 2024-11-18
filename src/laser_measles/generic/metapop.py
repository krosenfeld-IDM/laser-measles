"""
This module provides functionality to generate a scenario DataFrame by combining population data with geographical data.

Functions:

    get_scenario(params, verbose: bool = False) -> pd.DataFrame:

        Reads population data from a CSV file and geographical data from a shapefile,
        processes and merges them, and returns a GeoDataFrame with additional latitude
        and longitude columns.

        Parameters:

            params: An object containing the file paths for the population and shape files.
            verbose (bool): If True, enables verbose output. Default is False.

        Returns:

            pd.DataFrame: A GeoDataFrame containing the merged population and geographical data.
"""

import geopandas as gpd
import pandas as pd


def get_scenario(params, verbose: bool = False) -> pd.DataFrame:
    """
    Generates a scenario DataFrame by merging population data with geographical shape data.

    Args:

        params: An object containing the following attributes:

            - population_file (str): Path to the CSV file containing population data.
            - shape_file (str): Path to the shapefile containing geographical data.

        verbose (bool, optional): If True, enables verbose output. Defaults to False.

    Returns:

        pd.DataFrame: A DataFrame containing the merged population and geographical data with additional latitude and longitude columns.
    """

    pops = pd.read_csv(params.population_file)
    pops.rename(columns={"county": "name"}, inplace=True)
    pops.set_index("name", inplace=True)
    gpdf = gpd.read_file(params.shape_file)
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
    gpdf.rename(columns={"JURISDIC_2": "name"}, inplace=True)
    gpdf.set_index("name", inplace=True)

    gpdf = gpdf.join(pops)
    centroids = gpdf.centroid.to_crs(epsg=4326)  # convert from meters to degrees
    gpdf["latitude"] = centroids.y
    gpdf["longitude"] = centroids.x
    gpdf.to_crs(epsg=4326, inplace=True)
    gpdf.reset_index(inplace=True)  # return "name" to just a column

    return gpdf
