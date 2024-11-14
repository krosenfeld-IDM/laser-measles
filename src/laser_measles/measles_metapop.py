import geopandas as gpd
import pandas as pd


def get_scenario(params, verbose: bool = False) -> pd.DataFrame:
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
