from laser_measles.demographics import gadm
from laser_measles.demographics import patch, shapefiles
import polars as pl
import matplotlib.pyplot as plt
import alive_progress
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

population_raster_path = "/home/krosenfeld/code/laser-measles/examples/expanding_kano/gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2010_30_sec_crop.tif"

g = gadm.GADMShapefile("NGA")
# gadm.clear_cache()
# gadm.download()
# gadm.add_dotnames()
config = patch.DemographicConfig(
    region="NGA",
    start_year=2010,
    end_year=2020,
    granularity="patch",
    patch_size_km=25,
    shapefile_path=g.get_shapefile_path(admin_level=2),
    population_raster_path=population_raster_path,
)
generator = patch.PatchDemographicsGenerator(config)
generator.generate_demographics()

# Get dataframe with the patches
df = shapefiles.get_dataframe(generator.shapefile)
df_states = shapefiles.get_dataframe(g.get_shapefile_path(admin_level=1))

# Add "state" column
df = df.with_columns(
    pl.col("dotname")
    .str.split_exact(":", 2)
    .struct.field("field_1")
    .alias("state")
)
df_states = df_states.with_columns(
    pl.col("dotname")
    .str.split_exact(":", 1)
    .struct.field("field_1")
    .alias("state")
)

with alive_progress.alive_bar() as bar:
    bar.text("Plotting Kano")
    # Plot just Kano
    states = ["kano"]
    filtered_df = df.lazy().filter(pl.col("state").is_in(states)).collect()
    fig = shapefiles.plot_dataframe(filtered_df)
    fig.savefig("kano.png", dpi=300, bbox_inches="tight", transparent=True)

    # Plot Kano plus surrounding states
    bar.text("Plotting Kano region")
    states = [
        k.lower()
        for k in ["Kano", "Katsina", "Jigawa", "Kaduna", "Bauchi", "Plateau"]
    ]
    filtered_df = df.lazy().filter(pl.col("state").is_in(states)).collect()
    filtered_df_states = df_states.lazy().filter(pl.col("state").is_in(states)).collect()
    fig = shapefiles.plot_dataframe(filtered_df, plot_kwargs={'linewidth': 0.25})
    shapefiles.plot_dataframe(filtered_df_states, ax=fig.axes[0], plot_kwargs={'linewidth': 1})
    fig.savefig("kano_region.png", dpi=300, bbox_inches="tight", transparent=True)


    # Plot Northern nigeria states
    bar.text("Plotting Northern Nigeria")
    # Borno, Bauchi, Gombe, Taraba, Yobe, Adamawa, Jigawa, Kano, Kaduna, Katsina, Kebbi, Sokoto, Zamfara, Niger, Plateau, Nasarawa, Benue, Kogi, and Kwara
    states = [
        k.lower()
        for k in [
            "Borno",
            "Bauchi",
            "Gombe",
            "Taraba",
            "Yobe",
            "Adamawa",
            "Jigawa",
            "Kano",
            "Kaduna",
            "Katsina",
            "Kebbi",
            "Sokoto",
            "Zamfara",
            "Niger",
            "Plateau",
            "Nasarawa",
            "Benue",
            "Kogi",
            "Kwara",
            "Federal Capital Territory",
        ]
    ]
    filtered_df = df.lazy().filter(pl.col("state").is_in(states)).collect()
    filtered_df_states = df_states.lazy().filter(pl.col("state").is_in(states)).collect()
    fig = shapefiles.plot_dataframe(filtered_df, plot_kwargs={'linewidth': 0.1})
    shapefiles.plot_dataframe(filtered_df_states, ax=fig.axes[0], plot_kwargs={'linewidth': 1, 'fill': True, 'facecolor': 'none'})
    fig.savefig("nigeria_region.png", dpi=300, bbox_inches="tight", transparent=True)