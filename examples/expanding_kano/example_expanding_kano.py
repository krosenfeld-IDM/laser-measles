import os
from pathlib import Path

import alive_progress
import polars as pl

from laser_measles.demographics import cache
from laser_measles.demographics import gadm
from laser_measles.demographics import raster_patch
from laser_measles.demographics import shapefiles


def summarize_scenario(df: pl.DataFrame) -> None:
    """Summarize the scenario."""
    # Get the total population
    total_population = df["population"].sum()
    num_with_pop = (df["population"] > 0).sum()
    print(f"Total population: {total_population}")
    print(f"Number of patches: {len(df)} ({num_with_pop} with pop > 0)")

# Change directory
os.chdir(Path.resolve(Path(__file__)).parent.as_posix())
this_dir = Path(__file__).parent

# Set rasters
population_raster_path = "gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2010_30_sec_crop.tif"
mcv1_raster_path = "mcv1_cov_mean_raked_2000_2023_11_crop.tif"

# Prep shapefile
g = gadm.GADMShapefile("NGA")
# g.download()
# g.add_dotnames()
# g.shape_subdivide(admin_level=2, patch_size_km=50)
with cache.load_cache() as c:
    shapefile_path = c[g.get_cache_key() + ":2:50km"]

# Setup demographics generator
config = raster_patch.RasterPatchConfig(
    id='nigeria_50km',
    region="NGA",
    shapefile_path=shapefile_path,
    population_raster_path=this_dir / population_raster_path,
    mcv1_raster_path=this_dir / mcv1_raster_path,
)
generator = raster_patch.RasterPatchGenerator(config)
generator.generate_demographics()

# Grab the data
df_pop = generator.population # TODO: save everythign in a dataframe
df_mcv1 = generator.mcv1

# Get dataframe with the patches
df = shapefiles.get_dataframe(generator.shapefile).join(df_pop, on="dotname", how="left")
df = df.join(df_mcv1.drop(["lat", "lon"]), on="dotname", how="left")
assert(len(df) == len(df_pop))
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
    filtered_df_kano = df.lazy().filter(pl.col("state").is_in(states)).collect()
    filtered_df_kano = filtered_df_kano.with_columns(pl.lit("kano").alias("scenario_id"))
    fig = shapefiles.plot_dataframe(filtered_df_kano)
    fig.savefig("kano.png", dpi=300, bbox_inches="tight", transparent=True)
    summarize_scenario(filtered_df_kano)
    # Plot Kano plus surrounding states
    bar.text("Plotting Kano region")
    states = [
        k.lower()
        for k in ["Kano", "Katsina", "Jigawa", "Kaduna", "Bauchi", "Plateau"]
    ]
    filtered_df_kano_region = df.lazy().filter(pl.col("state").is_in(states)).collect()
    filtered_df_kano_region = filtered_df_kano_region.with_columns(pl.lit("kano_region").alias("scenario_id"))
    filtered_df_states = df_states.lazy().filter(pl.col("state").is_in(states)).collect()
    fig = shapefiles.plot_dataframe(filtered_df_kano_region, plot_kwargs={'linewidth': 0.25})
    shapefiles.plot_dataframe(filtered_df_states, ax=fig.axes[0], plot_kwargs={'linewidth': 1})
    fig.savefig("kano_region.png", dpi=300, bbox_inches="tight", transparent=True)
    summarize_scenario(filtered_df_kano_region)

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
    filtered_df_northern = df.lazy().filter(pl.col("state").is_in(states)).collect()
    filtered_df_northern = filtered_df_northern.with_columns(pl.lit("northern_nigeria").alias("scenario_id"))
    filtered_df_states = df_states.lazy().filter(pl.col("state").is_in(states)).collect()
    fig = shapefiles.plot_dataframe(filtered_df_northern, plot_kwargs={'linewidth': 0.1})
    shapefiles.plot_dataframe(filtered_df_states, ax=fig.axes[0], plot_kwargs={'linewidth': 1, 'fill': True, 'facecolor': 'none'})
    fig.savefig("nigeria_region.png", dpi=300, bbox_inches="tight", transparent=True)
    summarize_scenario(filtered_df_northern)

    # Combine all scenarios and save to disk
    combined_df = pl.concat([filtered_df_kano, filtered_df_kano_region, filtered_df_northern])
    combined_df.drop("shape").write_parquet("scenarios.parquet")
