from pathlib import Path
from typing import Optional

import polars as pl
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from rastertoolkit import shape_subdivide
from rastertoolkit import raster_clip

from laser_measles.demographics.base import DemographicsGenerator
from laser_measles.demographics.gadm import GADMShapefile
from laser_measles.demographics import shapefiles
from laser_measles.demographics import cache


class DemographicConfig(BaseModel):
    region: str = Field(..., description="Country identifier (ISO3 code)")
    start_year: int = Field(..., description="Start year for the demographic data")
    end_year: int = Field(..., description="End year for the demographic data")
    granularity: str = Field("admin0", description="admin0 | admin1 | admin2 | patch")
    patch_size_km: Optional[int] = None
    shapefile_path: str | Path = Field(..., description="Path to the shapefile")
    population_raster_path: str | Path = Field(
        ..., description="Path to the population raster"
    )

    @field_validator("end_year")
    def check_years(cls, v, info):
        if info.data.get("start_year") is not None and v < info.data["start_year"]:
            raise ValueError("end_year must be >= start_year")
        return v

    @field_validator("patch_size_km")
    def check_patch_size(cls, v, info):
        if info.data.get("granularity") == "patch" and v is None:
            raise ValueError("patch_size_km must be set when granularity='patch'")
        return v

    @field_validator("shapefile_path")
    def require_shapefile_for_non_national(cls, v, info):
        if info.data.get("granularity") in ("admin1", "admin2", "patch") and not v:
            raise ValueError(
                "shapefile_path is required for admin1, admin2, or patch granularity"
            )
        return v

    @field_validator("shapefile_path")
    def shapefile_path_exists(cls, v, info):
        path = Path(v) if isinstance(v, str) else v
        if not path.exists():
            raise ValueError(f"Shapefile path does not exist: {path}")
        return v


class PatchDemographicsGenerator(DemographicsGenerator):
    def __init__(self, config: DemographicConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose

        self._validate_config()

    def generate_demographics(self) -> None:
        self._validate_shapefile()
        self.generate_population()
        # self.generate_birth_rates()
        # self.generate_mortality_rates()

    def _validate_config(self) -> None:
        if not shapefiles.check_dotname(self.config.shapefile_path):
            raise ValueError(f"Shapefile {self.config.shapefile_path} does not have a DOTNAME field")
        
    def _validate_shapefile(self):
        """
        Generate the patches for the shapefile.
        """
        if self.config.granularity == "patch":
            patch_path = self.config.shapefile_path.with_name(f"{self.config.shapefile_path.stem}_{self.config.patch_size_km}km.shp")
            if not patch_path.exists():
                shape_subdivide(
                    shape_stem=self.config.shapefile_path,
                    out_dir=self.config.shapefile_path.parent,
                    out_suffix=f"{self.config.patch_size_km}km",
                    box_target_area_km2=self.config.patch_size_km,
                    verbose=self.verbose,
                )
            elif self.verbose:
                print(f"Patch shapefile {patch_path} already exists")
            self.shapefile = patch_path
        else:
            self.shapefile = self.config.shapefile_path

    def get_cache_key(self, key) -> str:
        keys = ['shapefile']
        if key not in keys:
            raise ValueError(f"Invalid key: {key}\nValid keys are: {keys}")
        return f"{self.config.region}_{self.config.granularity}"

    def generate_population(self) -> pl.DataFrame:

        # clip the raster to the shapefile
        popdict = raster_clip(self.config.population_raster_path, 
                            self.shape_file, include_latlon=True)
        
    def generate_birth_rates(self) -> pl.DataFrame: ...

    def generate_mortality_rates(self) -> pl.DataFrame: ...

    def _add_dotname(self) -> None:
        ...
        # sf = Reader(shapefile)


if __name__ == "__main__":
    gadm = GADMShapefile("NGA")
    gadm.clear_cache()
    gadm.download()
    gadm.add_dotnames()
    config = DemographicConfig(
        region="NGA",
        start_year=2000,
        end_year=2020,
        granularity="patch",
        patch_size_km=25,
        shapefile_path=gadm.get_shapefile_path(2),
        population_raster_path=gadm.shapefile_dir,
    )
    generator = PatchDemographicsGenerator(config)
    generator.generate_demographics()
    # print(generator.generate_population())
