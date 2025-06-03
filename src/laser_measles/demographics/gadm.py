"""
GADM shapefiles
"""

import io
import zipfile
from pathlib import Path

import alive_progress
import polars as pl
import pycountry
import requests
from shapefile import Reader
from rastertoolkit import shape_subdivide
from rastertoolkit import raster_clip

from laser_measles.demographics.base import Shapefile
from laser_measles.demographics import cache
from laser_measles.demographics import shapefiles

VERSION = "4.1"
VERSION_INT = VERSION.replace(".", "")
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm{VERSION}/shp/gadm{VERSION_INT}_{COUNTRY_CODE}_shp.zip"
GADM_SHP_FILE = "gadm{VERSION_INT}_{COUNTRY_CODE}_{LEVEL}.shp"


class GADMShapefile(Shapefile):
    country_code: str
    shapefile_dir: str | Path
    admin_levels: list[int] = set([0, 1, 2])
    key_dict: dict[str, str] = {} # Cache key -> description

    def __init__(self, country_code: str):
        self.country_code = country_code
        self.shapefile_dir = self.get_cache_dir() / f"gadm{VERSION_INT}_{self.country_code.upper()}_shp"

    def download(self, timeout: int = 60) -> str:
        """
        Download the GADM shapefile for a given country code and cache it.

        Args:
            country_code: The country code to download the shapefile for.
            timeout: The timeout for the request.

        Returns:
            The path to the downloaded shapefile.
        """
        # Check country_code for correctness
        country = pycountry.countries.get(alpha_3=self.country_code.upper())
        if not country:
            raise ValueError(f"Invalid country code: {self.country_code}")
        with cache.load_cache() as c:
            download_path = Path(c.directory) / f"{self.country_code.upper()}" / f"gadm41_{self.country_code}_shp"
            cache_key = self.get_cache_key() + ":root"
            if cache_key not in c:
                url = GADM_URL.format(VERSION=VERSION, VERSION_INT=VERSION_INT, COUNTRY_CODE=self.country_code.upper())
                with alive_progress.alive_bar(
                    title=f"Downloading GADM shapefile for {country.name}",
                ) as bar:
                    response = requests.get(url, timeout=timeout)
                    response.raise_for_status()
                    bar.text = f"Extracting GADM shapefile for {country.name}"
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                        zip_file.extractall(path=download_path)
                    bar.text = f"GADM shapefile for {country.name} downloaded and extracted"
                c[cache_key] = download_path
                self.key_dict[cache_key] = "Directory containing the shapefiles"
            return c[cache_key]
    
    def add_dotnames(self) -> None:
        """Add a DOTNAME to the GADM shapefile for all admin levels."""
        for admin_level in self.admin_levels:
            self.add_dotname(admin_level)
    
    def add_dotname(self, admin_level: int) -> None:
        """Add a DOTNAME to the GADM shapefile for a given admin level."""
        if admin_level == 0:
            dotname_fields = ["COUNTRY"]
        elif admin_level == 1:
            dotname_fields = ["COUNTRY", "NAME_1"]
        elif admin_level == 2:
            dotname_fields = ["COUNTRY", "NAME_1", "NAME_2"]    
        shapefiles.add_dotname(self.get_shapefile_path(admin_level), dot_name_fields = dotname_fields, inplace=True)

    def shape_subdivide(self, admin_level: int, patch_size_km: int, ) -> None:
        """Subdivide the GADM shapefile for a given admin level into patches of a given size."""
        with alive_progress.alive_bar(
            title=f"Subdividing GADM shapefile for {self.country_code} at admin level {admin_level}",
        ) as bar:
            # shapefiles.shape_subdivide(
            #     shape_stem=self.get_shapefile_path(admin_level),
            #     out_dir=self.shapefile_dir,
            #     out_suffix=f"{patch_size_km}km",
            # )
            with cache.load_cache() as c:
                cache_key = self.get_cache_key() + ":root"
                if cache_key in c:
                    shapefile_path = c[cache_key]
                else:
                    raise FileNotFoundError(f"Shapefiles not found")
            shape_subdivide(
                shape_stem=shapefile_path / f"gadm{VERSION_INT}_{self.country_code}_{admin_level}.shp",
                out_dir=self.shapefile_dir,
                out_suffix=f"{patch_size_km}km",
                box_target_area_km2=patch_size_km
            )            
            with cache.load_cache() as c:
                cache_key = self.get_cache_key() + f":{admin_level}:{patch_size_km}km"
                c[cache_key] = self.shapefile_dir / f"gadm{VERSION_INT}_{self.country_code}_{admin_level}_{patch_size_km}km.shp"
                self.key_dict[cache_key] = f"Shapefile for {self.country_code} at admin level {admin_level} with patch size {patch_size_km}km"

    def get_shapefile_path(self, admin_level: int) -> str:
        """Get the path to the GADM shapefile for a given admin level."""
        if admin_level not in self.admin_levels:
            raise ValueError(f"Admin level {admin_level} not found in {self.admin_levels}")
        return self.shapefile_dir / f"gadm{VERSION_INT}_{self.country_code}_{admin_level}.shp"

    def get_cache_dir(self) -> str:
        """Get the directory for the GADM shapefile in the cache."""
        return Path(cache.get_cache_dir()) / f"{self.country_code.upper()}"

    def clear_cache(self) -> None:
        """Remove the GADM shapefile from the cache."""
        with cache.load_cache() as c:
            cache_key = self.get_cache_key()
            for k in c.iterkeys():
                if k.startswith(cache_key):
                    f = c.pop(k)
                    if f.exists() and not f.is_dir():
                        f.unlink()
        # If the cache key is not in the cache, remove the directory
        gadm_cache_dir = self.get_cache_dir() / f"gadm{VERSION_INT}_{self.country_code}_shp"
        cache.clear_cache_dir(gadm_cache_dir)

    def list_cache_keys(self) -> list[str]:
        """List the cache keys for the GADM shapefile."""
        with cache.load_cache() as c:
            cache_key = self.get_cache_key()
            return [k for k in c.iterkeys() if k.startswith(cache_key)]

    def get_cache_key(self) -> str:
        """Get the key for the GADM shapefile in the cache. Related entries will start with this key (e.g.: gadm41_NGA_shp:root)
        """
        return f"gadm{VERSION_INT}_{self.country_code}_shp"

    def get_dataframe(self, admin_level: int) -> pl.DataFrame:
        """
        Get a Polars DataFrame containing the shapefile data with DOTNAME and shape columns.
        
        Args:
            admin_level: The administrative level to get data for.
            
        Returns:
            A Polars DataFrame with DOTNAME and shape columns.
        """
        shapefile_path = self.get_shapefile_path(admin_level)
        if not shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found at {shapefile_path}")
            
        with Reader(shapefile_path) as sf:
            # Get all records and shapes
            records = []
            shapes = []
            for shaperec in sf.iterShapeRecords():
                records.append(shaperec.record)
                shapes.append(shaperec.shape)
            
            # Convert to DataFrame
            df = pl.DataFrame(records)
            
            # Add shape column
            df = df.with_columns(pl.Series(name="shape", values=shapes))
            
            return df


if __name__ == "__main__":
    gadm = GADMShapefile("CUB")
    df = gadm.get_dataframe(admin_level=2)
    # gadm.download()
