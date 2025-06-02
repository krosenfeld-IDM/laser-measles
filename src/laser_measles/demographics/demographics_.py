import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple, Any, Type, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, date
from enum import Enum
import json

from pydantic import BaseModel, Field, validator, root_validator, ValidationError as PydanticValidationError
from pydantic.types import PositiveFloat, PositiveInt, confloat, conint
import polars as pl
import fiona
from shapely.geometry import shape, Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import diskcache
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = ".demogen_cache"
CACHE_SIZE_LIMIT = 1024**3  # 1GB


class CacheManager:
    """Centralized cache management with size limits and cleanup."""

    def __init__(self, cache_dir: str = CACHE_DIR, size_limit: int = CACHE_SIZE_LIMIT):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = diskcache.Cache(str(self.cache_dir), size_limit=size_limit)

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        return self.cache.set(key, value, expire=expire)

    def clear_expired(self) -> int:
        """Clear expired cache entries."""
        return self.cache.cull()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "volume": self.cache.volume(),
            "hits": getattr(self.cache, "stats", {}).get("hits", 0),
            "misses": getattr(self.cache, "stats", {}).get("misses", 0),
        }

    @contextmanager
    def transaction(self):
        """Context manager for cache transactions."""
        with self.cache.transact():
            yield


# Global cache instance
cache_manager = CacheManager()

# Type variable for generic data models
T = TypeVar("T", bound="DemographicDataModel")


class AgeGroup(str, Enum):
    """Standardized age group categories."""

    AGE_0_4 = "0-4"
    AGE_5_9 = "5-9"
    AGE_10_14 = "10-14"
    AGE_15_19 = "15-19"
    AGE_20_24 = "20-24"
    AGE_25_29 = "25-29"
    AGE_30_34 = "30-34"
    AGE_35_39 = "35-39"
    AGE_40_44 = "40-44"
    AGE_45_49 = "45-49"
    AGE_50_54 = "50-54"
    AGE_55_59 = "55-59"
    AGE_60_64 = "60-64"
    AGE_65_69 = "65-69"
    AGE_70_74 = "70-74"
    AGE_75_79 = "75-79"
    AGE_80_PLUS = "80+"


class DemographicDataModel(BaseModel):
    """Base class for all demographic data models."""

    class Config:
        # Allow arbitrary types for compatibility with Polars
        arbitrary_types_allowed = True
        # Use enum values in JSON
        use_enum_values = True
        # Validate assignment
        validate_assignment = True

    @classmethod
    def get_polars_schema(cls) -> Dict[str, Any]:
        """Get Polars schema from Pydantic model."""
        schema = {}
        for field_name, field_info in cls.__fields__.items():
            field_type = field_info.type_

            # Map Pydantic types to Polars types
            if field_type == int or field_type == PositiveInt:
                schema[field_name] = pl.Int64
            elif field_type == float or field_type == PositiveFloat:
                schema[field_name] = pl.Float64
            elif field_type == str:
                schema[field_name] = pl.Utf8
            elif field_type == date:
                schema[field_name] = pl.Date
            elif field_type == datetime:
                schema[field_name] = pl.Datetime
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                # Handle Optional types
                non_none_types = [t for t in field_type.__args__ if t != type(None)]
                if non_none_types:
                    first_type = non_none_types[0]
                    if first_type == int:
                        schema[field_name] = pl.Int64
                    elif first_type == float:
                        schema[field_name] = pl.Float64
                    else:
                        schema[field_name] = pl.Utf8
            else:
                schema[field_name] = pl.Utf8  # Default to string

        return schema

    @classmethod
    def validate_dataframe(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Validate a Polars DataFrame against the model schema."""
        required_columns = set(cls.__fields__.keys())
        df_columns = set(df.columns)

        # Check for missing columns
        missing_columns = required_columns - df_columns
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")

        # Validate each row (sample validation for performance)
        if len(df) > 0:
            # Validate first few rows as samples
            sample_size = min(100, len(df))
            sample_df = df.head(sample_size)

            for row in sample_df.iter_rows(named=True):
                try:
                    cls(**row)
                except PydanticValidationError as e:
                    raise ValidationError(f"Data validation failed: {e}")

        return df

    @classmethod
    def from_dataframe(cls: Type[T], df: pl.DataFrame) -> List[T]:
        """Create list of model instances from DataFrame."""
        validated_df = cls.validate_dataframe(df)
        return [cls(**row) for row in validated_df.iter_rows(named=True)]

    @classmethod
    def to_dataframe(cls, instances: List[T]) -> pl.DataFrame:
        """Convert list of model instances to DataFrame."""
        if not instances:
            # Return empty DataFrame with correct schema
            schema = cls.get_polars_schema()
            return pl.DataFrame(schema=schema)

        data = [instance.dict() for instance in instances]
        return pl.DataFrame(data)


class PopulationRecord(DemographicDataModel):
    """Standardized population data record."""

    year: conint(ge=1900, le=2200) = Field(..., description="Calendar year")
    region_id: str = Field(..., description="Unique region identifier", min_length=1)
    age_group: AgeGroup = Field(..., description="Standardized age group")
    population: PositiveInt = Field(..., description="Population count")

    # Optional demographic breakdowns
    male_population: Optional[PositiveInt] = Field(None, description="Male population count")
    female_population: Optional[PositiveInt] = Field(None, description="Female population count")

    # Metadata
    data_source: Optional[str] = Field(None, description="Source of population data")
    confidence_level: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Data confidence (0-1)")
    is_projected: bool = Field(False, description="Whether data is projected vs observed")

    @validator("male_population", "female_population")
    def validate_gender_split(cls, v, values):
        """Validate that gender populations don't exceed total."""
        if v is not None and "population" in values:
            if v > values["population"]:
                raise ValueError("Gender-specific population cannot exceed total population")
        return v

    @root_validator
    def validate_population_consistency(cls, values):
        """Validate consistency between total and gender-specific populations."""
        total = values.get("population")
        male = values.get("male_population")
        female = values.get("female_population")

        if total and male and female:
            if male + female != total:
                raise ValueError("Male + female population must equal total population")

        return values


class BirthRateRecord(DemographicDataModel):
    """Standardized birth rate data record."""

    year: conint(ge=1900, le=2200) = Field(..., description="Calendar year")
    region_id: str = Field(..., description="Unique region identifier", min_length=1)
    birth_rate: confloat(ge=0.0, le=1.0) = Field(..., description="Births per person per year")

    # Optional age-specific rates
    age_group: Optional[AgeGroup] = Field(None, description="Age group for age-specific rates")

    # Additional demographic measures
    total_fertility_rate: Optional[confloat(ge=0.0, le=15.0)] = Field(None, description="Total fertility rate")
    crude_birth_rate: Optional[confloat(ge=0.0, le=100.0)] = Field(None, description="Births per 1000 population")

    # Metadata
    data_source: Optional[str] = Field(None, description="Source of birth rate data")
    confidence_level: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Data confidence (0-1)")
    is_projected: bool = Field(False, description="Whether data is projected vs observed")


class MortalityRateRecord(DemographicDataModel):
    """Standardized mortality rate data record."""

    year: conint(ge=1900, le=2200) = Field(..., description="Calendar year")
    region_id: str = Field(..., description="Unique region identifier", min_length=1)
    age_group: AgeGroup = Field(..., description="Age group")
    mortality_rate: confloat(ge=0.0, le=1.0) = Field(..., description="Deaths per person per year")

    # Optional gender-specific rates
    male_mortality_rate: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Male-specific mortality rate")
    female_mortality_rate: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Female-specific mortality rate")

    # Additional mortality measures
    life_expectancy: Optional[confloat(ge=0.0, le=150.0)] = Field(None, description="Life expectancy at age group")
    infant_mortality_rate: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Infant mortality rate (age 0-1)")

    # Metadata
    cause_of_death: Optional[str] = Field(None, description="Primary cause of death category")
    data_source: Optional[str] = Field(None, description="Source of mortality data")
    confidence_level: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Data confidence (0-1)")
    is_projected: bool = Field(False, description="Whether data is projected vs observed")


class RegionRecord(DemographicDataModel):
    """Standardized region metadata record."""

    region_id: str = Field(..., description="Unique region identifier", min_length=1)
    name: str = Field(..., description="Human-readable region name", min_length=1)
    region_type: str = Field(..., description="Type of region (country, state, county, etc.)")

    # Geographic information
    area_km2: Optional[PositiveFloat] = Field(None, description="Area in square kilometers")
    centroid_longitude: Optional[confloat(ge=-180.0, le=180.0)] = Field(None, description="Centroid longitude")
    centroid_latitude: Optional[confloat(ge=-90.0, le=90.0)] = Field(None, description="Centroid latitude")

    # Hierarchical relationships
    parent_region_id: Optional[str] = Field(None, description="Parent region identifier")
    country_code: Optional[str] = Field(None, description="ISO country code", min_length=2, max_length=3)

    # Administrative information
    administrative_level: Optional[PositiveInt] = Field(None, description="Administrative level (1=country, 2=state, etc.)")
    population_density_category: Optional[str] = Field(None, description="Population density classification")

    # Metadata
    data_quality_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Data quality assessment")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class DemographicDataContainer(BaseModel):
    """Container for all demographic data with metadata."""

    population_data: Optional[pl.DataFrame] = Field(None, description="Population data")
    birth_rate_data: Optional[pl.DataFrame] = Field(None, description="Birth rate data")
    mortality_rate_data: Optional[pl.DataFrame] = Field(None, description="Mortality rate data")
    region_data: Optional[pl.DataFrame] = Field(None, description="Region metadata")

    # Generation metadata
    config: DemographicConfig = Field(..., description="Configuration used to generate data")
    generation_timestamp: datetime = Field(default_factory=datetime.now, description="When data was generated")
    data_version: str = Field("1.0", description="Data format version")

    class Config:
        arbitrary_types_allowed = True

    def validate_all_data(self) -> "DemographicDataContainer":
        """Validate all contained DataFrames against their schemas."""
        if self.population_data is not None:
            self.population_data = PopulationRecord.validate_dataframe(self.population_data)

        if self.birth_rate_data is not None:
            self.birth_rate_data = BirthRateRecord.validate_dataframe(self.birth_rate_data)

        if self.mortality_rate_data is not None:
            self.mortality_rate_data = MortalityRateRecord.validate_dataframe(self.mortality_rate_data)

        if self.region_data is not None:
            self.region_data = RegionRecord.validate_dataframe(self.region_data)

        return self

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all data."""
        stats = {}

        if self.population_data is not None:
            stats["population"] = {
                "total_records": len(self.population_data),
                "years_covered": self.population_data["year"].n_unique(),
                "regions_covered": self.population_data["region_id"].n_unique(),
                "total_population": self.population_data["population"].sum(),
            }

        if self.birth_rate_data is not None:
            stats["birth_rates"] = {
                "total_records": len(self.birth_rate_data),
                "avg_birth_rate": self.birth_rate_data["birth_rate"].mean(),
                "min_birth_rate": self.birth_rate_data["birth_rate"].min(),
                "max_birth_rate": self.birth_rate_data["birth_rate"].max(),
            }

        if self.mortality_rate_data is not None:
            stats["mortality_rates"] = {
                "total_records": len(self.mortality_rate_data),
                "avg_mortality_rate": self.mortality_rate_data["mortality_rate"].mean(),
                "min_mortality_rate": self.mortality_rate_data["mortality_rate"].min(),
                "max_mortality_rate": self.mortality_rate_data["mortality_rate"].max(),
            }

        if self.region_data is not None:
            stats["regions"] = {
                "total_regions": len(self.region_data),
                "total_area_km2": self.region_data["area_km2"].sum() if "area_km2" in self.region_data.columns else None,
            }

        return stats


def file_hash(path: Union[str, Path]) -> str:
    """Hash a file by path, size, and modification time for better change detection."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    stat = path.stat()
    hash_input = f"{path}-{stat.st_size}-{stat.st_mtime}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class DataGenerationError(Exception):
    """Custom exception for data generation errors."""

    pass


@dataclass
class RegionInfo:
    """Data class for region information."""

    region_id: str
    name: str
    properties: Dict[str, Any]
    geometry: Optional[Any] = None


class DemographicConfig(BaseModel):
    """Enhanced configuration with better validation and defaults."""

    region: str = Field(..., description="Country or region identifier (ISO code or name)")
    start_year: int = Field(..., ge=1900, le=2100, description="Start year for demographic data")
    end_year: int = Field(..., ge=1900, le=2100, description="End year for demographic data")
    granularity: str = Field("national", description="Analysis granularity", regex="^(national|subnational|patch)$")
    patch_size_km: Optional[float] = Field(None, gt=0, description="Patch size in kilometers")
    shapefile_path: Optional[str] = Field(None, description="Path to shapefile for spatial analysis")
    age_groups: List[str] = Field(
        default=[
            "0-4",
            "5-9",
            "10-14",
            "15-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80+",
        ],
        description="Age group categories",
    )
    cache_ttl: Optional[int] = Field(None, description="Cache time-to-live in seconds")
    parallel_processing: bool = Field(True, description="Enable parallel processing where applicable")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducible data generation")

    @root_validator
    def validate_config(cls, values):
        """Comprehensive configuration validation."""
        start_year = values.get("start_year")
        end_year = values.get("end_year")
        granularity = values.get("granularity")
        patch_size_km = values.get("patch_size_km")
        shapefile_path = values.get("shapefile_path")

        # Year validation
        if start_year and end_year and end_year < start_year:
            raise ValueError("end_year must be >= start_year")

        # Patch size validation
        if granularity == "patch" and patch_size_km is None:
            raise ValueError("patch_size_km must be set when granularity='patch'")

        # Shapefile validation
        if granularity in ("subnational", "patch"):
            if not shapefile_path:
                raise ValueError("shapefile_path is required for subnational or patch granularity")
            if not Path(shapefile_path).exists():
                raise ValueError(f"Shapefile not found: {shapefile_path}")

        return values

    def to_cache_key(self) -> str:
        """Generate a unique cache key for this configuration."""
        config_dict = self.dict(exclude={"cache_ttl"})
        return hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:16]


class DataGenerator(ABC):
    """Abstract base class for demographic data generators."""

    @abstractmethod
    def generate(self, config: DemographicConfig, regions: List[RegionInfo]) -> pl.DataFrame:
        """Generate demographic data for given configuration and regions."""
        pass


class PopulationGenerator(DataGenerator):
    """Generate population data with realistic demographic patterns."""

    def generate(self, config: DemographicConfig, regions: List[RegionInfo]) -> pl.DataFrame:
        """Generate population data with age structure and regional variation."""
        if config.random_seed:
            np.random.seed(config.random_seed)

        records = []
        years = range(config.start_year, config.end_year + 1)

        for region in regions:
            base_population = self._estimate_base_population(region)

            for year in years:
                year_multiplier = 1 + (year - config.start_year) * 0.01  # 1% annual growth

                for age_group_str in config.age_groups:
                    try:
                        age_group = AgeGroup(age_group_str)
                    except ValueError:
                        logger.warning(f"Invalid age group: {age_group_str}, skipping")
                        continue

                    age_weight = self._get_age_weight(age_group)
                    population = int(base_population * age_weight * year_multiplier)
                    population += np.random.randint(-population // 20, population // 20)  # ±5% variation
                    population = max(0, population)

                    # Create structured record
                    record = PopulationRecord(
                        year=year,
                        region_id=region.region_id,
                        age_group=age_group,
                        population=population,
                        is_projected=True,
                        confidence_level=0.8,
                        data_source="Generated",
                    )
                    records.append(record.dict())

        # Convert to DataFrame and validate
        df = pl.DataFrame(records)
        return PopulationRecord.validate_dataframe(df)

    def _estimate_base_population(self, region: RegionInfo) -> int:
        """Estimate base population based on region characteristics."""
        # Simple heuristic - in practice, this would use real demographic data
        area_estimate = getattr(region.geometry, "area", 1.0) if region.geometry else 1.0
        return max(1000, int(area_estimate * 1000))

    def _get_age_weight(self, age_group: AgeGroup) -> float:
        """Get demographic weight for age group (simplified population pyramid)."""
        weights = {
            AgeGroup.AGE_0_4: 0.06,
            AgeGroup.AGE_5_9: 0.06,
            AgeGroup.AGE_10_14: 0.06,
            AgeGroup.AGE_15_19: 0.06,
            AgeGroup.AGE_20_24: 0.07,
            AgeGroup.AGE_25_29: 0.08,
            AgeGroup.AGE_30_34: 0.08,
            AgeGroup.AGE_35_39: 0.08,
            AgeGroup.AGE_40_44: 0.07,
            AgeGroup.AGE_45_49: 0.07,
            AgeGroup.AGE_50_54: 0.06,
            AgeGroup.AGE_55_59: 0.06,
            AgeGroup.AGE_60_64: 0.05,
            AgeGroup.AGE_65_69: 0.04,
            AgeGroup.AGE_70_74: 0.03,
            AgeGroup.AGE_75_79: 0.02,
            AgeGroup.AGE_80_PLUS: 0.01,
        }
        return weights.get(age_group, 0.05)


class BirthRateGenerator(DataGenerator):
    """Generate birth rate data with temporal and regional variation."""

    def generate(self, config: DemographicConfig, regions: List[RegionInfo]) -> pl.DataFrame:
        if config.random_seed:
            np.random.seed(config.random_seed + 1)

        records = []
        years = range(config.start_year, config.end_year + 1)

        for region in regions:
            base_rate = 0.015 + np.random.normal(0, 0.005)  # Base rate with regional variation
            base_rate = max(0.005, base_rate)  # Ensure positive

            for year in years:
                # Slight decline over time (demographic transition)
                temporal_factor = 1 - (year - config.start_year) * 0.001
                birth_rate = max(0.005, base_rate * temporal_factor)

                # Create structured record
                record = BirthRateRecord(
                    year=year,
                    region_id=region.region_id,
                    birth_rate=round(birth_rate, 6),
                    crude_birth_rate=round(birth_rate * 1000, 2),  # Per 1000 population
                    is_projected=True,
                    confidence_level=0.75,
                    data_source="Generated",
                )
                records.append(record.dict())

        # Convert to DataFrame and validate
        df = pl.DataFrame(records)
        return BirthRateRecord.validate_dataframe(df)


class MortalityRateGenerator(DataGenerator):
    """Generate mortality rate data with age-specific patterns."""

    def generate(self, config: DemographicConfig, regions: List[RegionInfo]) -> pl.DataFrame:
        if config.random_seed:
            np.random.seed(config.random_seed + 2)

        records = []
        years = range(config.start_year, config.end_year + 1)

        for region in regions:
            for year in years:
                for age_group_str in config.age_groups:
                    try:
                        age_group = AgeGroup(age_group_str)
                    except ValueError:
                        logger.warning(f"Invalid age group: {age_group_str}, skipping")
                        continue

                    base_rate = self._get_base_mortality_rate(age_group)
                    regional_factor = 1 + np.random.normal(0, 0.1)
                    mortality_rate = max(0.0001, base_rate * regional_factor)

                    # Create structured record
                    record = MortalityRateRecord(
                        year=year,
                        region_id=region.region_id,
                        age_group=age_group,
                        mortality_rate=round(mortality_rate, 6),
                        is_projected=True,
                        confidence_level=0.75,
                        data_source="Generated",
                    )
                    records.append(record.dict())

        # Convert to DataFrame and validate
        df = pl.DataFrame(records)
        return MortalityRateRecord.validate_dataframe(df)

    def _get_base_mortality_rate(self, age_group: AgeGroup) -> float:
        """Get base mortality rate by age group (J-curve pattern)."""
        rates = {
            AgeGroup.AGE_0_4: 0.005,
            AgeGroup.AGE_5_9: 0.0003,
            AgeGroup.AGE_10_14: 0.0002,
            AgeGroup.AGE_15_19: 0.0005,
            AgeGroup.AGE_20_24: 0.0008,
            AgeGroup.AGE_25_29: 0.0010,
            AgeGroup.AGE_30_34: 0.0012,
            AgeGroup.AGE_35_39: 0.0015,
            AgeGroup.AGE_40_44: 0.002,
            AgeGroup.AGE_45_49: 0.003,
            AgeGroup.AGE_50_54: 0.005,
            AgeGroup.AGE_55_59: 0.008,
            AgeGroup.AGE_60_64: 0.012,
            AgeGroup.AGE_65_69: 0.020,
            AgeGroup.AGE_70_74: 0.035,
            AgeGroup.AGE_75_79: 0.060,
            AgeGroup.AGE_80_PLUS: 0.120,
        }
        return rates.get(age_group, 0.01)


class SpatialProcessor:
    """Handle spatial operations and patch generation."""

    @staticmethod
    def create_patches(geometry, patch_size_km: float) -> List[Polygon]:
        """Create regular grid patches within a geometry."""
        bounds = geometry.bounds

        # Convert patch size to degrees (rough approximation)
        patch_size_deg = patch_size_km / 111.0  # ~111km per degree

        patches = []
        x_min, y_min, x_max, y_max = bounds

        x = x_min
        while x < x_max:
            y = y_min
            while y < y_max:
                patch = Polygon(
                    [(x, y), (x + patch_size_deg, y), (x + patch_size_deg, y + patch_size_deg), (x, y + patch_size_deg), (x, y)]
                )

                if patch.intersects(geometry):
                    patches.append(patch.intersection(geometry))

                y += patch_size_deg
            x += patch_size_deg

        return patches


class DemographicsGenerator:
    """Enhanced main class with improved architecture and error handling."""

    def __init__(self, config: DemographicConfig):
        self.config = config
        self._shapefile_hash = self._get_shapefile_hash()
        self._regions_cache: Optional[List[RegionInfo]] = None

        # Initialize data generators
        self.generators = {"population": PopulationGenerator(), "birth_rates": BirthRateGenerator(), "mortality": MortalityRateGenerator()}

        logger.info(f"Initialized DemographicsGenerator for {config.region} ({config.start_year}-{config.end_year})")

    def _get_shapefile_hash(self) -> Optional[str]:
        """Get hash of shapefile if it exists."""
        if self.config.shapefile_path:
            try:
                return file_hash(self.config.shapefile_path)
            except FileNotFoundError as e:
                raise ValidationError(f"Shapefile not found: {e}")
        return None

    def _get_cache_key(self, data_type: str) -> str:
        """Generate cache key for data type."""
        components = [data_type, self.config.to_cache_key()]
        if self._shapefile_hash:
            components.append(self._shapefile_hash)
        return ":".join(components)

    def _load_regions_from_shapefile(self) -> List[RegionInfo]:
        """Load regions from shapefile with error handling."""
        if not self.config.shapefile_path:
            return [RegionInfo(region_id=self.config.region, name=self.config.region, properties={})]

        cache_key = f"regions:{self._shapefile_hash}"
        cached_regions = cache_manager.get(cache_key)
        if cached_regions:
            logger.info(f"Loaded {len(cached_regions)} regions from cache")
            return cached_regions

        try:
            regions = []
            with fiona.open(self.config.shapefile_path, "r") as src:
                for i, feature in enumerate(src):
                    props = feature["properties"]
                    region_id = props.get("region_id", props.get("id", f"region_{i}"))
                    name = props.get("name", region_id)
                    geometry = shape(feature["geometry"]) if feature["geometry"] else None

                    regions.append(RegionInfo(region_id=str(region_id), name=str(name), properties=props, geometry=geometry))

            if self.config.granularity == "patch" and self.config.patch_size_km:
                regions = self._generate_patches(regions)

            cache_manager.set(cache_key, regions, expire=self.config.cache_ttl)
            logger.info(f"Loaded and cached {len(regions)} regions from shapefile")
            return regions

        except Exception as e:
            raise DataGenerationError(f"Failed to load shapefile: {e}")

    def _generate_patches(self, regions: List[RegionInfo]) -> List[RegionInfo]:
        """Generate patch-level regions from parent regions."""
        patch_regions = []

        for region in regions:
            if region.geometry:
                patches = SpatialProcessor.create_patches(region.geometry, self.config.patch_size_km)

                for i, patch in enumerate(patches):
                    patch_id = f"{region.region_id}_patch_{i}"
                    patch_regions.append(
                        RegionInfo(
                            region_id=patch_id,
                            name=f"{region.name} Patch {i}",
                            properties={**region.properties, "parent_region": region.region_id},
                            geometry=patch,
                        )
                    )

        return patch_regions

    def get_regions(self) -> pl.DataFrame:
        """Get regions as a DataFrame with enhanced information."""
        if not self._regions_cache:
            self._regions_cache = self._load_regions_from_shapefile()

        data = []
        for region in self._regions_cache:
            row = {
                "region_id": region.region_id,
                "name": region.name,
                "area": region.geometry.area if region.geometry else None,
                "centroid_x": region.geometry.centroid.x if region.geometry else None,
                "centroid_y": region.geometry.centroid.y if region.geometry else None,
            }
            # Add custom properties
            for key, value in region.properties.items():
                if key not in row:
                    row[key] = value
            data.append(row)

        return pl.DataFrame(data)

    def get_region_shapes(self) -> Dict[str, Any]:
        """Get region geometries as dictionary."""
        if not self._regions_cache:
            self._regions_cache = self._load_regions_from_shapefile()

        return {region.region_id: region.geometry for region in self._regions_cache if region.geometry}

    def plot_regions(self, figsize: Tuple[int, int] = (12, 8), show_labels: bool = True, title: Optional[str] = None):
        """Enhanced region plotting with better visualization."""
        shapes = self.get_region_shapes()
        if not shapes:
            logger.warning("No regions to plot")
            return

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set3(np.linspace(0, 1, len(shapes)))

        for i, (region_id, geom) in enumerate(shapes.items()):
            if geom.geom_type == "Polygon":
                x, y = geom.exterior.xy
                ax.fill(x, y, alpha=0.7, color=colors[i], edgecolor="black", linewidth=0.5, label=region_id)
                if show_labels:
                    centroid = geom.centroid
                    ax.annotate(region_id, (centroid.x, centroid.y), ha="center", va="center", fontsize=8)
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, alpha=0.7, color=colors[i], edgecolor="black", linewidth=0.5)

        ax.set_title(title or f"{self.config.region} Regions ({self.config.granularity})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)

        if len(shapes) <= 20:  # Only show legend for reasonable number of regions
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()

    def _generate_data_parallel(self, generator: DataGenerator, regions: List[RegionInfo]) -> pl.DataFrame:
        """Generate data using parallel processing if enabled."""
        if not self.config.parallel_processing or len(regions) < 4:
            return generator.generate(self.config, regions)

        # Split regions into chunks for parallel processing
        chunk_size = max(1, len(regions) // 4)
        region_chunks = [regions[i : i + chunk_size] for i in range(0, len(regions), chunk_size)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generator.generate, self.config, chunk) for chunk in region_chunks]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Parallel generation failed: {e}")
                    raise DataGenerationError(f"Parallel generation failed: {e}")

        # Combine results
        if results:
            return pl.concat(results)
        else:
            return pl.DataFrame()

    def generate_population(self) -> pl.DataFrame:
        """Generate population data with caching and validation."""
        cache_key = self._get_cache_key("population")
        cached = cache_manager.get(cache_key)
        if cached is not None:
            logger.info("Loaded population data from cache")
            return cached

        if not self._regions_cache:
            self._regions_cache = self._load_regions_from_shapefile()

        logger.info("Generating population data...")
        df = self._generate_data_parallel(self.generators["population"], self._regions_cache)

        # Validate output
        if df.is_empty():
            raise DataGenerationError("Generated population data is empty")

        cache_manager.set(cache_key, df, expire=self.config.cache_ttl)
        logger.info(f"Generated population data: {len(df)} records")
        return df

    def generate_birth_rates(self) -> pl.DataFrame:
        """Generate birth rate data with caching and validation."""
        cache_key = self._get_cache_key("birth_rates")
        cached = cache_manager.get(cache_key)
        if cached is not None:
            logger.info("Loaded birth rates from cache")
            return cached

        if not self._regions_cache:
            self._regions_cache = self._load_regions_from_shapefile()

        logger.info("Generating birth rate data...")
        df = self._generate_data_parallel(self.generators["birth_rates"], self._regions_cache)

        cache_manager.set(cache_key, df, expire=self.config.cache_ttl)
        logger.info(f"Generated birth rate data: {len(df)} records")
        return df

    def generate_mortality_rates(self) -> pl.DataFrame:
        """Generate mortality rate data with caching and validation."""
        cache_key = self._get_cache_key("mortality")
        cached = cache_manager.get(cache_key)
        if cached is not None:
            logger.info("Loaded mortality rates from cache")
            return cached

        if not self._regions_cache:
            self._regions_cache = self._load_regions_from_shapefile()

        logger.info("Generating mortality rate data...")
        df = self._generate_data_parallel(self.generators["mortality"], self._regions_cache)

        cache_manager.set(cache_key, df, expire=self.config.cache_ttl)
        logger.info(f"Generated mortality rate data: {len(df)} records")
        return df

    def generate_all_data(self) -> Dict[str, pl.DataFrame]:
        """Generate all demographic data types."""
        logger.info("Generating all demographic data...")

        return {
            "population": self.generate_population(),
            "birth_rates": self.generate_birth_rates(),
            "mortality_rates": self.generate_mortality_rates(),
        }

    def export_data(self, output_dir: Union[str, Path], formats: List[str] = None):
        """Export generated data to various formats."""
        formats = formats or ["parquet"]
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        data = self.generate_all_data()

        for data_type, df in data.items():
            base_filename = f"{self.config.region}_{data_type}_{self.config.start_year}_{self.config.end_year}"

            for fmt in formats:
                if fmt == "parquet":
                    filepath = output_dir / f"{base_filename}.parquet"
                    df.write_parquet(filepath)
                elif fmt == "csv":
                    filepath = output_dir / f"{base_filename}.csv"
                    df.write_csv(filepath)
                elif fmt == "json":
                    filepath = output_dir / f"{base_filename}.json"
                    df.write_json(filepath)

                logger.info(f"Exported {data_type} to {filepath}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache usage."""
        return cache_manager.get_stats()

    def clear_cache(self, data_type: Optional[str] = None):
        """Clear cache for specific data type or all cache."""
        if data_type:
            cache_key = self._get_cache_key(data_type)
            if cache_manager.cache.pop(cache_key, None):
                logger.info(f"Cleared cache for {data_type}")
        else:
            cache_manager.cache.clear()
            logger.info("Cleared all cache")


# Utility functions
def create_demo_config(region: str = "demo_country", years: Tuple[int, int] = (2020, 2025)) -> DemographicConfig:
    """Create a demo configuration for testing."""
    return DemographicConfig(region=region, start_year=years[0], end_year=years[1], granularity="national", random_seed=42)
