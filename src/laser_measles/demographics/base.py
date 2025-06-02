from typing import Protocol
import polars as pl

class DemographicsGenerator(Protocol):
    
    def generate_population(self) -> pl.DataFrame:
        ...
    
    def generate_birth_rates(self) -> pl.DataFrame:
        ...
    
    def generate_mortality_rates(self) -> pl.DataFrame:
        ...



class Shapefile(Protocol):

    def add_dotname(self) -> None:
        ...

    def download(self, country_code: str, timeout: int = 60) -> None:
        ...

    def get_cache_key(self) -> str:
        ...

    def get_cache_dir(self) -> str:
        ...