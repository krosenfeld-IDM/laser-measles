import numpy as np
import json
from pydantic import BaseModel
from pydantic import Field
from collections import OrderedDict

# Constants
TIME_STEP_DAYS = 14  # Number of days per time step (biweekly)

def sort_dict(d: dict) -> OrderedDict:
    """
    Sort a dictionary by its keys alphabetically.
    
    Args:
        d (dict): Input dictionary to sort
        
    Returns:
        OrderedDict: Dictionary with alphabetically sorted keys
    """
    return OrderedDict(sorted(d.items()))

class BiweeklyParams(BaseModel):
    """
    Parameters for the biweekly model.
    """
    beta: float  = Field(32, description="Transmission rate")
    crude_birth_rate: float  = Field(0.0, description="Yearly crude birth rate per 1k population", ge=0.0)
    crude_death_rate: float  = Field(0.0, description="Yearly crude death rate per 1k population", ge=0.0)
    distance_exponent: float = Field(1.5, description="Distance exponent")
    mixing_scale: float = Field(0.001, description="Mixing scale")
    nticks: int = Field(..., description="Number of time steps (bi-weekly for 20 years)")
    seasonality: float = Field(0.06, description="Seasonality", ge=0.0)
    seed: int = Field(20241107, description="Random seed")
    start_time: str = Field("2005-01", description="Initial start time of simulation in YYYY-MM format")
    states: list[str] = Field(["S", "I", "R"], description="Compartments/states for discrete-time model")
    verbose: bool = Field(False, description="Whether to print verbose output")

    @property
    def time_step_days(self) -> int:
        return TIME_STEP_DAYS

    @property
    def mixing(self) -> np.ndarray:
        return self._mixing

    @mixing.setter
    def mixing(self, value: np.ndarray) -> None:
        self._mixing = value

    def __str__(self) -> str:
        return json.dumps(OrderedDict(sorted(self.model_dump().items())), indent=2)
