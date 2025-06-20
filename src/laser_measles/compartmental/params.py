import json
from collections import OrderedDict

from pydantic import BaseModel
from pydantic import Field

TIME_STEP_DAYS = 1
STATES = ["S", "E", "I", "R"]  # Compartments/states for SEIR model

class CompartmentalParams(BaseModel):
    """
    Parameters for the compartmental SEIR model with daily timesteps.
    """

    num_ticks: int = Field(..., description="Number of time steps (days)")
    seed: int = Field(20241107, description="Random seed")
    start_time: str = Field("2005-01", description="Initial start time of simulation in YYYY-MM format")
    verbose: bool = Field(False, description="Whether to print verbose output")
    
    # SEIR-specific epidemiological parameters
    beta: float = Field(0.5, description="Transmission rate per day", gt=0.0)
    sigma: float = Field(1.0/8.0, description="Progression rate from exposed to infectious (1/incubation_period)", gt=0.0)
    gamma: float = Field(1.0/5.0, description="Recovery rate from infection (1/infectious_period)", gt=0.0)
    
    # Spatial mixing parameters
    mixing_scale: float = Field(0.001, description="Mixing scale for spatial diffusion", ge=0.0)
    distance_exponent: float = Field(1.5, description="Distance exponent for gravity model", ge=0.0)
    
    # Seasonal parameters
    seasonality: float = Field(0.0, description="Seasonality factor", ge=0.0)
    season_start: int = Field(0, description="Season start day (0-364)", ge=0, le=364)

    @property
    def time_step_days(self) -> int:
        return TIME_STEP_DAYS

    @property
    def states(self) -> list[str]:
        return STATES
    
    @property
    def basic_reproduction_number(self) -> float:
        """Calculate R0 = beta / gamma"""
        return self.beta / self.gamma
    
    @property
    def incubation_period(self) -> float:
        """Average incubation period in days"""
        return 1.0 / self.sigma
    
    @property
    def infectious_period(self) -> float:
        """Average infectious period in days"""
        return 1.0 / self.gamma

    def __str__(self) -> str:
        return json.dumps(OrderedDict(sorted(self.model_dump().items())), indent=2)