from pydantic import BaseModel, Field
import json
from collections import OrderedDict

TIME_STEP_DAYS = 1
STATES = ["S", "E", "I", "R"] 

class ABMParams(BaseModel):
    """
    Parameters for the ABM model.
    """

    nticks: int = Field(..., description="Number of time steps (daily)")
    seed: int = Field(20241107, description="Random seed")
    start_time: str = Field("2000-01", description="Initial start time of simulation in YYYY-MM format")
    verbose: bool = Field(False, description="Whether to print verbose output")

    @property
    def time_step_days(self) -> int:
        return TIME_STEP_DAYS

    @property
    def states(self) -> list[str]:
        return STATES

    def __str__(self) -> str:
        return json.dumps(OrderedDict(sorted(self.model_dump().items())), indent=2)
