from pydantic import BaseModel, Field
import numpy as np
import json
from collections import OrderedDict

TIME_STEP_DAYS = 1
STATES = ["S", "E", "I", "R"]  # Compartments/states for discrete-time model

class GenericParams(BaseModel):
    """
    Parameters for the generic model.
    """

    nticks: int = Field(..., description="Number of time steps (bi-weekly for 20 years)")
    seed: int = Field(20241107, description="Random seed")
    start_time: str = Field("2005-01", description="Initial start time of simulation in YYYY-MM format")
    verbose: bool = Field(False, description="Whether to print verbose output")

    @property
    def time_step_days(self) -> int:
        return TIME_STEP_DAYS

    @property
    def states(self) -> list[str]:
        return STATES

    def __str__(self) -> str:
        return json.dumps(OrderedDict(sorted(self.model_dump().items())), indent=2)
