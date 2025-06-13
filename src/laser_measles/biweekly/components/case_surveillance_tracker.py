import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from typing import Callable

from laser_measles.biweekly.base import BaseComponent


def cast_type(a, dtype):
    return a.astype(dtype) if a.dtype != dtype else a


class CaseSurveillanceParams(BaseModel):
    """Parameters specific to the case surveillance component."""
    detection_rate: float = Field(0.1, description="Probability of detecting an infected case", ge=0.0, le=1.0)
    filter_fn: Callable[[str], bool] = Field(
        lambda x: True,
        description="Function to filter which nodes to include in aggregation"
    )


class CaseSurveillanceTracker(BaseComponent):
    """
    Component for tracking detected cases in the model.

    This component:
    1. Simulates case detection based on a detection rate
    2. Tracks detected cases aggregated by LGA (or other geographic level)
    3. Uses a filter function to determine which nodes to include in aggregation

    Parameters
    ----------
    model : object
        The simulation model containing nodes, states, and parameters
    verbose : bool, default=False
        Whether to print verbose output during simulation
    params : Optional[CaseSurveillanceParams], default=None
        Component-specific parameters. If None, will use default parameters

    Notes
    -----
    - Case detection is simulated using a binomial distribution
    - Cases are aggregated by LGA by default (can be modified to aggregate by other levels)
    - Only stores aggregated cases to save memory
    - Uses a filter function to determine which nodes to include in aggregation
    """

    def __init__(self, model, verbose: bool = False, params: CaseSurveillanceParams | None = None) -> None:
        super().__init__(model, verbose)
        self.params = params or CaseSurveillanceParams()
        self._validate_params()
        
        # Extract LGA IDs and create mapping for filtered nodes
        self.lga_mapping = {}
        for node_idx, node_id in enumerate(model.scenario['ids']):
            if self.params.filter_fn(node_id):
                lga_id = ':'.join(node_id.split(':')[:3])  # country:state:lga
                if lga_id not in self.lga_mapping:
                    self.lga_mapping[lga_id] = []
                self.lga_mapping[lga_id].append(node_idx)
        
        # Initialize reported cases tracker (nticks x num_lgas)
        self.reported_cases = np.zeros(
            (model.params.nticks, len(self.lga_mapping)),
            dtype=model.nodes.states.dtype
        )
        
        # Store LGA IDs in order
        self.lga_ids = sorted(self.lga_mapping.keys())

    def _validate_params(self) -> None:
        """Validate component parameters."""
        if not 0 <= self.params.detection_rate <= 1:
            raise ValueError("detection_rate must be between 0 and 1")

    def __call__(self, model, tick: int) -> None:
        # Get current infected cases
        infected = model.nodes.states[1]  # Infected state is index 1
        
        # For each LGA, aggregate detected cases from its nodes
        for lga_idx, (lga_id, node_indices) in enumerate(self.lga_mapping.items()):
            # Get infected cases for this LGA's nodes
            lga_infected = infected[node_indices]
            
            # Simulate case detection using binomial distribution
            detected_cases = cast_type(
                np.random.binomial(n=lga_infected, p=self.params.detection_rate),
                model.nodes.states.dtype
            )
            
            # Store total detected cases for this LGA
            self.reported_cases[tick, lga_idx] = detected_cases.sum()

    def get_reported_cases(self) -> pl.DataFrame:
        """
        Get a DataFrame of reported cases by LGA over time.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - tick: Time step
            - lga_id: LGA identifier
            - cases: Number of reported cases
        """
        # Create a list to store the data
        data = []
        
        # For each tick and LGA, add the reported cases
        for tick in range(self.model.params.nticks):
            for lga_idx, lga_id in enumerate(self.lga_ids):
                data.append({
                    'tick': tick,
                    'lga_id': lga_id,
                    'cases': self.reported_cases[tick, lga_idx]
                })
        
        # Create DataFrame
        return pl.DataFrame(data) 