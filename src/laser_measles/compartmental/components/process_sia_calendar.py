from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from pydantic import BaseModel
from pydantic import Field

from laser_measles.base import BasePhase
from laser_measles.base import BaseLaserModel
from laser_measles.utils import cast_type


class SIACalendarParams(BaseModel):
    """Parameters specific to the SIA calendar component."""

    model_config = {"arbitrary_types_allowed": True}

    sia_efficacy: float = Field(0.9, description="Fraction of susceptibles to vaccinate in SIA", ge=0.0, le=1.0)
    filter_fn: Callable[[str], bool] = Field(lambda x: True, description="Function to filter which nodes to include in aggregation")
    aggregation_level: int = Field(3, description="Number of levels to use for aggregation (e.g., 3 for country:state:lga)")
    sia_schedule: pl.DataFrame = Field(description="DataFrame containing SIA schedule information")
    date_column: str = Field("date", description="Name of the column containing SIA dates")
    group_column: str = Field("id", description="Name of the column containing group identifiers")


class SIACalendarProcess(BasePhase):
    """
    Phase for implementing Supplementary Immunization Activities (SIAs) based on a calendar schedule.

    This component:
    1. Groups nodes by geographic level using the same aggregation schema as CaseSurveillanceTracker
    2. Implements SIAs at scheduled times by moving susceptibles to recovered state
    3. Converts tick numbers to dates using the model's start_time parameter
    4. Applies vaccination with configurable efficacy rate

    Parameters
    ----------
    model : object
        The simulation model containing nodes, states, and parameters
    verbose : bool, default=False
        Whether to print verbose output during simulation
    params : Optional[SIACalendarParams], default=None
        Component-specific parameters. If None, will use default parameters

    Notes
    -----
    - SIA efficacy determines the fraction of susceptibles that get vaccinated
    - Vaccination is simulated using a binomial distribution
    - SIAs are implemented when the model's current date has passed the scheduled date
    - Since the model steps in 1-day increments, SIAs are implemented on the exact scheduled date
    - Each SIA is implemented exactly once
    - Adapted from biweekly model to work with compartmental model's SEIR structure
    """

    def __init__(self, model, verbose: bool = False, params: SIACalendarParams | None = None) -> None:
        super().__init__(model, verbose)
        if params is None:
            raise ValueError("SIACalendarParams must be provided")
        self.params = params
        self._validate_params()

        # Extract node IDs and create mapping for filtered nodes
        self.node_mapping = {}

        for node_idx, node_id in enumerate(model.scenario["id"]):
            if self.params.filter_fn(node_id):
                # Create geographic grouping key
                group_key = ":".join(node_id.split(":")[: self.params.aggregation_level])
                if group_key not in self.node_mapping:
                    self.node_mapping[group_key] = []
                self.node_mapping[group_key].append(node_idx)

        # Track which SIAs have been implemented
        self.implemented_sias = set()

        # Parse start time for date calculations
        self.start_date = self._parse_start_time(model.params.start_time)

        if self.verbose:
            print(f"SIACalendar initialized with {len(self.node_mapping)} groups")

    def _validate_params(self) -> None:
        """Validate component parameters."""
        if self.params.aggregation_level < 1:
            raise ValueError("aggregation_level must be at least 1")

        # Validate SIA schedule DataFrame
        required_columns = [self.params.group_column, self.params.date_column]
        if not all(col in self.params.sia_schedule.columns for col in required_columns):
            raise ValueError(f"sia_schedule must contain columns: {required_columns}")

    def _parse_start_time(self, start_time: str) -> datetime:
        """Parse start time string to datetime object."""
        try:
            # Handle YYYY-MM format
            if len(start_time) == 7:  # YYYY-MM
                return datetime.strptime(start_time + "-01", "%Y-%m-%d")
            # Handle YYYY-MM-DD format
            elif len(start_time) == 10:  # YYYY-MM-DD
                return datetime.strptime(start_time, "%Y-%m-%d")
            else:
                raise ValueError(f"Unsupported start_time format: {start_time}")
        except ValueError as e:
            raise ValueError(f"Invalid start_time format '{start_time}': {e}")

    def _tick_to_date(self, tick: int) -> datetime:
        """Convert tick number to date."""
        return self.start_date + timedelta(days=tick)

    def __call__(self, model, tick: int) -> None:
        # Get current state counts - use StateArray attribute access
        states = model.patches.states

        # Convert tick to current date
        current_date = self._tick_to_date(tick)

        # Check for SIAs scheduled for dates up to and including the current date
        # Convert sia_schedule dates to datetime for comparison
        sia_schedule = self.params.sia_schedule.with_columns([
            pl.col(self.params.date_column).str.to_datetime().alias("datetime_date")
        ]).filter(pl.col("datetime_date") <= current_date)

        # Apply SIAs to each scheduled group
        for row in sia_schedule.iter_rows(named=True):
            group_key = row[self.params.group_column]
            scheduled_date = row["datetime_date"]

            # Create a unique identifier for this SIA
            sia_id = f"{group_key}_{scheduled_date}"

            # Skip if this SIA has already been implemented
            if sia_id in self.implemented_sias:
                continue

            if group_key in self.node_mapping:
                node_indices = self.node_mapping[group_key]

                # Get susceptible population for this group using StateArray
                susceptibles = states.S[node_indices]

                # Sample number to vaccinate using binomial distribution
                vaccinated = cast_type(np.random.binomial(n=susceptibles, p=self.params.sia_efficacy), states.dtype)

                # Update states: move vaccinated from susceptible to recovered
                states.S[node_indices] -= vaccinated
                states.R[node_indices] += vaccinated

                # Mark this SIA as implemented
                self.implemented_sias.add(sia_id)

                if self.verbose:
                    total_vaccinated = vaccinated.sum()
                    if total_vaccinated > 0:
                        print(
                            f"Date {current_date.strftime('%Y-%m-%d')}: Implementing SIA for {group_key} (scheduled for {scheduled_date.strftime('%Y-%m-%d')}) - vaccinated {total_vaccinated} individuals"
                        )

    def initialize(self, model: BaseLaserModel) -> None:
        pass

    def get_sia_schedule(self) -> pl.DataFrame:
        """
        Get the SIA schedule.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - {group_column}: Group identifier
            - {date_column}: Scheduled date for SIA
        """
        return self.params.sia_schedule