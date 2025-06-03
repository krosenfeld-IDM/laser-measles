from pathlib import Path

import numpy as np
import pandas as pd
from .base import BaseComponent


class Step(BaseComponent):
    """
    Component for simulating the spread of infection in the model.
    """
    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)

    def __call__(self, model, tick: int) -> None:
        # TODO: implement math in-memory

        # state counts
        states = model.nodes.states

        def cast_type(a, dtype):
            return a.astype(dtype) if a.dtype != dtype else a

        # model parameters
        params = model.params
        # calculate the expected number of new infections
        expected = params.beta * (1 + params.seasonality * np.cos(2 * np.pi * tick / 26.0)) * np.matmul(params.mixing, states[1])
        prob = 1 - np.exp(-expected / states.sum(axis=0))  # probability of infection
        dI = cast_type(np.random.binomial(n=states[0], p=prob), states.dtype)  # number of new infections in S

        states[2] += states[1]  # move I to R (assuming 14 day recovery)
        states[1] = 0  # reset I

        # Vital dynamics
        population = states.sum(axis=0)
        biweek_avg_births = population * (params.crude_birth_rate / 26.0 / 1000.0)
        births = cast_type(np.random.poisson(biweek_avg_births), states.dtype)  # number of births
        biweek_avg_deaths = population * (params.crude_death_rate / 26.0 / 1000.0)
        deaths = cast_type(np.random.poisson(biweek_avg_deaths), states.dtype)  # number of deaths

        states[0] += births  # add births to S
        states -= deaths  # remove deaths from each compartment

        states[1] += dI  # add new infections to I
        states[0] -= dI  # remove new infections from S

        # make sure that states >= 0
        np.maximum(states, 0, out=states)

        return

class CrudeRateBirths(BaseComponent):
    """Component for calculating births based on crude birth rates."""

    def __call__(self, model, tick: int) -> None:
        states = model.nodes.states
        params = model.params

        def cast_type(a, dtype):
            return a.astype(dtype) if a.dtype != dtype else a

        # Calculate births based on crude birth rates (per 1000 population) for each node
        # The crude birth rate is annual, so we divide by 26 for biweekly rate
        current_population = states.sum(axis=0)
        biweek_birth_rates = params.demog_scale * params.crude_birth_rates / 1000 / 26.0
        expected_births = current_population * biweek_birth_rates

        births = np.random.poisson(expected_births).astype(states.dtype)
        births = cast_type(births, states.dtype)  # number of births, NB this rounds down

        # Apply MCV vaccination: split births between S and R based on coverage
        if hasattr(params, 'mcv_coverage_original'):
            # Scale the MCV coverage during the step function using current vac_fac
            scaled_mcv_coverage = np.minimum(1.0, params.mcv_coverage_original * params.vac_fac)

            # Calculate vaccinated births (go directly to R)
            vaccinated_births = cast_type(np.random.binomial(n=births, p=scaled_mcv_coverage), states.dtype)
            # Remaining births go to susceptible
            susceptible_births = births - vaccinated_births

            # Add vaccinated births directly to recovered compartment
            states[2] += vaccinated_births
            # Add remaining births to susceptible
            states[0] += susceptible_births
        else:
            # If no MCV coverage data, all births go to susceptible as before
            states[0] += births

class KanoStep(BaseComponent):
    """
    Component for simulating the spread of infection in the Kano model.
    """
    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)

    def __call__(self, model, tick: int) -> None:
        # TODO: implement math in-memory

        # state counts
        states = model.nodes.states

        def cast_type(a, dtype):
            return a.astype(dtype) if a.dtype != dtype else a

        # model parameters
        params = model.params

        # calculate the expected number of new infections
        expected = params.beta * (1 + params.seasonality * np.cos(2 * np.pi * tick / 26.0)) * np.matmul(params.mixing, states[1])
        prob = 1 - np.exp(-expected / states.sum(axis=0))  # probability of infection
        dI = cast_type(np.random.binomial(n=states[0], p=prob), states.dtype)  # number of new infections in S

        # Store new cases for tracking
        model.nodes.new_cases = dI.copy()

        states[2] += states[1]  # move I to R (assuming 14 day recovery)
        states[1] = 0  # reset I

        # Use the death probability calculated from the current population
        # rather than using a fixed probability
        deaths = cast_type(np.random.binomial(n=states, p=params.biweek_death_prob), states.dtype)  # number of deaths

        # Remove deaths from each compartment
        states -= deaths

        states[1] += dI  # add new infections to I
        states[0] -= dI  # remove new infections from S

        return


class SIA(BaseComponent):
    """
    Component for simulating the spread of infection in the SIA model.
    """
    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)

        # Check if node groups were initialized
        if not hasattr(model.nodes, "node_groups"):
            raise AttributeError("Nodes must have 'node_groups' attribute for SIA to work")

    def __call__(self, model, tick: int) -> None:
        # State counts
        states = model.nodes.states

        # Model parameters
        params = model.params

        # Get unique node groups
        unique_groups = np.unique(model.nodes.node_groups)

        def cast_type(a, dtype):
            return a.astype(dtype) if a.dtype != dtype else a

        # Decide for each group if SIA should happen
        for group in unique_groups:
            # Skip if group is -1 (ungrouped nodes)
            if group == -1:
                continue

            # Get mask for nodes in this group
            group_mask = model.nodes.node_groups == group

            # Roll dice to decide if SIA happens for this group
            if np.random.random() < params.prob_SIA:
                # Apply SIA to all nodes in this group
                susceptible_count = states[0, group_mask]

                # Calculate how many susceptibles to move to recovered
                to_move = cast_type(np.floor(susceptible_count * params.fac_SIA), states.dtype)

                # Move from S to R
                states[0, group_mask] -= to_move
                states[2, group_mask] += to_move

                if self.verbose:
                    print(f"SIA applied to group {group}, moved {np.sum(to_move)} from S to R")

        return

class ImportationPressure(BaseComponent):
    """
    Component for simulating importation of new cases based on population size.
    Default rate is 1 importation per 100k people per time step.
    """
    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        # Set default importation rate (can be overridden in parameters)
        if not hasattr(model.params, 'importation_rate'):
            raise AttributeError("Importation rate must be set in parameters")

    def __call__(self, model, tick: int) -> None:
        # state counts
        states = model.nodes.states
        params = model.params

        def cast_type(a, dtype):
            return a.astype(dtype) if a.dtype != dtype else a

        # Calculate current population for each node
        population = states.sum(axis=0)

        # Calculate expected number of importations based on population size
        expected_importations = population * params.importation_rate

        # Sample actual number of importations from Poisson distribution
        importations = cast_type(np.random.poisson(expected_importations), states.dtype)

        # Add importations to infected compartment, remove from susceptible
        states[1] += importations
        states[0] -= importations

        return

class SIACalendar(BaseComponent):
    """
    Component for immunization on fixed SIA calendar.
    """
    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        # Load SIA calendar data
        csv_path = Path(__file__).parent / "kano" / "sia_efficacy_by_state.csv"
        self.calendar_df = pd.read_csv(csv_path, parse_dates=["time"])
        self.calendar_df["state"] = self.calendar_df["state"].str.lower()
        # Extract date for matching
        self.calendar_df["date"] = self.calendar_df["time"].dt.date
        # Add a flag column to track which campaigns have been implemented
        self.calendar_df["implemented"] = False
        # Map node indices by state
        node_ids = model.scenario.index.tolist()
        self.nodes_by_state = {}
        for idx, node_id in enumerate(node_ids):
            st = node_id.split(".")[0].lower()
            self.nodes_by_state.setdefault(st, []).append(idx)

    def __call__(self, model, tick: int) -> None:
        states = model.nodes.states
        current_date = model.current_date.date()

        # Find campaigns that are due but not yet implemented
        due_camps = self.calendar_df[
            (self.calendar_df["date"] <= current_date) &
            (~self.calendar_df["implemented"])
        ]

        for _, camp in due_camps.iterrows():
            st = camp["state"]
            idxs = self.nodes_by_state.get(st, [])
            if not idxs:
                continue
            sus = states[0, idxs]
            frac = camp.get("avg", 0.0)
            # Vaccinate fraction of susceptibles
            to_move = np.floor(sus * frac).astype(states.dtype)
            states[0, idxs] -= to_move
            states[2, idxs] += to_move
            # Mark this campaign as implemented
            self.calendar_df.loc[self.calendar_df["time"] == camp["time"], "implemented"] = True
            if self.verbose:
                print(f"SIACalendar applied on {current_date} for state {st}, moved {to_move.sum()} from S to R")
        return
