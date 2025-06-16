"""
Base classes for laser-measles components and models
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar

import alive_progress
import click
from laser_core.random import seed as seed_prng
from matplotlib.figure import Figure

ScenarioType = TypeVar('ScenarioType')
ParamsType = TypeVar('ParamsType')


class BaseComponent:
    """
    Base class for all laser-measles components.
    
    Components follow a uniform interface with __call__(model, tick) method
    for execution during simulation loops.
    """
    
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose
        self.initialized = True

    def __call__(self, model, tick: int) -> None: 
        """Execute component logic for a given simulation tick."""
        ...

    def __str__(self) -> str:
        """Return string representation using class docstring."""
        # Use child class docstring if available, otherwise parent class
        doc = self.__class__.__doc__ or BaseComponent.__doc__
        return doc.strip() if doc else f"{self.__class__.__name__} component"

    def plot(self, fig: Figure | None = None):
        """
        Placeholder for plotting method.
        """
        yield None


class BaseLaserModel(ABC, Generic[ScenarioType, ParamsType]):
    """
    Base class for laser-measles simulation models.
    
    Provides common functionality for model initialization, component management,
    timing, metrics collection, and execution loops.
    """
    
    def __init__(self, scenario: ScenarioType, parameters: ParamsType, name: str) -> None:
        """
        Initialize the model with common attributes.
        
        Args:
            scenario: Scenario data (type varies by model)
            parameters: Model parameters (type varies by model)
            name: Model name
        """
        self.tinit = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tinit}: Creating the {name} model…")
        
        self.scenario = scenario
        self.params = parameters
        self.name = name
        
        # Initialize random number generator
        seed_value = parameters.seed if hasattr(parameters, 'seed') and parameters.seed is not None else self.tinit.microsecond
        self.prng = seed_prng(seed_value)
        
        # Component management attributes
        self._components: list = []
        self.instances: list = []
        self.phases: list = []
        
        # Metrics and timing
        self.metrics: list = []
        self.tstart: datetime | None = None
        self.tfinish: datetime | None = None
        
        # Time tracking
        self.start_time = datetime.strptime(self.params.start_time, "%Y-%m") # noqa DTZ007
        self.current_date = self.start_time

    
    @property
    def components(self) -> list:
        """
        Retrieve the list of model components.
        
        Returns:
            list: A list containing the components.
        """
        return self._components
    
    @components.setter
    def components(self, components: list) -> None:
        """
        Sets up the components of the model and initializes instances and phases.
        
        Args:
            components: A list of component classes to be initialized and integrated into the model.
        """
        self._components = components
        self.instances = [self]
        self.phases = [self]
        for component in components:
            instance = component(self, getattr(self.params, 'verbose', False))
            self.instances.append(instance)
            if "__call__" in dir(instance):
                self.phases.append(instance)
        
        # Allow subclasses to perform additional component setup
        self._setup_components()
    
    def _setup_components(self) -> None:
        """
        Hook for subclasses to perform additional component setup.
        Override in subclasses as needed.
        """
        pass
    
    @abstractmethod
    def __call__(self, model: Any, tick: int) -> None:
        """
        Updates the model for a given tick.
        
        Args:
            model: The model instance
            tick: The current time step or tick
        """
        pass
    
    def run(self) -> None:
        """
        Execute the model for a specified number of ticks, recording timing metrics.
        """
        nticks = getattr(self.params, 'nticks', 0)
        self.tstart = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tstart}: Running the {self.name} model for {nticks} ticks…")
        
        self.metrics = []
        with alive_progress.alive_bar(nticks) as bar:
            for tick in range(nticks):
                self._execute_tick(tick)
                bar()
        
        self.tfinish = datetime.now(tz=None)  # noqa: DTZ005
        print(f"Completed the {self.name} model at {self.tfinish}…")
        
        if getattr(self.params, 'verbose', False):
            self._print_timing_summary()
    
    def _execute_tick(self, tick: int) -> None:
        """
        Execute a single tick. Can be overridden by subclasses for custom behavior.
        
        Args:
            tick: The current tick number
        """
        timing = [tick]
        for phase in self.phases:
            tstart = datetime.now(tz=None)  # noqa: DTZ005
            phase(self, tick)            
            tfinish = datetime.now(tz=None)  # noqa: DTZ005
            delta = tfinish - tstart
            timing.append(delta.seconds * 1_000_000 + delta.microseconds)
        self.metrics.append(timing)

        # Update current date by time_step_days
        self.current_date += timedelta(days=self.params.time_step_days)
    
    def _print_timing_summary(self) -> None:
        """
        Print timing summary for verbose mode.
        """
        try:
            import pandas as pd
            names = [type(phase).__name__ for phase in self.phases]
            metrics = pd.DataFrame(self.metrics, columns=["tick", *list(names)])
            plot_columns = metrics.columns[1:]
            sum_columns = metrics[plot_columns].sum()
            width = max(map(len, sum_columns.index))
            for key in sum_columns.index:
                print(f"{key:{width}}: {sum_columns[key]:13,} µs")
            print("=" * (width + 2 + 13 + 3))
            print(f"{'Total:':{width + 1}} {sum_columns.sum():13,} microseconds")
        except ImportError:
            try:
                import polars as pl
                names = [type(phase).__name__ for phase in self.phases]
                metrics = pl.DataFrame(self.metrics, schema=["tick"] + names)
                plot_columns = metrics.columns[1:]
                sum_columns = metrics.select(plot_columns).sum()
                # Handle polars DataFrame differently
                print("Timing summary available but detailed formatting requires pandas")
            except ImportError:
                print("Timing summary requires pandas or polars")