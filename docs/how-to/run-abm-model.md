# How to run an ABM outbreak model

This guide shows you how to set up and run a single-patch ABM simulation and read the infectious time series from a `StateTracker`.

## Prerequisites

- laser-measles installed (`pip install laser-measles`)
- Basic familiarity with the [model construction pattern](../usage.md)

## Steps

### 1. Build the scenario

```python
from laser.measles.scenarios import single_patch_scenario

scenario = single_patch_scenario(population=100_000, mcv1_coverage=0.0)
```

`single_patch_scenario` returns a Polars DataFrame with all required columns (`id`, `lat`, `lon`, `pop`, `mcv1`). Patch IDs are 1-indexed — `id = "patch_1"`.

### 2. Configure parameters

```python
from laser.measles.abm import ABMParams

params = ABMParams(num_ticks=365, seed=42, start_time="2000-01")
```

`num_ticks` is in days. `start_time` must be `"YYYY-MM"` format — not `"YYYY-MM-DD"`.

### 3. Construct the model

```python
from laser.measles.abm import ABMModel

model = ABMModel(scenario, params)
```

### 4. Add components

```python
from laser.measles.abm import (
    NoBirthsProcess,
    InitializeEquilibriumStatesProcess,
    InfectionSeedingProcess,
    InfectionProcess,
    StateTracker,
)

model.add_component(NoBirthsProcess)
model.add_component(InitializeEquilibriumStatesProcess)
model.add_component(InfectionSeedingProcess)
model.add_component(InfectionProcess)
model.add_component(StateTracker)
```

Pass component **classes**, not instances. `NoBirthsProcess` keeps the population fixed; omit it and use `VitalDynamicsProcess` first when you need births and deaths.

### 5. Run the model

```python
model.run()
```

### 6. Retrieve results

```python
tracker = model.get_instance("StateTracker")[0]

peak_I   = int(tracker.I.max())
peak_day = int(tracker.I.argmax())
print(f"Peak infectious: {peak_I} on day {peak_day}")
```

`tracker.I` is a 1-D NumPy array of shape `(num_ticks,)`. Wrap scalar results in `int()` or `float()` before printing or passing to Polars.

## Full script

```python
import numpy as np
import polars as pl
from laser.measles.abm import ABMModel, ABMParams
from laser.measles.abm import NoBirthsProcess, InitializeEquilibriumStatesProcess
from laser.measles.abm import InfectionSeedingProcess, InfectionProcess, StateTracker
from laser.measles.scenarios import single_patch_scenario

scenario = single_patch_scenario(population=100_000, mcv1_coverage=0.0)
params   = ABMParams(num_ticks=365, seed=42, start_time="2000-01")
model    = ABMModel(scenario, params)

model.add_component(NoBirthsProcess)
model.add_component(InitializeEquilibriumStatesProcess)
model.add_component(InfectionSeedingProcess)
model.add_component(InfectionProcess)
model.add_component(StateTracker)

model.run()

tracker  = model.get_instance("StateTracker")[0]
peak_I   = int(tracker.I.max())
peak_day = int(tracker.I.argmax())
print(f"Peak infectious: {peak_I} on day {peak_day}")
```

## See also

- [Troubleshooting guide](../troubleshooting.md) — common errors with imports, scenario schema, tracker shapes, and components
- [ABM Introduction tutorial](../tutorials/tut_abm_intro.ipynb) — step-by-step introduction to the ABM
- [Usage reference](../usage.md) — full model type and component overview
