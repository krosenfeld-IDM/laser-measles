# How to run a multi-patch biweekly model

This guide shows you how to set up and run a five-patch biweekly model with births, deaths, and importation, and read per-patch infectious time series.

## Prerequisites

- laser-measles installed (`pip install laser-measles`)
- Basic familiarity with the [model construction pattern](../usage.md)

## Steps

### 1. Build the scenario

```python
import polars as pl

scenario = pl.DataFrame({
    "id":   [f"patch_{i}" for i in range(5)],
    "lat":  [0.0] * 5,
    "lon":  [float(i) for i in range(5)],   # must be Float64, not int
    "pop":  [50_000, 80_000, 120_000, 200_000, 150_000],
    "mcv1": [0.90, 0.85, 0.80, 0.75, 0.70],
})
```

`lat` and `lon` must be `Float64`. Using `list(range(5))` produces `Int64` and fails schema validation.

### 2. Configure parameters

```python
from laser.measles.biweekly import BiweeklyParams

# 26 ticks per year; 130 ticks = 5 years
params = BiweeklyParams(num_ticks=130, seed=42, start_time="2000-01")
```

`BiweeklyModel` uses 14-day ticks: 26 ticks = 1 year.

### 3. Construct the model

```python
from laser.measles.biweekly import BiweeklyModel

model = BiweeklyModel(scenario, params)
```

### 4. Add components

```python
from laser.measles.biweekly import (
    InitializeEquilibriumStatesProcess,
    ImportationPressureProcess,
    InfectionProcess,
    VitalDynamicsProcess,
    StateTracker,
    StateTrackerParams,
)
from laser.measles import create_component

model.add_component(InitializeEquilibriumStatesProcess)
model.add_component(ImportationPressureProcess)
model.add_component(InfectionProcess)
model.add_component(VitalDynamicsProcess)

# aggregation_level=0 produces a per-patch tracker with shape (num_ticks, n_patches)
model.add_component(
    create_component(
        StateTracker,
        params=StateTrackerParams(aggregation_level=0),
    )
)
```

!!! note
    The "VitalDynamics must be first" ordering rule applies to `ABMModel` only.
    In `BiweeklyModel`, `VitalDynamicsProcess` can appear after `InfectionProcess`.

### 5. Run the model

```python
model.run()
```

### 6. Retrieve results

```python
tracker = model.get_instance("StateTracker")[0]

# tracker.I shape: (num_ticks, n_patches) when aggregation_level=0
I = tracker.I   # shape: (130, 5)

print("Mean infectious per community (last 26 ticks = final year):")
for p, patch_id in enumerate(scenario["id"]):
    mean_I = float(I[-26:, p].mean())
    print(f"  {patch_id}: {mean_I:.1f}")
```

## Full script

```python
import polars as pl
from laser.measles.biweekly import BiweeklyModel, BiweeklyParams
from laser.measles.biweekly import InitializeEquilibriumStatesProcess, ImportationPressureProcess
from laser.measles.biweekly import InfectionProcess, VitalDynamicsProcess, StateTracker, StateTrackerParams
from laser.measles import create_component

scenario = pl.DataFrame({
    "id":   [f"patch_{i}" for i in range(5)],
    "lat":  [0.0] * 5,
    "lon":  [float(i) for i in range(5)],
    "pop":  [50_000, 80_000, 120_000, 200_000, 150_000],
    "mcv1": [0.90, 0.85, 0.80, 0.75, 0.70],
})

params = BiweeklyParams(num_ticks=130, seed=42, start_time="2000-01")
model  = BiweeklyModel(scenario, params)

model.add_component(InitializeEquilibriumStatesProcess)
model.add_component(ImportationPressureProcess)
model.add_component(InfectionProcess)
model.add_component(VitalDynamicsProcess)
model.add_component(
    create_component(
        StateTracker,
        params=StateTrackerParams(aggregation_level=0),
    )
)

model.run()

tracker = model.get_instance("StateTracker")[0]
I = tracker.I   # shape: (130, 5)

print("Mean infectious per community (last 26 ticks = final year):")
for p, patch_id in enumerate(scenario["id"]):
    mean_I = float(I[-26:, p].mean())
    print(f"  {patch_id}: {mean_I:.1f}")
```

## See also

- [Troubleshooting guide](../troubleshooting.md) — common errors with imports, scenario schema, tracker shapes, and components
- [Spatial Mixing tutorial](../tutorials/tut_spatial_mixing.ipynb) — configuring transmission between patches
- [Usage reference](../usage.md) — full model type and component overview
