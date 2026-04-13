# How to run a Compartmental model with an R0 sweep

This guide shows you how to run the same single-patch compartmental model at multiple R0 values and compare peak infectious counts.

## Prerequisites

- laser-measles installed (`pip install laser-measles`)
- Basic familiarity with the [model construction pattern](../usage.md)

## Steps

### 1. Identify the beta-to-R0 relationship

The default `InfectionParams` ships with a `beta` calibrated to R0 ≈ 8. Scale `beta` linearly to reach any other R0:

```python
R0_DEFAULT   = 8.0
BETA_DEFAULT = 0.5714285714285714
```

### 2. Loop over target R0 values

For each R0, build a fresh scenario, params, and model:

```python
from laser.measles.scenarios import single_patch_scenario
from laser.measles.compartmental import CompartmentalModel, CompartmentalParams
from laser.measles.compartmental import (
    InitializeEquilibriumStatesProcess,
    InfectionSeedingProcess,
    InfectionProcess,
    InfectionParams,
    StateTracker,
    StateTrackerParams,
)
from laser.measles import create_component

R0_DEFAULT   = 8.0
BETA_DEFAULT = 0.5714285714285714

for target_r0 in [4.0, 8.0, 16.0]:

    scenario = single_patch_scenario(population=100_000, mcv1_coverage=0.0)
    params   = CompartmentalParams(num_ticks=730, seed=42, start_time="2000-01")
    model    = CompartmentalModel(scenario, params)

    model.add_component(InitializeEquilibriumStatesProcess)
    model.add_component(InfectionSeedingProcess)

    scaled_beta = target_r0 * (BETA_DEFAULT / R0_DEFAULT)
    model.add_component(
        create_component(
            InfectionProcess,
            params=InfectionParams(beta=scaled_beta),
        )
    )

    model.add_component(
        create_component(
            StateTracker,
            params=StateTrackerParams(aggregation_level=0),
        )
    )

    model.run()

    tracker = model.get_instance("StateTracker")[0]
    I = tracker.I[:, 0]   # single patch → 1-D array of length num_ticks
    print(f"R0={target_r0:.0f}: peak I = {int(I.max()):,} on day {int(I.argmax())}")
```

`CompartmentalModel` uses daily ticks: `num_ticks=730` is 2 years.

### Key points

- `InfectionParams` accepts `beta` directly — there is no `beta_scale` field.
- Per-patch tracker shape is `(num_ticks, n_patches)`. For a single patch, index column 0.
- `CompartmentalParams` does not accept `beta`, `sigma`, or `gamma` directly; those live on `InfectionParams`.

## Full script

```python
import polars as pl
from laser.measles.compartmental import CompartmentalModel, CompartmentalParams
from laser.measles.compartmental import InitializeEquilibriumStatesProcess, InfectionSeedingProcess
from laser.measles.compartmental import InfectionProcess, InfectionParams, StateTracker, StateTrackerParams
from laser.measles.scenarios import single_patch_scenario
from laser.measles import create_component

R0_DEFAULT   = 8.0
BETA_DEFAULT = 0.5714285714285714

for target_r0 in [4.0, 8.0, 16.0]:

    scenario = single_patch_scenario(population=100_000, mcv1_coverage=0.0)
    params   = CompartmentalParams(num_ticks=730, seed=42, start_time="2000-01")
    model    = CompartmentalModel(scenario, params)

    model.add_component(InitializeEquilibriumStatesProcess)
    model.add_component(InfectionSeedingProcess)

    scaled_beta = target_r0 * (BETA_DEFAULT / R0_DEFAULT)
    model.add_component(
        create_component(
            InfectionProcess,
            params=InfectionParams(beta=scaled_beta),
        )
    )

    model.add_component(
        create_component(
            StateTracker,
            params=StateTrackerParams(aggregation_level=0),
        )
    )

    model.run()

    tracker = model.get_instance("StateTracker")[0]
    I = tracker.I[:, 0]
    print(f"R0={target_r0:.0f}: peak I = {int(I.max()):,} on day {int(I.argmax())}")
```

## See also

- [Troubleshooting guide](../troubleshooting.md) — common errors with imports, scenario schema, tracker shapes, and components
- [Usage reference](../usage.md) — full model type and component overview
