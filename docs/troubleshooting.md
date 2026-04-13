# Troubleshooting

Common errors and their fixes, organized by topic. If you are new to laser-measles, read the [Usage](usage.md) page and work through the [tutorials](tutorials/index.md) first.

---

## Import and namespace errors

### Where does `create_component` come from?

`create_component` is available from the top-level `laser.measles` namespace and from any model subpackage:

```python
# PREFERRED
from laser.measles import create_component

# Also valid
from laser.measles.abm import create_component
from laser.measles.biweekly import create_component
from laser.measles.compartmental import create_component
from laser.measles.components import create_component
```

### How do I import component classes and their parameter classes?

Import component and parameter classes directly from the model subpackage. Each subpackage re-exports its own components at the top level:

```python
# PREFERRED — import from the correct model subpackage
from laser.measles.abm import ABMModel, ABMParams
from laser.measles.abm import NoBirthsProcess, InfectionSeedingProcess, InfectionSeedingParams
from laser.measles.abm import InfectionProcess, StateTracker, StateTrackerParams
from laser.measles import create_component
```

!!! warning
    Component and parameter classes are **model-specific**. `InfectionParams`, `SIACalendarParams`, and similar classes have different fields per model type and live in their respective subpackage. Do not import them from `laser.measles.components` or from the wrong model subpackage.

    ```python
    # CORRECT — import each class from its own model subpackage
    from laser.measles.abm          import InfectionParams   # ABM variant
    from laser.measles.biweekly     import InfectionParams   # Biweekly variant
    from laser.measles.compartmental import InfectionParams  # Compartmental variant
    ```

    `NoBirthsProcess` and `SIACalendarProcess` exist in the ABM subpackage only.

### There is no `lm` object in `laser.measles`

The top-level `laser.measles` package does not export a convenience object called `lm`. Some AI-generated examples use this alias, but it is not part of the API.

```python
# CORRECT
from laser.measles.abm import ABMModel, ABMParams
```

### Scenario helpers are in `laser.measles` or `laser.measles.scenarios`, not in model subpackages

```python
# CORRECT
from laser.measles import single_patch_scenario, two_patch_scenario, two_cluster_scenario
# or equivalently
from laser.measles.scenarios import single_patch_scenario
```

Do not import scenario helpers from `laser.measles.abm`, `laser.measles.biweekly`, or `laser.measles.compartmental` — they raise `ImportError`.

### Do not use try/except import blocks or dict fallbacks for params

Do not write defensive import patterns like:

```python
try:
    from laser.measles.abm import InfectionParams
except ImportError:
    InfectionParams = None
```

and then fall back to passing a plain dict as `params`. If an import fails, fix the import path. See [How do I import component classes](#how-do-i-import-component-classes-and-their-parameter-classes) above for correct paths.

---

## Model construction and components

### `model.components` is assigned after construction

The model constructors accept only `scenario` and `params`. Assign components after construction:

```python
# CORRECT
model = BiweeklyModel(scenario=scenario, params=params)

model.components = [
    InitializeEquilibriumStatesProcess,
    ImportationPressureProcess,
    InfectionProcess,
    VitalDynamicsProcess,
    StateTracker,
]
```

Do not pass `components` as a constructor argument — it raises `TypeError: unexpected keyword argument "components"`.

### Components are classes, not instances

Pass component **classes** to `add_component` or `model.components`. The model instantiates them internally.

```python
# CORRECT — pass the class
model.add_component(InfectionProcess)

# If parameters are needed, use create_component
model.add_component(
    create_component(InfectionProcess, params=InfectionParams(beta=0.8))
)
```

Passing an already-instantiated object raises `TypeError: 'InfectionProcess' object is not callable`.

### `VitalDynamicsProcess` must be the first component (ABM only)

When using vital dynamics in `ABMModel`, `VitalDynamicsProcess` must be the **first** component. It calls `calculate_capacity` to pre-allocate the `LaserFrame` — adding any other component first results in a crash at runtime.

```python
# CORRECT — VitalDynamicsProcess first in ABMModel
model.add_component(VitalDynamicsProcess)
model.add_component(InitializeEquilibriumStatesProcess)
model.add_component(InfectionProcess)
```

!!! note
    This ordering rule applies to `ABMModel` only. In `BiweeklyModel`, `VitalDynamicsProcess` can appear after `InfectionProcess`.

### Do NOT add `TransmissionProcess` separately when using `InfectionProcess` (ABM)

`InfectionProcess` already instantiates `TransmissionProcess` internally. Adding it as a separate component causes `ValueError: Property 'etimer' already exists`.

```python
# CORRECT — InfectionProcess is self-contained
model.add_component(InfectionProcess)
```

### Custom components must accept `verbose`

`ABMModel.add_component` instantiates components as `ComponentClass(model, verbose=False)`. Custom components must accept `verbose`:

```python
class MyTracker:
    def __init__(self, model, verbose: bool = False):
        self.model = model
```

Omitting `verbose` raises `TypeError: MyTracker.__init__() got an unexpected keyword argument 'verbose'`.

### Never pass a plain dict as `params`

All params objects are Pydantic models. Passing a plain dict raises `AttributeError` immediately at construction — `BaseLaserModel.__init__` accesses `params.verbose` and `params.start_time` before any component runs.

```python
# CORRECT — use the typed Pydantic class
from laser.measles.abm import InfectionProcess, InfectionParams
from laser.measles import create_component

model.add_component(
    create_component(InfectionProcess, params=InfectionParams(beta=1.2))
)
# NOT: params={"beta": 1.2}
```

---

## Scenario data

### Required columns

All model constructors require these columns in the scenario DataFrame:

| Column | Type | Description |
|--------|------|-------------|
| `id` | `Utf8` (string) | Patch identifier |
| `lat` | `Float64` | Latitude |
| `lon` | `Float64` | Longitude |
| `pop` | `Int32` | Population size |
| `mcv1` | `Float64` | Routine vaccination coverage |

Missing columns trigger a validation error at construction.

### Use scenario helper functions for test scenarios

The `synthetic` module provides ready-made DataFrames with correct dtypes:

```python
from laser.measles.scenarios import single_patch_scenario, two_patch_scenario
from laser.measles.scenarios import two_cluster_scenario, satellites_scenario

scenario = single_patch_scenario(population=50_000, mcv1_coverage=0.85)
```

!!! warning
    **Patch IDs from helper functions are 1-indexed**, not 0-indexed:

    - `single_patch_scenario()` → `id = "patch_1"`
    - `two_patch_scenario()` → `id = ["patch_1", "patch_2"]`

    If you pass `target_patches=["patch_0"]` to `InfectionSeedingParams` when using a helper-built scenario, the model raises `ValueError: Target patches not found`. Omit `target_patches` entirely to seed all patches, or read the ID from the scenario:

    ```python
    patch_id = scenario["id"][0]   # "patch_1"
    ```

### `lat` and `lon` must be `Float64`, not `Int64`

Using Python's `range()` or integer literals produces `Int64` columns, which fail schema validation. Use float literals:

```python
# CORRECT
scenario = pl.DataFrame({
    "lat": [0.0] * 5,
    "lon": [float(i) for i in range(5)],
    ...
})
```

### `id` must be a string; `pop` must be `Int32`

- `id`: Use string lists like `["patch_0", "patch_1"]`. Integer lists like `[0, 1]` produce `Int64` and fail validation.
- `pop`: Python integer lists produce `Int64` by default. Use `np.array(..., dtype=np.int32)` or cast after construction:

```python
import numpy as np, polars as pl

scenario = pl.DataFrame({
    "pop": np.array([100_000, 80_000], dtype=np.int32),
    ...
})
# Or cast after construction
scenario = scenario.with_columns(pl.col("pop").cast(pl.Int32))
```

### Tick granularity: daily vs biweekly

`ABMModel` and `CompartmentalModel` use **daily** ticks (1 tick = 1 day).
`BiweeklyModel` uses **14-day** ticks (26 ticks = 1 year).

Scale `num_ticks` accordingly:

```python
# 5 years
ABMParams(num_ticks=5 * 365)           # 1825
BiweeklyParams(num_ticks=5 * 26)       # 130
CompartmentalParams(num_ticks=5 * 365) # 1825
```

---

## Trackers and results

### `StateTracker` output shape depends on `aggregation_level`

**Default (global)**: `aggregation_level=-1` — arrays are **1-D** with shape `(num_ticks,)`:

```python
model.add_component(StateTracker)
model.run()

tracker = model.get_instance("StateTracker")[0]
peak_I = int(tracker.I.max())
```

**Per-patch**: `aggregation_level=0` — arrays are **2-D** with shape `(num_ticks, n_patches)`:

```python
model.add_component(
    create_component(StateTracker, params=StateTrackerParams(aggregation_level=0))
)
model.run()

tracker = model.get_instance("StateTracker")[0]
peak_patch_0 = int(tracker.I[:, 0].max())
```

Do not mix these up — indexing a global tracker with `[:, 0]` will raise `IndexError`.

### Retrieving results from `StateTracker`

`StateTracker` does not expose `.data`, `.results`, `.to_polars()`, or `.df`. These do not exist.

After `model.run()`, retrieve the tracker with `model.get_instance("StateTracker")[0]` and access time-series arrays directly:

- Global tracker: `tracker.I`, `tracker.S`, `tracker.R` — shape `(num_ticks,)`
- Per-patch tracker: `tracker.state_tracker` — shape `(n_states, n_ticks, n_patches)`; state index order: `S=0, E=1, I=2, R=3`

```python
# Global tracker
tracker = model.get_instance("StateTracker")[0]
peak_I = int(tracker.I.max())

# Per-patch tracker
st = tracker.state_tracker   # shape: (n_states, n_ticks, n_patches)
peak_I_patch0 = int(st[2, :, 0].max())   # I index = 2
```

Use `get_dataframe()` for global trackers or `.state_tracker` for per-patch trackers.

### Cast NumPy scalars before building a Polars DataFrame

Tracker arrays are NumPy arrays, so `.max()` returns `np.int64` or `np.float64`. Polars expects Python primitives when constructing row-oriented DataFrames. Wrap with `int()` or call `.item()`:

```python
# CORRECT
rows.append([patch_id, int(tracker.I[:, p].max())])
# or
rows.append([patch_id, tracker.I[:, p].max().item()])
```

### `StateTracker` values are `StateArray` objects, not plain Python scalars

Indexing `tracker.I[tick]` returns a `StateArray`, not a `float`. Using it in an f-string format spec raises `TypeError: unsupported format string`. Extract a scalar first:

```python
# CORRECT
frac = float(tracker.I[tick])
print(f"infected fraction: {frac:.4f}")
```

### Per-patch attack rates from `StateTracker` (multi-patch models)

When `aggregation_level=0`, the raw array has shape `(n_states, n_ticks, n_patches)`:

```python
st  = model.get_instance(StateTracker)[0].state_tracker
# Typical ABM state order: S=0, E=1, I=2, R=3, D=4
initial_S = st[0,  0, :].astype(float)
final_R   = st[3, -1, :].astype(float)

# Use scenario population as the denominator, not tracker values
pop = scenario["pop"].to_numpy().astype(float)
attack_rate = final_R / pop
```

### `two_cluster_scenario` returns 100 patches by default

`two_cluster_scenario(n_nodes_per_cluster=50)` creates **100 patches** (2 × 50). A per-patch `StateTracker` will have shape `(n_states, n_ticks, 100)`.

To use a smaller scenario, pass `n_nodes_per_cluster`:

```python
scenario = two_cluster_scenario(n_nodes_per_cluster=5)  # 10 patches
```

---

## Component-specific behavior

### SIA schedule date column must use `datetime.date` values, not strings

`SIACalendarProcess` raises `InvalidOperationError: cannot compare 'date/datetime/time' to a string value` if the `date` column contains strings. Use `datetime.date` objects:

```python
import datetime, polars as pl

sia_df = pl.DataFrame({
    "date": [datetime.date(2024, 6, 1), datetime.date(2025, 6, 1)],
    ...
})
# Or cast after construction
sia_df = sia_df.with_columns(pl.col("date").str.to_date())
```

### `AgePyramidTracker` — reading the age distribution data

`AgePyramidTracker` stores snapshots in `.age_pyramid`, a `dict[str, np.ndarray]` keyed by date strings (`"YYYY-MM-DD"`). There is no `.counts` attribute. Iterate or access by key:

```python
apt = model.get_instance(AgePyramidTracker)[0]

for date_str, counts in apt.age_pyramid.items():
    print(f"{date_str}: {counts.sum()} agents tracked")

# First and last snapshots
dates = sorted(apt.age_pyramid.keys())
start_counts = apt.age_pyramid[dates[0]]
end_counts   = apt.age_pyramid[dates[-1]]
```

Never index with an integer — `tracker.age_pyramid[0]` raises `KeyError: 0`. Do not hardcode date strings — retrieve keys dynamically as shown above.

### `SIACalendarParams.aggregation_level` must be ≥ 1

Passing `aggregation_level=0` raises `ValueError: aggregation_level must be at least 1`. Use `aggregation_level=1` for flat hierarchies, or a higher number for hierarchical IDs like `"country:state:lga"`:

```python
from laser.measles.abm.components import SIACalendarParams

params = SIACalendarParams(aggregation_level=1, sia_schedule=schedule_df, ...)
```

### `model.people` has `date_of_birth`, not `age`

The ABM people `LaserFrame` stores `date_of_birth` (in ticks). Accessing `model.people.age` raises `AttributeError`:

```python
# CORRECT
dob = model.people.date_of_birth[model.people.active.view(bool)]
current_tick = model.params.num_ticks - 1
age_years = (current_tick - dob) / 365.0
```

Available people properties: `state`, `susceptibility`, `patch_id`, `active`, `date_of_birth`, `date_of_vaccination`.

### `get_mixing_matrix()` takes no arguments

All mixing models accept the scenario at construction time, not at `get_mixing_matrix()` call time:

```python
from laser.measles import RadiationMixing, RadiationParams

mixer = RadiationMixing(scenario=scenario, params=RadiationParams())
mixing_matrix = mixer.get_mixing_matrix()   # no arguments
```

Calling `mixer.get_mixing_matrix(scenario)` raises `TypeError: takes 1 positional argument but 2 were given`.

### `lookup_state_idx` does not exist — use `params.states.index()`

There is no `lookup_state_idx` function in `laser.measles`. Find state indices via the `states` list on the model params:

```python
params = BiweeklyParams(...)
S_IDX = params.states.index('S')
I_IDX = params.states.index('I')
R_IDX = params.states.index('R')
```

---

## Python and library pitfalls

### Multiprocessing workers must be defined at module level

Python's `multiprocessing` cannot pickle functions defined inside another function. Define worker functions at the top level of the module:

```python
# CORRECT — top-level function is picklable
def _worker(model_type):
    ...

def run_all_models():
    with Pool() as p:
        results = p.map(_worker, model_types)
```

Alternatively, use `concurrent.futures.ProcessPoolExecutor` with `functools.partial`.

### polars `with_column` (singular) was removed — use `with_columns`

```python
# CORRECT
df = df.with_columns(pl.col("pop").cast(pl.Int32))
```

`DataFrame.with_column` (singular) was removed in a recent Polars release. Using it raises `AttributeError: 'DataFrame' object has no attribute 'with_column'`.

### `AgePyramidTracker.age_pyramid` is a dict, not an array

`tracker.age_pyramid[0]` raises `KeyError: 0`. Use dict access:

```python
keys = sorted(tracker.age_pyramid.keys())
start_pyramid = tracker.age_pyramid[keys[0]]
end_pyramid   = tracker.age_pyramid[keys[-1]]
# or
first_array = next(iter(tracker.age_pyramid.values()))
```

### `numpy` has no `cummax` — use `np.maximum.accumulate`

`np.cummax` does not exist. The equivalent is:

```python
result = np.maximum.accumulate(arr)
```
