# %% [markdown]
# # Generic Model Initialization Tutorial
# 
# This tutorial demonstrates how to initialize a generic measles model with three core components:
# - Births (with constant population)
# - Transmission
# - Disease progression
# 
# The example shows the basic workflow for setting up and running a spatial epidemiological model.

# %%
import numpy as np
import pandas as pd
from laser_core.propertyset import PropertySet
import matplotlib.pyplot as plt

from laser_measles.generic.model import Model
from laser_measles.generic.params import GenericParams
from laser_measles.generic.components import (
    BirthsConstantPopProcess, BirthsParams,
    TransmissionProcess, TransmissionParams,
    StatesTracker,
    PopulationTracker,
)
from laser_measles.generic.components.process_disease import DiseaseProcess, DiseaseParams
from laser_measles.components import create_component
# %% [markdown]
# ## Step 1: Create a Simple Scenario
# 
# Define a scenario with multiple patches representing different population centers.

# %%
# Create scenario with 5 patches of different population sizes
scenario = pd.DataFrame({
    'name': ['City_A', 'City_B', 'Town_C', 'Village_D', 'Settlement_E'],
    'population': [50000, 30000, 15000, 8000, 2000],
    'latitude': [40.7128, 34.0522, 41.8781, 39.9526, 47.6062],
    'longitude': [-74.0060, -118.2437, -87.6298, -75.1652, -122.3321]
})

print("Scenario overview:")
print(scenario)
print(f"\nTotal population: {scenario['population'].sum():,}")

# %% [markdown]
# ## Step 2: Configure Model Parameters
# 
# Set up the simulation parameters including disease characteristics and simulation length.

# %%
# Basic simulation parameters
nticks = 365 * 2  # 2 years
seed = 42

# Disease parameters
R0 = 12.0  # Basic reproduction number for measles
infectious_period = 8.0  # days
transmission_rate = R0 / infectious_period

parameters = PropertySet({
    'seed': seed,
    'nticks': nticks,
    'verbose': True,
    'start_time': '2000-01',
    
    # Birth parameters (constant population)
    'cbr': 20.0,  # Crude birth rate per 1000 per year
    
    # Transmission parameters
    'beta': transmission_rate,
    'exp_mu': np.log(11.0),  # Exposure period mean (lognormal) #TODO: figure out if this should be log
    'exp_sigma': 2.0,        # Exposure period std (lognormal)
    
    # Infection parameters
    'inf_mean': infectious_period,
    'inf_sigma': 2.0,
})

print("Model parameters:")
for key, value in parameters.to_dict().items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## Step 3: Initialize the Model
# 
# Create the model instance with the scenario and parameters.

# %%
model_params = GenericParams(nticks=parameters.nticks, start_time=parameters.start_time, seed=parameters.seed)
model = Model(scenario, model_params, name="Tutorial_Generic_Model")

print(f"Model initialized with {len(model.patches)} patches")
print(f"Total population capacity: {model.people.capacity:,}")

# %% [markdown]
# ## Step 4: Configure Model Components
# 
# Set up the three core components: births, transmission, and disease progression.

# %%
# Create component parameter objects
births_params = BirthsParams(cbr=parameters.cbr)

transmission_params = TransmissionParams(
    beta=parameters.beta,
    exp_mu=parameters.exp_mu,
    exp_sigma=parameters.exp_sigma,
)

disease_params = DiseaseParams(
    inf_mean=parameters.inf_mean,
    inf_sigma=parameters.inf_sigma
)

# Set up model components
model.components = [
    create_component(BirthsConstantPopProcess, params=births_params),  # Births with constant population
    create_component(TransmissionProcess, params=transmission_params),      # Disease transmission
    create_component(DiseaseProcess, params=disease_params),          # Disease progression (E->I->R)
    StatesTracker,
    PopulationTracker,
]

print("Components configured:")
for i, component in enumerate(model.components, 1):
    print(f"  {i}. {component.__name__}")

# %% [markdown]
# ## Step 5: Set Initial Conditions
# 
# Initialize population susceptibility and seed initial infections.

# %%

# Set initial susceptibility based on herd immunity threshold
# For measles with R0=12, herd immunity threshold ≈ 1-1/R0 ≈ 0.92
herd_immunity_threshold = 1 - 1/R0
initial_susceptible_fraction = 1 - herd_immunity_threshold + 0.05  # Slightly above threshold

# Randomly assign susceptibility
np.random.seed(seed)
n_susceptible = int(model.people.capacity * initial_susceptible_fraction)
susceptible_indices = np.random.choice(
    model.people.capacity, 
    size=n_susceptible, 
    replace=False
)
print(f"Initial susceptible population: {n_susceptible:,}")

# # Initialize everyone as recovered (immune), then set some as susceptible
# model.population.susceptibility[:] = 0  # 0 = immune/recovered
# model.population.susceptibility[susceptible_indices] = 1  # 1 = susceptible

# Add state property for disease states (S=0, E=1, I=2, R=3)
# model.population.add_scalar_property("state", dtype=np.uint8, default=3)  # Start as recovered
model.people.state[:] = 3 # Start as recovered
model.people.state[susceptible_indices] = 0  # Set susceptible individuals to S

print(f"Initial conditions:")
print(f"Initial population: {model.patches.populations.sum():,}")


# Seed initial infections in the largest patch
largest_patch_id = scenario['population'].idxmax()
patch_population_mask = model.people.nodeid == largest_patch_id
patch_susceptible_mask = (model.people.state == 0) & patch_population_mask

# Infect 10 individuals in the largest patch
patch_susceptible_indices = np.where(patch_susceptible_mask)[0]
if len(patch_susceptible_indices) >= 10:
    initial_infected_indices = np.random.choice(patch_susceptible_indices, size=10, replace=False)
    model.people.state[initial_infected_indices] = 2  # Set to infectious
    
    print(f"  Seeded 10 initial infections in {scenario.loc[largest_patch_id, 'name']}")

# %% [markdown]
# ## Step 6: Run the Simulation
# 
# Execute the model for the specified number of ticks.

# %%
print(f"\nRunning simulation for {nticks} days...")
model.run()

print(f"Simulation completed!")
print(f"Final population: {model.patches.populations.sum():,}")

# %% [markdown]
# ## Step 7: Visualize Results
# 
# Create basic plots to examine the simulation results.

# %%
# Get the PopulationTracker and StateStracker instances
for instance in model.instances:
    if isinstance(instance, PopulationTracker):
        population_tracker = instance
    elif isinstance(instance, StatesTracker):
        states_tracker = instance

# %%
# Plot population over time for each patch
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Population dynamics by patch
ax1 = axes[0]
for i, patch_name in enumerate(scenario['name']):
    ax1.plot(population_tracker.population_tracker[i,:], label=patch_name)
ax1.set_title('Population Over Time by Patch')
ax1.set_xlabel('Days')
ax1.set_ylabel('Population')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cases over time (if transmission component added case tracking)
ax2 = axes[1]
total_cases = states_tracker.state_tracker[2,:]
ax2.plot(total_cases)
ax2.set_title('Total Cases Over Time')
ax2.set_xlabel('Days')
ax2.set_ylabel('Cases')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
# 
# This tutorial demonstrated the basic workflow for initializing a generic measles model:
# 
# 1. **Scenario Definition**: Created a multi-patch scenario with population data
# 2. **Parameter Setup**: Configured simulation and disease parameters
# 3. **Model Initialization**: Created the Model instance
# 4. **Component Configuration**: Added births, transmission, and disease components
# 5. **Initial Conditions**: Set susceptibility levels and seeded infections
# 6. **Simulation**: Ran the model for the specified time period
# 7. **Visualization**: Examined the results
# 
# The model includes:
# - **BirthsConstantPopProcess**: Maintains constant population through balanced births/deaths
# - **TransmissionProcess**: Handles disease transmission between individuals
# - **DiseaseProcess**: Manages disease state progression (Exposed → Infectious → Recovered)
# 
# This framework can be extended with additional components like vaccination, seasonal forcing, 
# or spatial connectivity to create more complex epidemiological models.

# %%