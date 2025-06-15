# %%
import numpy as np
import pandas as pd
from laser_core.propertyset import PropertySet
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve

from laser_measles.generic import Model
from laser_measles.generic import Infection
from laser_measles.generic import Exposure
from laser_measles.generic import Susceptibility
from laser_measles.generic import Transmission
from laser_measles.generic import BirthsConstantPop
from laser_measles.generic.components.importation import InfectAgentsInPatch

from laser_measles.generic.utils import set_initial_susceptibility_in_patch
from laser_measles.generic.utils import seed_infections_in_patch


# %% [markdown]
# Construct the synthetic populations. We'll have 61 patches with populations distributed logarithmicaly between 1k and 1M people.

# %%
nticks = 10 * 365 # lenth of the simulation in days
npatches = 61 # number of patches (spatial units)
pops = np.logspace(3, 6, npatches)
scenario = pd.DataFrame({"ids": [str(i) for i in range(npatches)], "population": pops})

# %% [markdown]
# Run `nsims=200` iterations sampling over R0, mean infectious perios, and crude birth rate

# %%
nsims = 1 # 200
R0_samples = np.random.uniform(3, 16, nsims)
infmean_samples = 5 + np.random.gamma(2, 10, nsims)
cbr_samples = 10 + np.random.gamma(2, 20, nsims)
i = 0
outputs = np.zeros((nsims, nticks, npatches))
# Create a folder to store the outputs
output_folder = os.path.abspath(os.path.join(os.getcwd(), "CCS"))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for R0, infmean, cbr in zip(R0_samples, infmean_samples, cbr_samples):
    parameters = PropertySet(
        {
            "seed": np.random.randint(0, 1000000),
            "nticks": nticks,
            "verbose": True,
            "beta": R0 / infmean,
            "inf_mean": infmean,
            "exp_mu": 2.5,
            "exp_sigma":0.4,
            "inf_shape": 2,
            "cbr": cbr,
            "importation_period": 180,
            "importation_end": 20 * 365,
        }
    )

    mu = (1 + parameters.cbr / 1000) ** (1 / 365) - 1

    model = Model(scenario, parameters)
    model.components = [
        BirthsConstantPop,
        Susceptibility,
        Exposure,
        Infection,
        Transmission,
        InfectAgentsInPatch,
    ]

    # Start them slightly asynchronously - different initial susceptibilities, infection only in 1 patch
    # Want to see how connectivity drives correlation over time.
    for j in range(npatches):
        set_initial_susceptibility_in_patch(model, j, 1 / R0 + 0.1 / R0 * np.random.normal())

    model.run()
    outputs[i, :, :] = model.patches.cases_test
    np.save(f"{output_folder}/CCSSIRoutputs_{i}.npy", outputs[i, :, :])
    i += 1

# %%
print(model.population.susceptibility.min(), model.population.susceptibility.max())
print(model.population.susceptibility.sum())


# %%
print(model.population.itimer.max(), model.population.itimer.min())
print(np.sum(model.population.itimer > 0))

# %%
plt.plot(outputs.sum(axis=-1).flatten())

# %%
plt.plot(model.patches.cases_test.sum(axis=-1))

# %%
model.patches.cases.dtype

# %%
model.patches.cases_test.dtype

# %%
hasattr(model.population, "etimer")

# %%



