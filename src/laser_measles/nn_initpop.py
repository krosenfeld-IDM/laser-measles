import numpy as np
from laser_core.demographics import AliasedDistribution
from laser_core.demographics import load_pyramid_csv
from tqdm import tqdm


def setup_initial_population(model, verbose: bool = False) -> None:
    # Add a property for node IDs. 419 nodes requires 9 bits so we will allocate a 16-bit value.
    # Negative IDs don't make sense, so, uint16
    model.population.add_scalar_property("nodeid", dtype=np.uint16)
    for nodeid, count in enumerate(model.patches.populations[0, :]):
        first, last = model.population.add(count)
        model.population.nodeid[first:last] = nodeid

    model.population.add_scalar_property("alive", dtype=bool)
    model.population.alive[0 : model.population.count] = True

    # Initialize population ages

    pyramid_file = model.params.pyramid_file
    age_distribution = load_pyramid_csv(pyramid_file)
    both = age_distribution[:, 2] + age_distribution[:, 3]  # males + females
    sampler = AliasedDistribution(both)
    bin_min_age_days = age_distribution[:, 0] * 365  # minimum age for bin, in days (include this value)
    bin_max_age_days = (age_distribution[:, 1] + 1) * 365  # maximum age for bin, in days (exclude this value)
    initial_pop = model.population.count
    samples = sampler.sample(initial_pop)  # sample for bins from pyramid
    model.population.add_scalar_property("dob", dtype=np.int32)
    mask = np.zeros(initial_pop, dtype=bool)
    dobs = model.population.dob[0:initial_pop]
    print("Assigning day of year of birth to agents…")
    for i in tqdm(range(len(age_distribution))):  # for each possible bin value...
        mask[:] = samples == i  # ...find the agents that belong to this bin
        # ...and assign a random age, in days, within the bin
        dobs[mask] = model.prng.integers(bin_min_age_days[i], bin_max_age_days[i], mask.sum())

    return
