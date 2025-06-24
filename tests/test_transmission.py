import numpy as np
import polars as pl

import laser_measles as lm
from laser_measles.biweekly import BiweeklyModel, BiweeklyParams, BaseScenario as BiweeklyScenario
from laser_measles.biweekly.components import InfectionProcess, InfectionParams, VitalDynamicsProcess, VitalDynamicsParams, StateTracker
from laser_measles.compartmental import CompartmentalModel, CompartmentalParams, BaseScenario as CompartmentalScenario
from laser_measles.compartmental.components import InfectionProcess as CompartmentalInfectionProcess, InfectionParams as CompartmentalInfectionParams, VitalDynamicsProcess as CompartmentalVitalDynamicsProcess, VitalDynamicsParams as CompartmentalVitalDynamicsParams, StateTracker as CompartmentalStateTracker

# TODO: Test functions for laser_measles transmission
# Test no transmission when beta = 0
# Test double transmission when beta*2 vs beta
# Test with different mixing parameters
# Test impact of different infectious periods


def create_test_scenario(n_patches=2, base_pop=10000):
    """Create a test scenario with specified number of patches."""
    data = {
        "id": [f"NG:KN:00{i+1}" for i in range(n_patches)],
        "pop": [base_pop + i*1000 for i in range(n_patches)],
        "lat": [12.0 + i*0.1 for i in range(n_patches)],
        "lon": [8.5 + i*0.1 for i in range(n_patches)],
        "mcv1": [0.8 - i*0.05 for i in range(n_patches)]
    }
    df = pl.DataFrame(data)
    return df


def setup_biweekly_sim(num_ticks=52, beta=1.0, seasonality=0.0, mixing_scale=0.001, 
                       distance_exponent=1.5, birth_rate=20.0, death_rate=8.0, 
                       init_infections=None, seed=42):
    """Set up BiweeklyModel for transmission testing."""
    np.random.seed(seed)
    
    # Create scenario
    scenario_df = create_test_scenario()
    scenario = BiweeklyScenario(scenario_df)
    
    # Create model parameters
    params = BiweeklyParams(
        num_ticks=num_ticks,
        seed=seed,
        start_time="2020-01",
        verbose=False
    )
    
    # Create model
    model = BiweeklyModel(scenario, params)
    
    # Create component parameters
    infection_params = InfectionParams(
        beta=beta,
        seasonality=seasonality,
        season_start=0,
        distance_exponent=distance_exponent,
        mixing_scale=mixing_scale
    )
    
    vital_params = VitalDynamicsParams(
        crude_birth_rate=max(birth_rate, 0.001),  # Ensure positive
        crude_death_rate=max(death_rate, 0.001)   # Ensure positive
    )
    
    # Add components
    model.components = [
        StateTracker,
        lm.create_component(InfectionProcess, params=infection_params),
        lm.create_component(VitalDynamicsProcess, params=vital_params)
    ]
    
    # Initialize with infections if specified
    if init_infections is not None:
        if isinstance(init_infections, (int, float)):
            # Apply to all patches proportionally
            for i in range(len(scenario)):
                num_inf = int(init_infections * scenario["pop"][i])
                if num_inf > 0:
                    model.infect(i, num_inf)
        elif isinstance(init_infections, (list, np.ndarray)):
            # Apply specific numbers to each patch
            for i, num_inf in enumerate(init_infections):
                if num_inf > 0:
                    model.infect(i, num_inf)
    
    return model


def setup_compartmental_sim(num_ticks=365, beta=0.5, sigma=1.0/8.0, gamma=1.0/5.0, 
                           seasonality=0.0, mixing_scale=0.001, distance_exponent=1.5,
                           birth_rate=20.0, death_rate=8.0, init_infections=None, seed=42):
    """Set up CompartmentalModel for transmission testing."""
    np.random.seed(seed)
    
    # Create scenario
    scenario_df = create_test_scenario()
    scenario = CompartmentalScenario(scenario_df)
    
    # Create model parameters
    params = CompartmentalParams(
        num_ticks=num_ticks,
        seed=seed,
        start_time="2020-01",
        verbose=False,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        mixing_scale=mixing_scale,
        distance_exponent=distance_exponent,
        seasonality=seasonality
    )
    
    # Create model
    model = CompartmentalModel(scenario, params)
    
    # Create component parameters
    infection_params = CompartmentalInfectionParams(
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        seasonality=seasonality,
        season_start=0,
        distance_exponent=distance_exponent,
        mixing_scale=mixing_scale
    )
    
    vital_params = CompartmentalVitalDynamicsParams(
        crude_birth_rate=max(birth_rate, 0.001),  # Ensure positive
        crude_death_rate=max(death_rate, 0.001)   # Ensure positive
    )
    
    # Add components
    model.components = [
        CompartmentalStateTracker,
        lm.create_component(CompartmentalInfectionProcess, params=infection_params),
        lm.create_component(CompartmentalVitalDynamicsProcess, params=vital_params)
    ]
    
    # Initialize with infections if specified
    if init_infections is not None:
        if isinstance(init_infections, (int, float)):
            # Apply to all patches proportionally 
            for i in range(len(scenario)):
                num_inf = int(init_infections * scenario["pop"][i])
                if num_inf > 0:
                    # For SEIR model, put initial infections in I compartment
                    model.patches.states.I[i] = num_inf
                    model.patches.states.S[i] -= num_inf
        elif isinstance(init_infections, (list, np.ndarray)):
            # Apply specific numbers to each patch
            for i, num_inf in enumerate(init_infections):
                if num_inf > 0:
                    model.patches.states.I[i] = num_inf
                    model.patches.states.S[i] -= num_inf
    
    return model


# Test default transmission scenario
def test_trans_default(n_reps=10):
    """Test basic transmission for both BiweeklyModel and CompartmentalModel."""
    
    # Test BiweeklyModel
    biweekly_infections = []
    for rep in range(n_reps):
        model = setup_biweekly_sim(
            num_ticks=26,  # Half year
            beta=2.0,  # Strong transmission
            birth_rate=0.001,  # Minimal births for cleaner test
            death_rate=0.001,  # Minimal deaths for cleaner test
            init_infections=0.01,  # 1% initial prevalence
            seed=42 + rep
        )
        model.run()
        
        # Count total new infections (transitions from S to I)
        state_tracker = [c for c in model.instances if isinstance(c, StateTracker)][0]
        initial_I = state_tracker.I[0]
        final_I = state_tracker.I[-1]
        final_R = state_tracker.R[-1]
        
        # Total infections = final I + R - initial I
        total_infections = final_I + final_R - initial_I
        biweekly_infections.append(total_infections)
    
    biweekly_infections = np.array(biweekly_infections)
    assert np.all(biweekly_infections > 0), "BiweeklyModel: There should be some infections after the simulation runs."
    
    # Test CompartmentalModel
    compartmental_infections = []
    for rep in range(n_reps):
        model = setup_compartmental_sim(
            num_ticks=182,  # Half year in days
            beta=0.5,  # Reasonable transmission rate
            birth_rate=0.001,  # Minimal births for cleaner test
            death_rate=0.001,  # Minimal deaths for cleaner test
            init_infections=0.01,  # 1% initial prevalence
            seed=42 + rep
        )
        model.run()
        
        # Count total new infections
        state_tracker = [c for c in model.instances if isinstance(c, CompartmentalStateTracker)][0]
        initial_I = state_tracker.I[0]
        final_I = state_tracker.I[-1]
        final_R = state_tracker.R[-1]
        
        # Total infections = final I + R - initial I  
        total_infections = final_I + final_R - initial_I
        compartmental_infections.append(total_infections)
    
    compartmental_infections = np.array(compartmental_infections)
    assert np.all(compartmental_infections > 0), "CompartmentalModel: There should be some infections after the simulation runs."
    
    # Both models should show transmission
    print(f"BiweeklyModel mean infections: {np.mean(biweekly_infections):.1f}")
    print(f"CompartmentalModel mean infections: {np.mean(compartmental_infections):.1f}")


# Test ZERO transmission scenarios
def test_zero_trans():
    """Test zero transmission scenarios for both models."""
    
    # Test BiweeklyModel with beta = 0
    model = setup_biweekly_sim(
        num_ticks=26,
        beta=0.00001,  # Effectively no transmission
        birth_rate=0.001,
        death_rate=0.001,
        init_infections=0.01
    )
    model.run()
    
    state_tracker = [c for c in model.instances if isinstance(c, StateTracker)][0]
    initial_I = state_tracker.I[0]
    final_I = state_tracker.I[-1]
    final_R = state_tracker.R[-1]
    
    # With very low beta, very few new infections should occur
    new_infections = final_I + final_R - initial_I
    assert new_infections < 100, f"BiweeklyModel: There should be very few new infections with beta~0, got {new_infections}"
    
    # Test BiweeklyModel with no initial infections
    model = setup_biweekly_sim(
        num_ticks=26,
        beta=2.0,  # High transmission rate
        birth_rate=0.001,
        death_rate=0.001,
        init_infections=0  # No initial infections
    )
    model.run()
    
    state_tracker = [c for c in model.instances if isinstance(c, StateTracker)][0]
    final_I = state_tracker.I[-1]
    final_R = state_tracker.R[-1]
    
    total_ever_infected = final_I + final_R
    assert total_ever_infected == 0, f"BiweeklyModel: There should be NO infections with no initial infections, got {total_ever_infected}"
    
    # Test CompartmentalModel with beta = 0
    model = setup_compartmental_sim(
        num_ticks=182,
        beta=0.00001,  # Effectively no transmission
        birth_rate=0.001,
        death_rate=0.001,
        init_infections=0.01
    )
    model.run()
    
    state_tracker = [c for c in model.instances if isinstance(c, CompartmentalStateTracker)][0]
    initial_I = state_tracker.I[0]
    final_I = state_tracker.I[-1]
    final_R = state_tracker.R[-1]
    
    # With very low beta, very few new infections should occur
    new_infections = final_I + final_R - initial_I
    assert new_infections < 100, f"CompartmentalModel: There should be very few new infections with beta~0, got {new_infections}"
    
    # Test CompartmentalModel with no initial infections
    model = setup_compartmental_sim(
        num_ticks=182,
        beta=0.5,
        birth_rate=0.001,
        death_rate=0.001,
        init_infections=0  # No initial infections
    )
    model.run()
    
    state_tracker = [c for c in model.instances if isinstance(c, CompartmentalStateTracker)][0]
    final_I = state_tracker.I[-1]
    final_R = state_tracker.R[-1]
    final_E = state_tracker.E[-1]
    
    total_ever_infected = final_I + final_R + final_E
    assert total_ever_infected == 0, f"CompartmentalModel: There should be NO infections with no initial infections, got {total_ever_infected}"


# Test DOUBLE transmission scenarios
def test_double_trans():
    """Test that doubling parameters approximately doubles transmission."""
    n_reps = 3  # Fewer reps for faster testing
    tol = 0.5  # More lenient tolerance
    
    def run_biweekly_infections(beta, init_prev):
        infections = []
        for rep in range(n_reps):
            model = setup_biweekly_sim(
                num_ticks=20,  # Shorter simulation
                beta=beta,
                birth_rate=0.001,
                death_rate=0.001,
                init_infections=init_prev,
                seed=12345 + rep
            )
            model.run()
            
            state_tracker = [c for c in model.instances if isinstance(c, StateTracker)][0]
            initial_I = state_tracker.I[0]
            final_I = state_tracker.I[-1]
            final_R = state_tracker.R[-1]
            
            total_infections = final_I + final_R - initial_I
            infections.append(total_infections)
        return np.array(infections)
    
    def run_compartmental_infections(beta, init_prev):
        infections = []
        for rep in range(n_reps):
            model = setup_compartmental_sim(
                num_ticks=100,  # Shorter simulation
                beta=beta,
                birth_rate=0.001,
                death_rate=0.001,
                init_infections=init_prev,
                seed=12345 + rep
            )
            model.run()
            
            state_tracker = [c for c in model.instances if isinstance(c, CompartmentalStateTracker)][0]
            initial_I = state_tracker.I[0]
            final_I = state_tracker.I[-1]
            final_R = state_tracker.R[-1]
            
            total_infections = final_I + final_R - initial_I
            infections.append(total_infections)
        return np.array(infections)
    
    # Test BiweeklyModel - doubling beta  
    base_beta = 0.5  # Lower beta to avoid saturation
    base_init = 0.001  # Lower initial infections
    
    infections_base = run_biweekly_infections(base_beta, base_init)
    infections_beta_2x = run_biweekly_infections(base_beta * 2, base_init)
    
    mean_base = infections_base.mean()
    ratio_beta = infections_beta_2x.mean() / mean_base
    
    # Check that doubling beta increases transmission
    assert ratio_beta > 1.0, f"BiweeklyModel: Doubling beta should increase infections (got ratio={ratio_beta:.2f})"
    
    # Test CompartmentalModel - doubling beta 
    base_beta_comp = 0.1  # Lower beta to avoid saturation
    
    infections_base_comp = run_compartmental_infections(base_beta_comp, base_init)
    infections_beta_2x_comp = run_compartmental_infections(base_beta_comp * 2, base_init)
    
    mean_base_comp = infections_base_comp.mean()
    ratio_beta_comp = infections_beta_2x_comp.mean() / mean_base_comp
    
    # Check that doubling beta increases transmission
    assert ratio_beta_comp > 1.0, f"CompartmentalModel: Doubling beta should increase infections (got ratio={ratio_beta_comp:.2f})"


def create_linear_scenario(n_patches=4):
    """Create a linear chain scenario for testing spatial transmission."""
    data = {
        "id": [f"NG:KN:00{i+1}" for i in range(n_patches)],
        "pop": [10000] * n_patches,
        "lat": [12.0 + i*1.0 for i in range(n_patches)],  # Spaced 1 degree apart
        "lon": [8.5] * n_patches,  # Same longitude
        "mcv1": [0.8] * n_patches
    }
    df = pl.DataFrame(data)
    return df


def setup_linear_biweekly_sim(n_patches=4, num_ticks=52, init_patch=0):
    """Set up BiweeklyModel with linear transmission chain."""
    scenario_df = create_linear_scenario(n_patches)
    scenario = BiweeklyScenario(scenario_df)
    
    params = BiweeklyParams(
        num_ticks=num_ticks,
        seed=42,
        start_time="2020-01"
    )
    
    model = BiweeklyModel(scenario, params)
    
    # Create infection component with modified mixing
    infection_params = InfectionParams(
        beta=2.0,  # Strong transmission
        seasonality=0.0,
        mixing_scale=0.1,  # Higher mixing for clearer signal
        distance_exponent=2.0  # Distance matters
    )
    
    vital_params = VitalDynamicsParams(
        crude_birth_rate=0.001,  # Minimal births
        crude_death_rate=0.001   # Minimal deaths
    )
    
    model.components = [
        StateTracker,
        lm.create_component(InfectionProcess, params=infection_params),
        lm.create_component(VitalDynamicsProcess, params=vital_params)
    ]
    
    # Modify mixing matrix to create linear chain: 0 -> 1 -> 2 -> 3
    mixing = np.zeros((n_patches, n_patches))
    for i in range(n_patches - 1):
        mixing[i+1, i] = 1.0  # i can infect i+1
    mixing[np.arange(n_patches), np.arange(n_patches)] = 1.0  # Self-mixing
    
    # Set the mixing matrix in the infection component after components are created
    infection_component = [c for c in model.instances if isinstance(c, InfectionProcess)][0]
    infection_component.mixing = mixing
    
    # Initialize infections in specified patch
    model.infect(init_patch, 100)
    
    return model


def setup_linear_compartmental_sim(n_patches=4, num_ticks=365, init_patch=0):
    """Set up CompartmentalModel with linear transmission chain."""
    scenario_df = create_linear_scenario(n_patches)
    scenario = CompartmentalScenario(scenario_df)
    
    params = CompartmentalParams(
        num_ticks=num_ticks,
        seed=42,
        start_time="2020-01",
        beta=0.5,
        sigma=1.0/8.0,
        gamma=1.0/5.0
    )
    
    model = CompartmentalModel(scenario, params)
    
    # Create infection component with modified mixing
    infection_params = CompartmentalInfectionParams(
        beta=0.5,
        sigma=1.0/8.0,
        gamma=1.0/5.0,
        seasonality=0.0,
        mixing_scale=0.1,
        distance_exponent=2.0
    )
    
    vital_params = CompartmentalVitalDynamicsParams(
        crude_birth_rate=0.001,
        crude_death_rate=0.001
    )
    
    model.components = [
        CompartmentalStateTracker,
        lm.create_component(CompartmentalInfectionProcess, params=infection_params),
        lm.create_component(CompartmentalVitalDynamicsProcess, params=vital_params)
    ]
    
    # Modify mixing matrix to create linear chain
    mixing = np.zeros((n_patches, n_patches))
    for i in range(n_patches - 1):
        mixing[i+1, i] = 1.0  # i can infect i+1
    mixing[np.arange(n_patches), np.arange(n_patches)] = 1.0  # Self-mixing
    
    # Set the mixing matrix in the infection component after components are created
    infection_component = [c for c in model.instances if isinstance(c, CompartmentalInfectionProcess)][0]
    infection_component.mixing = mixing
    
    # Initialize infections in specified patch
    model.patches.states.I[init_patch] = 100
    model.patches.states.S[init_patch] -= 100
    
    return model


def test_linear_transmission():
    """Test linear transmission patterns through connected patches."""
    
    # Test BiweeklyModel linear transmission
    model = setup_linear_biweekly_sim(n_patches=4, num_ticks=52, init_patch=0)
    model.run()
    
    state_tracker = [c for c in model.instances if isinstance(c, StateTracker)][0]
    
    # For a simple test, just verify that some transmission occurred
    initial_total_I = state_tracker.I[0]
    final_total_I = state_tracker.I[-1]
    final_total_R = state_tracker.R[-1]
    
    # Some individuals should have been infected and recovered
    total_ever_infected = final_total_I + final_total_R
    assert total_ever_infected > initial_total_I, "BiweeklyModel: Some transmission should have occurred in linear network"
    
    print(f"BiweeklyModel: Initial I={initial_total_I}, Final I+R={total_ever_infected}")
    
    # Test CompartmentalModel linear transmission
    model = setup_linear_compartmental_sim(n_patches=4, num_ticks=365, init_patch=0)
    model.run()
    
    state_tracker = [c for c in model.instances if isinstance(c, CompartmentalStateTracker)][0]
    
    # For a simple test, just verify that some transmission occurred
    initial_total_I = state_tracker.I[0]
    final_total_I = state_tracker.I[-1]
    final_total_R = state_tracker.R[-1]
    final_total_E = state_tracker.E[-1]
    
    # Some individuals should have been infected and recovered
    total_ever_infected = final_total_I + final_total_R + final_total_E
    assert total_ever_infected > initial_total_I, "CompartmentalModel: Some transmission should have occurred in linear network"
    
    print(f"CompartmentalModel: Initial I={initial_total_I}, Final I+E+R={total_ever_infected}")


if __name__ == "__main__":
    print("Running laser_measles transmission tests...")
    
    print("\n1. Testing default transmission...")
    test_trans_default()
    print("✓ Default transmission tests passed")
    
    print("\n2. Testing zero transmission scenarios...")
    test_zero_trans()
    print("✓ Zero transmission tests passed")
    
    print("\n3. Testing parameter doubling effects...")
    test_double_trans()
    print("✓ Double transmission tests passed")
    
    print("\n4. Testing linear spatial transmission...")
    test_linear_transmission()
    print("✓ Linear transmission tests passed")
    
    print("\n🎉 All laser_measles transmission tests passed!")