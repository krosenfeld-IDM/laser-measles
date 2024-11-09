import re
from pathlib import Path

import numpy as np
from laser_core.propertyset import PropertySet


def get_parameters(nticks, verbose, kwargs) -> PropertySet:
    meta_params = PropertySet(
        {
            "nticks": nticks,
            "verbose": verbose,
            "cbr": np.float32(35.0),
            "pyramid_file": Path(__file__).parent / "nigeria_pyramid.csv",
        }
    )

    measles_params = PropertySet(
        {
            "exp_scale": np.float32(1.0),
            "exp_shape": np.float32(3.5),
            "inf_mean": np.float32(18.0),
            "inf_std": np.float32(2.0),
            "r_naught": np.float32(15.0),
            "seasonality_factor": np.float32(0.125),
            "seasonality_phase": np.float32(182),
        }
    )

    network_params = PropertySet(
        {
            "k": np.float32(50.0),
            "a": np.float32(1.0),
            "b": np.float32(0.0),
            "c": np.float32(1.0),
            "max_frac": np.float32(0.05),
        }
    )

    ri_params = PropertySet(
        {
            "ri_coverage": np.float32(0.7),
            "mcv1_start": int(8.5 * 365 / 12),  # 8.5 months
            "mcv1_end": int(9.5 * 365 / 12),  # 9.5 months
            "mcv2_start": int(14.5 * 365 / 12),  # 14.5 months
            "mcv2_end": int(15.5 * 365 / 12),  # 15.5 months
            "probability_mcv1_take": np.float32(0.85),
            "probability_mcv2_take": np.float32(0.95),
        }
    )

    params = PropertySet(meta_params, measles_params, network_params, ri_params)

    # Second, optionally, load parameters from a JSON file
    if kwargs.get("params") is not None:
        paramfile = Path(kwargs.get("params"))
        params += PropertySet.load(paramfile)
        print(f"Loaded parameters from `{paramfile}`…")

    # Third, optionally, override parameters from the command line
    if kwargs.get("param") is not None:
        for kvp in kwargs.get("param"):
            key, value = re.split("[=:]+", kvp)
            if key not in params:
                raise ValueError(f"Unknown parameter: {key}")
            value = type(params[key])(value)  # convert from string to existing type
            params[key] = value
            print(f"Overriding parameter `{key}` with `{value}` ({type(value)})…")

    params.beta = params.r_naught / params.inf_mean  # type: ignore

    return params
