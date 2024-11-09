import re
from pathlib import Path

import click
import numpy as np
from laser_core.propertyset import PropertySet


def get_parameters(nticks, verbose, kwargs) -> PropertySet:
    meta_params = PropertySet(
        {
            "nticks": nticks,
            "verbose": verbose,
            "cbr": np.float32(13.7),
            "pyramid_file": Path(__file__).parent / "USA pyramid-2000.csv",
            "mortality_file": Path(__file__).parent / "USA mortality-2000.csv",
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

    params = PropertySet(meta_params, measles_params, network_params)

    # Overwrite any default parameters with those from a JSON file (optional)
    if kwargs.get("params") is not None:
        paramfile = Path(kwargs.get("params"))
        params += PropertySet.load(paramfile)
        click.echo(f"Loaded parameters from `{paramfile}`…")

    # Finally, overwrite any parameters with those from the command line (optional)
    if "param" in kwargs:
        for kvp in kwargs["param"]:
            key, value = re.split("[=:]+", kvp)
            if key not in params:
                click.echo(f"Unknown parameter `{key}`. Skipping…")
                continue
            value = type(params[key])(value)  # Cast the value to the same type as the existing parameter
            params[key] = value
            click.echo(f"Using `{value}` for parameter `{key}` from the command line…")

    params.beta = np.float32(np.float32(params.r_naught) / np.float32(params.inf_mean))

    return params
