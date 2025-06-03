import numpy as np
from typing import Union, Tuple

# Constants for gravity diffusion model
MAX_DISTANCE = 100000000  # km, used to prevent self-migration
MIN_DISTANCE = 10  # km, minimum distance to prevent excessive neighbor migration

def pairwise_haversine(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances for all (lon, lat) points using the Haversine formula.
    
    Args:
        lon: Array of longitude values in degrees
        lat: Array of latitude values in degrees
        
    Returns:
        Matrix of pairwise distances in kilometers
    """
    earth_radius_km = 6367


    # matrices of pairwise differences for latitudes & longitudes
    dlat = lat[:, None] - lat
    dlon = lon[:, None] - lon

    # vectorized haversine distance calculation
    d = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(d))


def init_gravity_diffusion(
    df: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
    scale: float, 
    dist_exp: float
) -> np.ndarray:
    """Initialize a gravity diffusion matrix for population mixing.
    
    Args:
        df: Either a DataFrame with 'population', 'latitude', and 'longitude' columns,
            or a tuple of (lon, lat) arrays
        scale: Scaling factor for the diffusion matrix
        dist_exp: Distance exponent for the gravity model
        
    Returns:
        Normalized diffusion matrix where each row sums to 1
    """
    if len(df) == 1:
        return np.ones((1, 1))

    distances = pairwise_haversine(df['lon'].to_numpy(), df['lat'].to_numpy())

    pops = np.array(df['population'])
    pops = pops[:, np.newaxis].T
    pops = np.repeat(pops, pops.size, axis=0).astype(np.float64)

    np.fill_diagonal(distances, MAX_DISTANCE)  # Prevent divide by zero errors and self migration
    diffusion_matrix = pops / (distances + MIN_DISTANCE) ** dist_exp  # minimum distance prevents excessive neighbor migration
    np.fill_diagonal(diffusion_matrix, 0)

    # normalize average total outbound migration to 1
    diffusion_matrix = diffusion_matrix / np.mean(np.sum(diffusion_matrix, axis=1))

    diffusion_matrix *= scale
    diagonal = 1 - np.sum(diffusion_matrix, axis=1)  # normalized outbound migration by source
    np.fill_diagonal(diffusion_matrix, diagonal)

    return diffusion_matrix