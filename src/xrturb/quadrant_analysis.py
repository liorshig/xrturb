import xarray as xr
import numpy as np
from typing import Optional, Union, Tuple
from .statistics import fluct, _weighted_mean


def _quadrants(ds: xr.Dataset, x_var: str = 'u', y_var: str = 'w',
               weights: Optional[str] = None, dim: str = 'time') -> xr.Dataset:
    """
    Classify data points into quadrants based on fluctuation signs.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    x_var, y_var : str
        Variable names for quadrant classification
    weights : str, optional
        Weighting variable for mean calculation
    dim : str
        Dimension along which to calculate means

    Returns
    -------
    xr.Dataset
        Dataset with quadrant coordinate added
    """
    # Calculate fluctuations using proper weighted means
    u_prime = fluct(ds, x_var, weights=weights, dim=dim)
    w_prime = fluct(ds, y_var, weights=weights, dim=dim)

    quadrant = xr.where(
        (u_prime > 0) & (w_prime > 0), 1,
        xr.where(
            (u_prime < 0) & (w_prime > 0), 2,
            xr.where(
                (u_prime < 0) & (w_prime < 0), 3,
                xr.where((u_prime > 0) & (w_prime < 0), 4, 0)
            )
        )
    )
    ds = ds.assign_coords(quadrantime=quadrant)
    return ds


def _octants(ds: xr.Dataset, x_var: str = 'u', y_var: str = 'w', z_var: str = 'c',
             weights: Optional[str] = None, dim: str = 'time') -> xr.Dataset:
    """
    Classify data points into octants based on fluctuation signs.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    x_var, y_var, z_var : str
        Variable names for octant classification
    weights : str, optional
        Weighting variable for mean calculation
    dim : str
        Dimension along which to calculate means

    Returns
    -------
    xr.Dataset
        Dataset with quadrant coordinate added (containing octant numbers)
    """
    # Calculate fluctuations using proper weighted means
    u_prime = fluct(ds, x_var, weights=weights, dim=dim)
    w_prime = fluct(ds, y_var, weights=weights, dim=dim)
    c_prime = fluct(ds, z_var, weights=weights, dim=dim)

    octant = xr.where((u_prime > 0) & (w_prime > 0) & (c_prime > 0), 1, 0) + \
        xr.where((u_prime < 0) & (w_prime > 0) & (c_prime > 0), 2, 0) + \
        xr.where((u_prime < 0) & (w_prime > 0) & (c_prime < 0), 3, 0) + \
        xr.where((u_prime > 0) & (w_prime > 0) & (c_prime < 0), 4, 0) + \
        xr.where((u_prime > 0) & (w_prime < 0) & (c_prime > 0), 5, 0) + \
        xr.where((u_prime < 0) & (w_prime < 0) & (c_prime > 0), 6, 0) + \
        xr.where((u_prime < 0) & (w_prime < 0) & (c_prime < 0), 7, 0) + \
        xr.where((u_prime > 0) & (w_prime < 0) & (c_prime < 0), 8, 0)

    ds = ds.assign_coords(quadrantime=octant)
    return ds


def _hole_quadrants(ds: xr.Dataset, x_var: str = 'u', y_var: str = 'w',
                    hole_size: float = 0, weights: Optional[str] = None,
                    dim: str = 'time') -> xr.Dataset:
    """
    Apply hole filtering to quadrant analysis.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    x_var, y_var : str
        Variables for hole calculation (typically u and w)
    hole_size : float
        Hole size parameter (0 = no hole, larger values filter more)
    weights : str, optional
        Weighting variable for mean calculation
    dim : str
        Dimension along which to calculate means

    Returns
    -------
    xr.Dataset
        Dataset with hole coordinate added
    """
    # Calculate fluctuations
    u_prime = fluct(ds, x_var, weights=weights, dim=dim)
    w_prime = fluct(ds, y_var, weights=weights, dim=dim)

    # Calculate Reynolds stress and its magnitude threshold
    upwp = u_prime * w_prime

    # Use weighted mean for hole threshold calculation
    upwp_mean = _weighted_mean(upwp, weights=weights, dim=dim)

    hole_mask = xr.where(np.abs(upwp) > hole_size * np.abs(upwp_mean), 1, 0)
    ds = ds.assign_coords(hole=hole_mask)
    return ds



def add_quadrants(ds: xr.Dataset, x_var: str = 'u', y_var: str = 'w', z_var: str = 'c',
                  analysis_type: str = 'quadrant', hole_size: float = 0,
                  hole_vars: Optional[Tuple[str, str]] = None,
                  weights: Optional[str] = None, dim: str = 'time') -> xr.Dataset:
    """
    Add quadrant/octant classification and hole filtering to dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    x_var, y_var : str
        Primary variables for classification
    z_var : str
        Third variable for octant analysis
    analysis_type : str
        'quadrant' or 'octant'
    hole_size : float
        Hole size parameter for filtering
    hole_vars : tuple of str, optional
        Variables for hole calculation (defaults to x_var, y_var)
    weights : str, optional
        Weighting variable for mean calculations
    dim : str
        Dimension for averaging

    Returns
    -------
    xr.Dataset
        Dataset with quadrant and hole coordinates
    """
    if analysis_type not in ['quadrant', 'octant']:
        raise ValueError(f"analysis_type must be 'quadrant' or 'octant', got '{analysis_type}'")

    if analysis_type == 'quadrant':
        ds = _quadrants(ds, x_var, y_var, weights=weights, dim=dim)
    elif analysis_type == 'octant':
        ds = _octants(ds, x_var, y_var, z_var, weights=weights, dim=dim)

    # Apply hole filtering
    hole_x, hole_y = hole_vars if hole_vars is not None else (x_var, y_var)
    ds = _hole_quadrants(ds, hole_x, hole_y, hole_size, weights=weights, dim=dim)

    return ds


def calc_quadrant_mean(ds: xr.Dataset, analysis_type: str = 'quadrant',
                       x_var: str = 'u', y_var: str = 'w', z_var: str = 'c',
                       hole_size: float = 0, hole_vars: Optional[Tuple[str, str]] = None,
                       weights: Optional[str] = None, dim: str = 'time',
                       time_var: str = 'transit_time') -> xr.Dataset:
    """
    Calculate conditional averages for each quadrant/octant.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    analysis_type : str
        'quadrant' or 'octant'
    x_var, y_var, z_var : str
        Variable names
    hole_size : float
        Hole size parameter
    hole_vars : tuple, optional
        Variables for hole calculation
    weights : str, optional
        Weighting variable for fluctuations
    dim : str
        Averaging dimension
    time_var : str
        Time weighting variable (default: 'transit_time')

    Returns
    -------
    xr.Dataset
        Conditional means for each quadrant
    """
    if time_var not in ds.data_vars and time_var not in ds.coords:
        # Fallback: assume uniform time weighting
        ds = ds.assign({time_var: xr.ones_like(ds[dim])})

    ds_classified = add_quadrants(ds, x_var, y_var, z_var, analysis_type,
                                  hole_size, hole_vars, weights, dim)

    # Filter by hole
    ds_filtered = ds_classified.where(ds_classified.hole == 1, drop=True)

    quadrants_lst = []
    for q in [1, 2, 3, 4] if analysis_type == 'quadrant' else list(range(1, 9)):
        # Select data for this quadrant
        q_mask = ds_filtered.quadrant == q
        q_ds = ds_filtered.where(q_mask, drop=True)

        # Only process if there are valid data points
        if len(q_ds[dim]) > 0:
            # Calculate weighted mean along the specified dimension
            q_mean = q_ds.weighted(q_ds[time_var]).mean(dim=dim)
            q_mean = q_mean.assign_coords(quadrantime=q)
            quadrants_lst.append(q_mean)

    if not quadrants_lst:
        raise ValueError("No valid quadrant data found after filtering")

    quadrant_mean = xr.concat(quadrants_lst, dim='quadrant')
    return quadrant_mean


def calc_quadrant_fraction(ds: xr.Dataset, analysis_type: str = 'quadrant',
                          x_var: str = 'u', y_var: str = 'w', z_var: str = 'c',
                          hole_size: float = 0, hole_vars: Optional[Tuple[str, str]] = None,
                          weights: Optional[str] = None, dim: str = 'time',
                          time_var: str = 'transit_time') -> xr.Dataset:
    """
    Calculate the fraction of total contribution from each quadrant.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    analysis_type : str
        'quadrant' or 'octant'
    x_var, y_var, z_var : str
        Variable names
    hole_size : float
        Hole size parameter
    hole_vars : tuple, optional
        Variables for hole calculation
    weights : str, optional
        Weighting variable for fluctuations
    dim : str
        Averaging dimension
    time_var : str
        Time weighting variable

    Returns
    -------
    xr.Dataset
        Contribution fractions for each quadrant
    """
    if time_var not in ds.data_vars and time_var not in ds.coords:
        ds = ds.assign({time_var: xr.ones_like(ds[dim])})

    ds_classified = add_quadrants(ds, x_var, y_var, z_var, analysis_type,
                                  hole_size, hole_vars, weights, dim)

    # Calculate weighted sums for each quadrant
    quadrants_lst = []
    for q in [1, 2, 3, 4] if analysis_type == 'quadrant' else list(range(1, 9)):
        # Select data for this quadrant and apply hole filter
        q_mask = (ds_classified.quadrant == q) & (ds_classified.hole == 1)
        q_ds = ds_classified.where(q_mask, drop=True)

        if len(q_ds[dim]) > 0:
            q_sum = q_ds.weighted(q_ds[time_var]).sum(dim=dim)
            q_sum = q_sum.assign_coords(quadrantime=q)
            quadrants_lst.append(q_sum)

    total_sum = ds_classified.where(ds_classified.hole == 1, drop=True).weighted(ds_classified[time_var]).sum(dim=dim)
    quadrant_sum = xr.concat(quadrants_lst, dim='quadrant')
    quadrant_fraction = quadrant_sum / total_sum

    return quadrant_fraction


def calc_quadrant_contribution(ds: xr.Dataset, analysis_type: str = 'quadrant',
                              x_var: str = 'u', y_var: str = 'w', z_var: str = 'c',
                              hole_size: float = 0, hole_vars: Optional[Tuple[str, str]] = None,
                              weights: Optional[str] = None, dim: str = 'time',
                              time_var: str = 'transit_time') -> xr.Dataset:
    """
    Calculate the contribution of each quadrant to total Reynolds stress.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    analysis_type : str
        'quadrant' or 'octant'
    x_var, y_var, z_var : str
        Variable names
    hole_size : float
        Hole size parameter
    hole_vars : tuple, optional
        Variables for hole calculation
    weights : str, optional
        Weighting variable for fluctuations
    dim : str
        Averaging dimension
    time_var : str
        Time weighting variable

    Returns
    -------
    xr.Dataset
        Contribution of each quadrant to total Reynolds stress
    """
    if time_var not in ds.data_vars and time_var not in ds.coords:
        ds = ds.assign({time_var: xr.ones_like(ds[dim])})

    ds_classified = add_quadrants(ds, x_var, y_var, z_var, analysis_type,
                                  hole_size, hole_vars, weights, dim)

    # Calculate weighted sums for each quadrant
    quadrants_lst = []
    for q in [1, 2, 3, 4] if analysis_type == 'quadrant' else list(range(1, 9)):
        # Select data for this quadrant and apply hole filter
        q_mask = (ds_classified.quadrant == q) & (ds_classified.hole == 1)
        q_ds = ds_classified.where(q_mask, drop=True)

        if len(q_ds[dim]) > 0:
            q_sum = q_ds.weighted(q_ds[time_var]).sum(dim=dim)
            q_sum = q_sum.assign_coords(quadrantime=q)
            quadrants_lst.append(q_sum)

    total_time = ds_classified.where(ds_classified.hole == 1, drop=True)[time_var].sum(dim=dim)
    quadrant_sum = xr.concat(quadrants_lst, dim='quadrant')
    quadrant_contribution = quadrant_sum / total_time

    return quadrant_contribution


def calc_quadrant_duration(ds: xr.Dataset, analysis_type: str = 'quadrant',
                          x_var: str = 'u', y_var: str = 'w', z_var: str = 'c',
                          hole_size: float = 0, hole_vars: Optional[Tuple[str, str]] = None,
                          weights: Optional[str] = None, dim: str = 'time',
                          time_var: str = 'transit_time') -> xr.Dataset:
    """
    Calculate the time duration spent in each quadrant.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    analysis_type : str
        'quadrant' or 'octant'
    x_var, y_var, z_var : str
        Variable names
    hole_size : float
        Hole size parameter
    hole_vars : tuple, optional
        Variables for hole calculation
    weights : str, optional
        Weighting variable for fluctuations
    dim : str
        Averaging dimension
    time_var : str
        Time weighting variable

    Returns
    -------
    xr.Dataset
        Time fractions for each quadrant
    """
    if time_var not in ds.data_vars and time_var not in ds.coords:
        ds = ds.assign({time_var: xr.ones_like(ds[dim])})

    ds_classified = add_quadrants(ds, x_var, y_var, z_var, analysis_type,
                                  hole_size, hole_vars, weights, dim)

    # Calculate time spent in each quadrant (accounting for holes)
    duration_lst = []
    total_time = ds_classified.where(ds_classified.hole == 1, drop=True)[time_var].sum(dim=dim)

    for q in [1, 2, 3, 4] if analysis_type == 'quadrant' else list(range(1, 9)):
        # Select data for this quadrant and apply hole filter
        q_mask = (ds_classified.quadrant == q) & (ds_classified.hole == 1)
        q_ds = ds_classified.where(q_mask, drop=True)

        if len(q_ds[dim]) > 0:
            q_time = q_ds[time_var].sum(dim=dim)
            q_time = q_time.assign_coords(quadrantime=q)
            duration_lst.append(q_time)

    quadrant_duration = xr.concat(duration_lst, dim='quadrant') / total_time
    return quadrant_duration


