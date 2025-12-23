import itertools
from typing import List, Optional
import re
import numpy as np
import xarray as xr


def _weighted_mean(da: xr.DataArray, weights: Optional[xr.DataArray], dim: str = 't') -> xr.DataArray:
    """Internal helper for weighted averaging."""
    if weights is not None:
        # Weighted Mean = Sum(x*w) / Sum(w)
        return da.weighted(weights).mean(dim=dim)
    return da.mean(dim=dim)


def mean(ds: xr.Dataset, weights: Optional[xr.DataArray] = None, dim: str = 't') -> xr.Dataset:
    """Returns the (weighted) mean flow field."""
    if weights is not None:
        # We calculate weighted mean for all data variables
        ds_avg = ds.map(lambda x: _weighted_mean(x, weights, dim) if x.name != weights.name else x)
        return ds_avg
    return ds.mean(dim=dim)


def fluct(ds: xr.Dataset, var_key: str, weights: Optional[xr.DataArray] = None, dim: str = 't') -> xr.DataArray:
    """
    Calculate fluctuations: u' = u - mean(u)
    Supports weighted means for LDV.
    """
    if var_key not in ds:
        raise ValueError(f"Variable '{var_key}' not found.")
    
    mean_val = _weighted_mean(ds[var_key], weights, dim)
    fluctuation = ds[var_key] - mean_val
    fluctuation.name = f"{var_key}'"
    return fluctuation


def calculate_product(ds: xr.Dataset, product_key: str, weights: Optional[xr.DataArray] = None, dim: str = 't') -> xr.DataArray:
    """
    Calculates the element-wise product of variables (raw or fluctuating).
    Does NOT average the result.

    Parsing Logic:
    - "uv"   -> ds['u'] * ds['v']
    - "u'v'" -> fluct('u') * fluct('v')
    - "u'v"  -> fluct('u') * ds['v']

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    product_key : str
        String defining the product (e.g., "u'v'", "uv").
    weights : xr.DataArray, optional
        Weighting variable (needed only if calculating fluctuations).
    dim : str, default 't'
        Dimension used to calculate the mean for fluctuations.

    Returns
    -------
    xr.DataArray
        The product series (same shape as input variables).
    """

    # Parse the key using Regex
    # Captures pairs like ('u', "'") or ('v', '')
    tokens = re.findall(r"([a-zA-Z0-9])('?)", product_key)

    if not tokens:
        raise ValueError(f"Could not parse product key: {product_key}")
    combined_product = 1.0

    # Iterate through variables and multiply
    for var_name, prime_tag in tokens:
        
        # Handle Aliases
        if var_name not in ds:
            if var_name == 'u' and 'U' in ds: var_name = 'U'
            elif var_name == 'v' and 'V' in ds: var_name = 'V'
            elif var_name == 'w' and 'W' in ds: var_name = 'W'
            else:
                raise ValueError(f"Variable '{var_name}' not found in dataset.")

        # Check for Prime (') to decide between Raw vs. Fluctuation
        if prime_tag == "'":
            # Pass weights/dim down so the fluctuation is calculated relative
            # to the correct (weighted) mean
            term = fluct(ds, var_name, weights=weights, dim=dim)
        else:
            term = ds[var_name]

        combined_product = combined_product * term

    combined_product.name = product_key
    return combined_product


def reynolds_stress(ds: xr.Dataset, weights: Optional[xr.DataArray] = None) -> xr.Dataset:
    """
    Calculates Reynolds Shear Stress components: -rho * <u_i'u_j'>
    (Here returning just <u_i'u_j'> to be density independent).
    
    Automatically detects available velocity components (u, v, w) and computes
    all possible shear stress components:
    - For 3D data (u,v,w): uv, uw, vw
    - For 2D data (u,v): uv
    """
    if 't' not in ds.coords:
        raise ValueError("Time dimension 't' required for Reynolds Stress.")

    # Detect available velocity components
    velocity_components = []
    for comp in ['u', 'v', 'w']:
        if comp in ds.data_vars:
            velocity_components.append(comp)

    if len(velocity_components) < 2:
        raise ValueError("At least 2 velocity components required for Reynolds stress calculation.")

    # Generate all unique pairs
    rss_components = {}
    for i, j in itertools.combinations(velocity_components, 2):
        product_key = f"{i}'{j}'"
        product = calculate_product(ds, product_key, weights)
        rss = _weighted_mean(product, weights, dim='t')
        rss.name = f"{i}'{j}'"
        rss.attrs['standard_name'] = f"Reynolds_stress_{i}'{j}'"
        rss_components[product_key] = rss

    return xr.Dataset(rss_components)


def tke(ds: xr.Dataset, weights: Optional[xr.DataArray] = None) -> xr.DataArray:
    """
    Calculates Turbulent Kinetic Energy: 0.5 * sum(<comp'comp'> for available velocity components)

    Automatically detects available velocity components (u, v, w) and computes TKE
    based on what's present in the dataset.
    """
    # Detect available velocity components
    velocity_components = [comp for comp in ['u', 'v', 'w'] if comp in ds.data_vars]

    if not velocity_components:
        raise ValueError("No velocity components (u, v, w) found in dataset.")

    if 't' not in ds.coords:
        # Fallback for single frame (Instantaneous KE)
        ke = 0.5 * sum(ds[comp]**2 for comp in velocity_components)
        ke.attrs["standard_name"] = "kinetic_energy"
        return ke

    # Time-averaged TKE
    tke_sum = 0
    for comp in velocity_components:
        var_key = f"{comp}'{comp}'"
        var = _weighted_mean(calculate_product(ds, var_key, weights), weights, dim='t')
        tke_sum += var
    
    k = 0.5 * tke_sum # noqa: E741
    k.attrs["standard_name"] = "TKE" # noqa: E741
    k.attrs["units"] = "m^2/s^2" # noqa: E741

    if len(velocity_components) < 3: # noqa: E741
        k.attrs['note'] = f"TKE calculated with {len(velocity_components)} velocity components: {', '.join(velocity_components)}"
    
    return k


def _calc_central_moments(ds: xr.Dataset, variables: List[str], order: int, weights: Optional[xr.DataArray]) -> xr.Dataset:
    """Calculate raw central moments (unstandardized)."""
    ds_moments = xr.Dataset()
    for n in range(2, order + 1):
        for combination in itertools.combinations_with_replacement(variables, n): # noqa: B007
            name = "".join(combination) 
            # Calculate product of fluctuations
            product_key = ''.join(var + "'" for var in combination)
            prod = calculate_product(ds, product_key, weights=weights)
            # Average it
            ds_moments[product_key] = _weighted_mean(prod, weights, dim='t')
    return ds_moments


def _calc_standardized_moments(ds: xr.Dataset, variables: List[str], order: int, weights: Optional[xr.DataArray]) -> xr.Dataset:
    """Calculate standardized central moments (normalized by std)."""
    # Calculate standard deviations for normalization
    stds = {}
    for var in variables:
        var_sq = _weighted_mean(calculate_product(ds, f"{var}'{var}'", weights), weights, dim='t')
        stds[var] = var_sq ** 0.5

    ds_moments = xr.Dataset()
    for n in range(2, order + 1):
        for combination in itertools.combinations_with_replacement(variables, n):
            name = "".join(combination) 
            # Calculate raw moment
            product_key = ''.join(var + "'" for var in combination)
            prod = calculate_product(ds, product_key, weights=weights)
            raw_moment = _weighted_mean(prod, weights, dim='t')
            # Normalize by product of stds
            norm_factor = 1.0
            for var in combination:
                norm_factor *= stds[var]
            standardized_moment = raw_moment / norm_factor
            ds_moments['M' + product_key] = standardized_moment

    return ds_moments


def calc_moments(ds: xr.Dataset, variables: Optional[List[str]] = None, order: int = 2, weights: Optional[xr.DataArray] = None, standardized: bool = False) -> xr.Dataset:
    """
    Calculate high-order moments for given variables.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    variables : list of str, optional
        Variables to compute moments for. If None, detects available velocity components.
    order : int, default 2
        Maximum order of moments to compute
    weights : xr.DataArray, optional
        Weighting variable for averaging
    standardized : bool, default False
        If True, compute standardized moments (normalized by std).
        If False, compute raw central moments.
        
    Returns
    -------
    xr.Dataset
        Dataset containing moments
    """
    if variables is None:
        # Detect available velocity components
        variables = [comp for comp in ['u', 'v', 'w'] if comp in ds.data_vars or comp in ds.coords]
        if not variables: # noqa: E741
            raise ValueError("No velocity components found in dataset.")

    if standardized:
        return _calc_standardized_moments(ds, variables, order, weights)
    else:
        return _calc_central_moments(ds, variables, order, weights)


def calc_all_products(ds: xr.Dataset, variables: Optional[List[str]] = None, order: int = 2, weights: Optional[xr.DataArray] = None, dim: str = 't') -> xr.Dataset:
    """
    Calculate all fluctuation products up to specified order and add them to the dataset.
    
    Computes the element-wise products of fluctuations and adds them as new variables
    to the input dataset. This is useful for computing raw products before statistical averaging.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    variables : list of str, optional
        Variables to compute products for. If None, detects available velocity components.
    order : int, default 2
        Maximum order of products to compute
    weights : xr.DataArray, optional
        Weighting variable for mean calculation in fluctuations
    dim : str, default 't'
        Dimension along which to calculate means for fluctuations
        
    Returns
    -------
    xr.Dataset
        Input dataset with additional product variables
    """
    if variables is None:
        # Detect available velocity components
        variables = [comp for comp in ['u', 'v', 'w'] if comp in ds.data_vars or comp in ds.coords]
        if not variables: # noqa: E741
            raise ValueError("No velocity components found in dataset.")

    ds_out = ds.copy()
    for n in range(2, order + 1):
        for combination in itertools.combinations_with_replacement(variables, n):
            name = "".join(combination) 
            # Calculate product of fluctuations (without averaging)
            product_key = ''.join(var + "'" for var in combination)
            product = calculate_product(ds, product_key, weights=weights, dim=dim)
            ds_out[product_key] = product

    return ds_out



def two_point_correlation(data: xr.Dataset, var_name: str = 'u', x_ref: Optional[float] = None, y_ref: Optional[float] = None) -> xr.DataArray:
    """
    Calculates the 2-point spatial correlation for a specific variable 
    in a pivpy xarray Dataset, relative to a reference point.
    
    The correlation is calculated as:
    R_uu(dx, dy) = < u'(x_ref,y_ref,t) * u'(x_ref+dx, y_ref+dy, t) > / <u'^2>

    Parameters
    ----------
    data : xarray.Dataset
        The PIV dataset (standard pivpy format with t, y, x dimensions).
    var_name : str, optional
        The variable to correlate (default 'u').
    x_ref : float, optional
        The x-coordinate of the reference point. Default is data.x[0].
    y_ref : float, optional
        The y-coordinate of the reference point. Default is data.y[0].

    Returns
    -------
    corr_xr : xarray.DataArray
        A DataArray containing the 2D correlation map.
        Coordinates are lag_x and lag_y, centered at the reference point.
    """
    
    # 1. Validation and Setup
    if var_name not in data:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")

    # Extract the DataArray for the specific variable
    da = data[var_name]

    # Ensure dimensions are in expected order
    if 't' not in da.dims:
        raise ValueError("Dataset is missing the 't' (time) dimension.")

    # 2. Determine reference point
    if x_ref is None:
        x_ref = float(data.x[0].values)
    if y_ref is None:
        y_ref = float(data.y[0].values)
    # 3. Reynolds Decomposition (Fluctuations)
    u_mean = da.mean(dim='t').compute()
    u_prime = (da - u_mean).compute()
    # Handle NaNs
    u_prime_vals = u_prime.interpolate_na(dim='x', method='linear').fillna(0.0)
    ref_point = u_prime.sel(x=x_ref, y=y_ref, method='nearest').compute()
    corr = (u_prime_vals * ref_point).mean(dim='t').compute()
    u_variance = u_prime_vals.var(dim='t').compute()
    ref_point_variance = ref_point.var(dim='t').compute()
    corr_coeff = corr / np.sqrt(u_variance * ref_point_variance)
    return corr_coeff
