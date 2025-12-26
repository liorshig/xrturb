import itertools
from typing import List, Optional
import re
import numpy as np
import xarray as xr


def _weighted_mean(da: xr.DataArray, weights: Optional[xr.DataArray], dim: str = 'time') -> xr.DataArray:
    """Internal helper for weighted averaging."""
    if weights is not None:
        # Weighted Mean = Sum(x*w) / Sum(w)
        return da.weighted(weights).mean(dim=dim)
    return da.mean(dim=dim)


def mean(ds: xr.Dataset, weights: Optional[xr.DataArray] = None, dim: str = 'time') -> xr.Dataset:
    """Returns the (weighted) mean flow field."""
    if weights is not None:
        # We calculate weighted mean for all data variables
        ds_avg = ds.map(lambda x: _weighted_mean(x, weights, dim) if x.name != weights.name else x)
        return ds_avg
    return ds.mean(dim=dim)


def fluct(ds: xr.Dataset, var_key: str, weights: Optional[xr.DataArray] = None, dim: str = 'time') -> xr.DataArray:
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


def calculate_product(ds: xr.Dataset, product_key: str, weights: Optional[xr.DataArray] = None, dim: str = 'time') -> xr.DataArray:
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
    dim : str, default 'time'
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
    if 'time' not in ds.coords:
        raise ValueError("Time dimension 'time' required for Reynolds Stress.")

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
        rss = _weighted_mean(product, weights, dim='time')
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

    if 'time' not in ds.coords:
        # Fallback for single frame (Instantaneous KE)
        ke = 0.5 * sum(ds[comp]**2 for comp in velocity_components)
        ke.attrs["standard_name"] = "kinetic_energy"
        return ke

    # Time-averaged TKE
    tke_sum = 0
    for comp in velocity_components:
        var_key = f"{comp}'{comp}'"
        var = _weighted_mean(calculate_product(ds, var_key, weights), weights, dim='time')
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
            ds_moments[product_key] = _weighted_mean(prod, weights, dim='time')
    return ds_moments


def _calc_standardized_moments(ds: xr.Dataset, variables: List[str], order: int, weights: Optional[xr.DataArray]) -> xr.Dataset:
    """Calculate standardized central moments (normalized by std)."""
    # Calculate standard deviations for normalization
    stds = {}
    for var in variables:
        var_sq = _weighted_mean(calculate_product(ds, f"{var}'{var}'", weights), weights, dim='time')
        stds[var] = var_sq ** 0.5

    ds_moments = xr.Dataset()
    for n in range(2, order + 1):
        for combination in itertools.combinations_with_replacement(variables, n):
            name = "".join(combination) 
            # Calculate raw moment
            product_key = ''.join(var + "'" for var in combination)
            prod = calculate_product(ds, product_key, weights=weights)
            raw_moment = _weighted_mean(prod, weights, dim='time')
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


def calc_all_products(ds: xr.Dataset, variables: Optional[List[str]] = None, order: int = 2, weights: Optional[xr.DataArray] = None, dim: str = 'time') -> xr.Dataset:
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
    dim : str, default 'time'
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



def two_point_correlation(
    data: xr.Dataset, 
    var_name: str = 'u', 
    x_ref: Optional[float] = None, 
    y_ref: Optional[float] = None,
    x_dim: str = 'x',
    y_dim: str = 'y'
) -> xr.DataArray:
    """
    Calculates the 2-point spatial correlation for a specific variable 
    relative to a reference point, with customizable coordinate names.
    
    Parameters
    ----------
    ...
    x_dim : str, optional
        The name of the x-coordinate dimension (default 'x').
    y_dim : str, optional
        The name of the y-coordinate dimension (default 'y').
    """
    
    # 1. Validation
    if var_name not in data:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")
    if x_dim not in data.dims or y_dim not in data.dims:
        raise ValueError(f"Dimensions '{x_dim}' or '{y_dim}' not found in dataset.")

    da = data[var_name]

    if 'time' not in da.dims:
        raise ValueError("Dataset is missing the 'time' dimension.")

    # 2. Determine reference point using the dynamic dimension names
    if x_ref is None:
        x_ref = float(data[x_dim][0].values)
    if y_ref is None:
        y_ref = float(data[y_dim][0].values)

    # 3. Reynolds Decomposition (Fluctuations)
    # Using 'time' as a hardcoded dimension for ensemble averaging
    u_mean = da.mean(dim='time')
    u_prime = da - u_mean
    
    # Handle NaNs - interpolate along the user-defined x_dim
    u_prime_vals = u_prime
    
    # 4. Selection and Correlation
    # We use a dictionary for .sel() to handle dynamic dimension names
    selection_dict = {x_dim: x_ref, y_dim: y_ref}
    ref_point = u_prime.sel(selection_dict, method='nearest')
    
    # Calculation
    corr = (u_prime_vals * ref_point).mean(dim='time')
    u_variance = u_prime_vals.var(dim='time')
    ref_point_variance = ref_point.var(dim='time')
    
    corr_coeff = corr / np.sqrt(u_variance * ref_point_variance)
    
    # Optional: Update the output name to reflect the correlation
    corr_coeff.name = f"R_{var_name}{var_name}"
    
    return corr_coeff
