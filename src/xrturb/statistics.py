import itertools
from typing import List, Optional, Tuple
import re
import warnings
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit


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
    y_dim: str = 'y',
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
    if x_ref is None and y_ref is None:
        x_ref = float(data[x_dim][0].values)
        y_ref = float(data[y_dim][0].values)
        selection_dict = { x_dim: x_ref, y_dim: y_ref}

    elif x_ref is None or y_ref is None:
        if x_ref is None:
            selection_dict = { y_dim: y_ref}
        else:
            selection_dict = { x_dim: x_ref}
    else:
        selection_dict = { x_dim: x_ref, y_dim: y_ref}
    # 3. Reynolds Decomposition (Fluctuations)
    # Using 'time' as a hardcoded dimension for ensemble averaging
    u_prime_vals = data.turb.fluct(var_name, dim='time')
    
    # 4. Selection and Correlation
    # We use a dictionary for .sel() to handle dynamic dimension names
    ref_point = u_prime_vals.sel(selection_dict, method='nearest')
    
    # Calculation
    corr = (u_prime_vals * ref_point).mean(dim='time')
    u_variance = u_prime_vals.var(dim='time')
    ref_point_variance = ref_point.var(dim='time')
    
    corr_coeff = corr / np.sqrt(u_variance * ref_point_variance)
    
    # Optional: Update the output name to reflect the correlation
    corr_coeff.name = f"R_{var_name}{var_name}"
    if x_ref is not None:
        corr_coeff.coords['lag_'+x_dim] = corr_coeff.coords[x_dim] - corr_coeff.coords[x_dim].sel({x_dim: x_ref}, method='nearest')
        corr_coeff = corr_coeff.swap_dims({x_dim: 'lag_'+x_dim})
    if y_ref is not None:   
        corr_coeff.coords['lag_'+y_dim] = corr_coeff.coords[y_dim] - corr_coeff.coords[y_dim].sel({y_dim: y_ref}, method='nearest')
        corr_coeff = corr_coeff.swap_dims({y_dim: 'lag_'+y_dim})

    corr_coeff.attrs['description'] = f"Two-point correlation of {var_name} relative to point ({x_ref}, {y_ref})"
    corr_coeff.attrs[x_dim+'_ref'] = x_ref
    corr_coeff.attrs[y_dim+'_ref'] = y_ref
    
    return corr_coeff


def _calc_L_1e_kernel(lag_values : np.ndarray, corr_values : np.ndarray) -> float:
    """1D Kernel: Finds the lag where correlation drops to 1/e."""
    # Find first zero crossing to avoid noise-driven 1/e crossings
    zero_idx = np.where(corr_values <= 0)[0]
    if len(zero_idx) > 0:
        limit = zero_idx[0]
        lag_lim = lag_values[:limit]
        corr_lim = corr_values[:limit]
    else:
        lag_lim, corr_lim = lag_values, corr_values

    target = 1 / np.e
    if len(corr_lim) < 2:
        warnings.warn(
                        "Not enough points to determine 1/e length scale. Returning NaN.",
                        UserWarning, stacklevel=2
                    )
        return np.nan
    if np.min(corr_lim) > target:
        warnings.warn(
                        "1/e point not found before zero crossing. Returning NaN.",
                        UserWarning, stacklevel=2
                    )
        return np.nan
    try:
        # Find the index where correlation first drops below the target.
        # searchsorted works on ascending arrays, so we search in the reversed array.
        idx = np.searchsorted(corr_lim[::-1], target)
        
        # Correct index for original (descending) array
        # This gives us the first index where corr_lim[i] < target
        i = len(corr_lim) - 1 - idx

        # Basic linear interpolation between point i and i+1
        # target = (1-t)*corr[i+1] + t*corr[i]
        # lag = (1-t)*lag[i+1] + t*lag[i]
        # Solving for t: t = (target - corr[i+1]) / (corr[i] - corr[i+1])
        t = (target - corr_lim[i+1]) / (corr_lim[i] - corr_lim[i+1])
        
        return (1-t) * lag_lim[i+1] + t * lag_lim[i]

    except (ValueError, IndexError):
        return np.nan

def _calc_L_fit_kernel(lag_values: np.ndarray, corr_values: np.ndarray) -> float:
    """1D Kernel: Fits a single exponential decay: exp(-x/L)."""

    def exp_decay(x, L):
        return np.exp(-x / L)

    try:
        popt = np.asarray(curve_fit(exp_decay, lag_values, corr_values, p0=[50], maxfev=100000)[0], dtype=float)
        return float(popt[0])
    except (ValueError, RuntimeError):
        return np.nan


def _calc_L_fit_biexp_kernel(lag_values: np.ndarray, corr_values: np.ndarray) -> np.ndarray:
    """1D Kernel: Fits a bi-exponential decay and returns a, L1, and L2."""

    def biexp_decay(x, a, L1, L2):
        return a * np.exp(-x / L1) + (1.0 - a) * np.exp(-x / L2)

    lag_values = np.asarray(lag_values, dtype=float)
    corr_values = np.asarray(corr_values, dtype=float)

    if lag_values.size < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    lag_max = float(np.nanmax(lag_values)) if np.any(np.isfinite(lag_values)) else 1.0
    if not np.isfinite(lag_max) or lag_max <= 0:
        lag_max = 1.0

    p0 = [0.5, max(lag_max / 10.0, 1e-6), max(lag_max / 2.0, 1e-6)]
    bounds = ([0.0, 1e-12, 1e-12], [1.0, np.inf, np.inf])

    try:
        result = curve_fit(
            biexp_decay,
            lag_values,
            corr_values,
            p0=p0,
            bounds=bounds,
            maxfev=100000,
        )
        popt = np.asarray(result[0], dtype=float)
        return np.asarray([popt[0], popt[1], popt[2]], dtype=float)
    except (ValueError, RuntimeError):
        return np.array([np.nan, np.nan, np.nan], dtype=float)


def _calc_L_integral_kernel(lag_values: np.ndarray, corr_values: np.ndarray) -> float:
    """1D Kernel: Integrates normalized correlation up to the first zero crossing."""
    lag_values = np.asarray(lag_values, dtype=float)
    corr_values = np.asarray(corr_values, dtype=float)

    finite_mask = np.isfinite(lag_values) & np.isfinite(corr_values)
    lag_values = lag_values[finite_mask]
    corr_values = corr_values[finite_mask]

    if lag_values.size < 2:
        return np.nan

    c0 = corr_values[0]
    if not np.isfinite(c0) or c0 == 0:
        return np.nan

    corr_norm = corr_values / c0
    zero_idx = np.where(corr_norm <= 0)[0]
    if zero_idx.size == 0:
        warnings.warn(
            "Zero crossing not found for integral length scale. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan

    end = int(zero_idx[0])
    if end == 0:
        return 0.0

    if corr_norm[end] == 0:
        tau = lag_values[: end + 1]
        corr_use = corr_norm[: end + 1]
    else:
        x0 = lag_values[end - 1]
        x1 = lag_values[end]
        y0 = corr_norm[end - 1]
        y1 = corr_norm[end]
        if y1 == y0:
            x_zero = x1
        else:
            x_zero = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
        tau = np.concatenate([lag_values[:end], np.array([x_zero], dtype=float)])
        corr_use = np.concatenate([corr_norm[:end], np.array([0.0], dtype=float)])

    return float(np.trapz(corr_use, tau))



def compute_length_scale(da: xr.DataArray, dim: str = 'lag_x', method: str = '1e', fit_model: str = 'single') -> xr.DataArray:
    """
    Computes integral length scale and broadcasts over all other coordinates.
    """
    # 1. Cleaning and Pre-processing
    # Ensure we only look at positive lags for the integral calculation
    subset = da.where(da[dim] >= 0, drop=True)
    
    # Sort and interpolate to handle missing data in the correlation map
    subset = subset.sortby(dim).interpolate_na(dim=dim)

    # 2. Select the calculation method
    if method == '1e':
        kernel = _calc_L_1e_kernel
        output_core_dims = [[]]
    elif method == 'integral':
        kernel = _calc_L_integral_kernel
        output_core_dims = [[]]
    elif method == 'fit' and fit_model == 'bi':
        kernel = _calc_L_fit_biexp_kernel
        output_core_dims = [['parameter']]
    else:
        kernel = _calc_L_fit_kernel
        output_core_dims = [[]]

    # 3. Use apply_ufunc for automatic broadcasting
    if method == 'fit' and fit_model == 'bi':
        result = xr.apply_ufunc(
            kernel,
            subset[dim],      # This maps to 'lag_vec' in the kernel
            subset,           # This maps to 'corr_vec' in the kernel
            input_core_dims=[[dim], [dim]], # The 'row' dimension to consume
            output_core_dims=output_core_dims,
            vectorize=True,                 # Tells xarray to loop over other dims
            dask="parallelized",            # Supports dask chunks if data is large
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'parameter': 3}},
        )
    else:
        result = xr.apply_ufunc(
            kernel,
            subset[dim],      # This maps to 'lag_vec' in the kernel
            subset,           # This maps to 'corr_vec' in the kernel
            input_core_dims=[[dim], [dim]], # The 'row' dimension to consume
            output_core_dims=output_core_dims,
            vectorize=True,                 # Tells xarray to loop over other dims
            dask="parallelized",            # Supports dask chunks if data is large
            output_dtypes=[float],
        )

    if method == 'fit' and fit_model == 'bi':
        result = result.assign_coords(parameter=['a', 'L1', 'L2'])

    return result


def non_uniform_spectra(
    t: np.ndarray,
    u: np.ndarray,
    w: np.ndarray,
    dt: float,
    K2: int,
    meanrem: bool = True,
    atw: bool = False,
    locnor: bool = False,
    selfproducts: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the autocorrelation and PSD for irregularly sampled data.

    Parameters
    ----------
    t : array-like
        Sample times [s].
    u : array-like
        Signal values.
    w : array-like
        Weights for each sample (for LDV typically transit-time based).
    dt : float
        Quantization time step [s].
    K2 : int
        Maximum lag index. Lags are evaluated from -K2 to +K2.
    meanrem : bool, default True
        Remove weighted mean before estimating correlation.
    atw : bool, default False
        Use arrival-time weighting.
    locnor : bool, default False
        Use local normalization.
    selfproducts : bool, default False
        Re-introduce self products at lag 0.

    Returns
    -------
    tau : ndarray
        Lag vector [s].
    R : ndarray
        Estimated autocorrelation function.
    f : ndarray
        Frequency vector [Hz].
    S : ndarray
        Power spectral density estimate from FFT(R).
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)

    N = len(t)
    if not (len(u) == N and len(w) == N):
        raise ValueError("The time, signal, and weights vectors must have the same length.")
    if N < 2:
        raise ValueError("At least two samples are required.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if K2 < 1:
        raise ValueError("K2 must be >= 1.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(w)):
        raise ValueError("Input arrays must be finite.")
    if not np.all(np.diff(t) >= 0):
        raise ValueError("The time vector t must be monotonically non-decreasing.")

    K1 = -int(K2)
    K2 = int(K2)
    K = K2 - K1 + 1

    if atw:
        wf = np.append(t[1:N] - t[0:N - 1], [0])
        wf[(wf > 5 * (t[N - 1] - t[0]) / (N - 1)) | (wf < 0)] = 0.0

        wb = np.append([0], t[1:N] - t[0:N - 1])
        wb[(wb > 5 * (t[N - 1] - t[0]) / (N - 1)) | (wb < 0)] = 0.0
    else:
        wf = w
        wb = w

    if meanrem:
        ur = u - np.sum(wb * u) / float(np.sum(wb))
    else:
        ur = u

    Je = int(np.ceil(t[N - 1] / float(dt))) + K2
    Je = 2 ** int(np.ceil(np.log(Je) / np.log(2)))

    if atw:
        df = 1 / float(Je * dt)
        fe = np.roll(np.arange(-(Je // 2), Je - (Je // 2)), -(Je // 2)) * df
        U1 = np.zeros(Je, dtype=complex)
        U0 = np.zeros(Je, dtype=complex)
        Ub = np.zeros(Je, dtype=complex)
        Wb = np.zeros(Je, dtype=complex)

        if locnor:
            Ui = np.zeros(Je, dtype=complex)
            Uj = np.zeros(Je, dtype=complex)
            Qb = np.zeros(Je, dtype=complex)

        for i in range(0, N):
            E = np.exp(-2j * np.pi * fe * np.floor(t[i] / float(dt)) * dt)
            Uf = wf[i] * ur[i] * E
            U1 += np.conj(Ub) * Uf
            Ub += wb[i] * ur[i] * E
            Wf = wf[i] * E
            U0 += np.conj(Wb) * Wf
            if locnor:
                Qf = wf[i] * np.abs(ur[i]) ** 2 * E
                Ui += np.conj(Qb) * Wf
                Uj += np.conj(Wb) * Qf
                Qb += wb[i] * np.abs(ur[i]) ** 2 * E
            Wb += wb[i] * E

        R1 = np.fft.ifft(U1 + np.conj(U1))
        if locnor:
            Ri = np.real(np.fft.ifft(Ui + np.conj(Uj)))
            Rj = np.real(np.fft.ifft(Uj + np.conj(Ui)))
        if (not locnor) or meanrem:
            R0 = np.real(np.fft.ifft(U0 + np.conj(U0)))

    else:
        ue = np.zeros(Je, dtype=complex)
        we = np.zeros(Je)
        if locnor:
            qe = np.zeros(Je)

        for i in range(0, N):
            idx = int(np.floor(t[i] / dt))
            ue[idx] += w[i] * ur[i]
            we[idx] += w[i]
            if locnor:
                qe[idx] += w[i] * np.abs(ur[i]) ** 2

        U1 = np.fft.fft(ue)
        U0 = np.fft.fft(we)

        if locnor:
            U2 = np.fft.fft(qe)

        R1 = np.fft.ifft(np.conj(U1) * U1 - np.sum(np.abs(w * ur) ** 2))

        if locnor:
            Ri = np.real(np.fft.ifft(np.conj(U2) * U0 - np.sum(np.abs(w * ur) ** 2)))
            Rj = np.real(np.fft.ifft(np.conj(U0) * U2 - np.sum(np.abs(w * ur) ** 2)))

        if (not locnor) or meanrem:
            R0 = np.real(np.fft.ifft(np.conj(U0) * U0 - np.sum(w ** 2)))

    if selfproducts:
        R1[0] += np.sum(wb * wf * np.abs(ur) ** 2)
        if locnor:
            Ri[0] += np.sum(wb * wf * np.abs(ur) ** 2)
            Rj[0] += np.sum(wb * wf * np.abs(ur) ** 2)
        if (not locnor) or meanrem:
            R0[0] += np.sum(wb * wf)

    tau = np.roll(np.arange(K1, K2 + 1), K1) * dt
    R = np.zeros(K, dtype=complex)

    if locnor:
        s2u = np.sum(wb * np.abs(ur) ** 2) / float(np.sum(wb))
        for k in range(K1, K2 + 1):
            if Ri[k] * Rj[k] > 0:
                R[k] = s2u * R1[k] / np.sqrt(Ri[k] * Rj[k])
    else:
        for k in range(K1, K2 + 1):
            if R0[k] > 0:
                R[k] = R1[k] / float(R0[k])

    if meanrem:
        denom = float(np.sum(wf) * np.sum(wb) - R0[0] - 2 * np.sum(R0[1:K2 + 1]))
        if selfproducts:
            R += (R[0] * R0[0] + 2 * np.sum(np.real(R1[1:K2 + 1]))) / denom
        else:
            R += (
                R[0] * R0[0]
                + 2 * np.sum(np.real(R1[1:K2 + 1]))
                + np.sum(wf * wb * np.abs(ur) ** 2)
            ) / (denom - np.sum(wf * wb))

    df = 1 / float(K * dt)
    f = np.roll(np.arange(-(K // 2), K - (K // 2)), -(K // 2)) * df
    S = dt * np.fft.fft(R)

    return tau, R, f, S