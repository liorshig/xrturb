import numpy as np
import xarray as xr
from scipy.interpolate import splrep, BSpline
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from typing import List


def filter_gaussian(ds: xr.Dataset, sigma: List[float] = [1.0, 1.0, 0.0], variables: List[str] = ['u', 'v'], **kwargs) -> xr.Dataset:
    """Gaussian filtering of velocity fields."""
    ds_filtered = ds.copy()
    for var in variables:
        if var in ds_filtered:
            ds_filtered[var] = xr.DataArray(
                gaussian_filter(ds_filtered[var].values, sigma, **kwargs),
                dims=ds_filtered[var].dims,
                coords=ds_filtered[var].coords,
                attrs=ds_filtered[var].attrs
            )
    return ds_filtered


def smooth_savgol(ds: xr.Dataset, var_key: str, dim: str = 'z', window_length: int = 5, polyorder: int = 2, **kwargs) -> xr.DataArray:
    """
    Smooth a DataArray using Savitzky-Golay filter.
    Interpolates to uniform grid if necessary.
    """
    da = ds[var_key]
    
    # Check if grid is uniform
    coord_vals = da[dim].values
    if np.unique(np.diff(coord_vals)).size > 1:
        # Non-uniform: Interp to agrid
        min_val, max_val = np.nanmin(coord_vals), np.nanmax(coord_vals)
        diffs = np.diff(np.sort(coord_vals))
        step = np.nanmin(diffs[diffs > 0])
        agrid = np.arange(min_val, max_val + step, step)
        da_interp = da.interp({dim: agrid})
        delta = step
    else:
        da_interp = da
        delta = coord_vals[1] - coord_vals[0]

    # Apply filter
    kwargs = kwargs | {'window_length': window_length, 'polyorder': polyorder, 'delta': delta}
    
    smoothed = xr.apply_ufunc(
        savgol_filter,
        da_interp,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=kwargs,
    )

    # Interpolate back to original coords if we shifted
    if np.unique(np.diff(coord_vals)).size > 1:
         return smoothed.interp({dim: coord_vals})
    return smoothed


def smooth_spline(ds: xr.Dataset, var_key: str, dim: str = 'z', s_factor: float = None, order: int = 3, deriv: int = 0) -> xr.DataArray:
    """
    Smooth using B-Splines (Robust for non-uniform data).
    """
    da = ds[var_key]
    coords = ds[dim]

    return xr.apply_ufunc(
        _spline_smoother_func,
        da,
        coords,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        kwargs={'s': s_factor, 'k': order, 'nu': deriv},
        vectorize=True 
    )


def _spline_smoother_func(y, x, s, k, nu):
    """Static helper for spline ufunc."""
    finite_mask = np.isfinite(y) & np.isfinite(x)
    if finite_mask.sum() <= k:
        return np.full_like(y, np.nan) # Return NaNs if not enough points

    x_fin, y_fin = x[finite_mask], y[finite_mask]
    
    # Sort is required for splrep
    sort_idx = np.argsort(x_fin)
    x_fin, y_fin = x_fin[sort_idx], y_fin[sort_idx]

    try:
        tck = splrep(x_fin, y_fin, s=s, k=k)
        spl = BSpline(*tck)
        if nu > 0:
            spl = spl.derivative(nu=nu)
        return spl(x)
    except Exception:
        return np.full_like(y, np.nan)