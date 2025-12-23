import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from typing import List, Optional, Literal


def fill_nans(ds: xr.Dataset, method: Literal["linear", "nearest", "cubic"] = "nearest") -> xr.Dataset:
    """2D spatial interpolation to fill NaNs."""
    # Note: This is computationally expensive for large T.
    # Consider using xarray's native .interpolate_na if 1D filling is acceptable.
    
    ds_filled = ds.copy()
    
    # Grid for interpolation
    x_grid, y_grid = np.meshgrid(ds_filled.x, ds_filled.y)
    
    def _fill_frame(data):
        # Flatten
        vals = data.flatten()
        valid = ~np.isnan(vals)
        if not valid.any(): return data # Empty frame
        
        points = np.array([x_grid.flatten()[valid], y_grid.flatten()[valid]]).T
        values = vals[valid]
        
        return griddata(points, values, (x_grid, y_grid), method=method)

    # Apply over time
    for var in ['u', 'v']:
        if var in ds_filled:
            # Use xarray map_blocks or explicit loop (explicit loop safer for griddata)
            # For brevity, implementing a simplified loop:
            if 't' in ds_filled.dims:
                for t in range(len(ds_filled.t)):
                    frame = ds_filled[var].isel(t=t).values
                    if np.isnan(frame).any():
                        ds_filled[var].isel(t=t).values[:] = _fill_frame(frame)
            else:
                ds_filled[var].values[:] = _fill_frame(ds_filled[var].values)
                
    return ds_filled


def crop(ds: xr.Dataset, vector: List[Optional[float]] = None) -> xr.Dataset:
    if vector is None:
        vector = [None, None, None, None]
    """Crops dataset by coordinates: [xmin, xmax, ymin, ymax]."""
    xmin, xmax, ymin, ymax = vector
    
    # Get current bounds if None provided
    if xmin is None: xmin = ds.x.min()
    if xmax is None: xmax = ds.x.max()
    if ymin is None: ymin = ds.y.min()
    if ymax is None: ymax = ds.y.max()

    return ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))


def pan(ds: xr.Dataset, dx: float = 0.0, dy: float = 0.0) -> xr.Dataset:
    """Shifts coordinates."""
    return ds.assign_coords({
        "x": ds.x + dx,
        "y": ds.y + dy
    })


def rotate(ds: xr.Dataset, theta_deg: float) -> xr.Dataset:
    """Rotates the spatial grid AND the velocity vectors."""
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 1. Rotate Coordinates
    x_new = ds.x * cos_t + ds.y * sin_t
    y_new = ds.y * cos_t - ds.x * sin_t
    
    # 2. Rotate Vectors
    u_new = ds.u * cos_t + ds.v * sin_t
    v_new = ds.v * cos_t - ds.u * sin_t

    ds_rot = ds.copy()
    ds_rot["x"] = x_new
    ds_rot["y"] = y_new
    ds_rot["u"] = u_new
    ds_rot["v"] = v_new
    
    ds_rot.attrs['rotation_angle'] = ds_rot.attrs.get('rotation_angle', 0) + theta_deg
    return ds_rot