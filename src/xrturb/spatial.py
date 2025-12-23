import numpy as np
import xarray as xr


def vorticity(ds: xr.Dataset, u: str = 'u', v: str = 'v') -> xr.DataArray:
    """Calculates vertical vorticity (dv/dx - du/dy)."""
    dv_dx = ds[v].differentiate("x")
    du_dy = ds[u].differentiate("y")
    
    vort = dv_dx - du_dy
    vort.attrs["standard_name"] = "vorticity"
    vort.attrs["units"] = "1/s" # Should arguably derive from input units
    return vort


def divergence(ds: xr.Dataset, u: str = 'u', v: str = 'v') -> xr.DataArray:
    """Calculates 2D divergence (du/dx + dv/dy)."""
    du_dx = ds[u].differentiate("x")
    dv_dy = ds[v].differentiate("y")
    
    div = du_dx + dv_dy
    div.attrs["standard_name"] = "divergence"
    div.attrs["units"] = "1/s"
    return div


def strain(ds: xr.Dataset, u: str = 'u', v: str = 'v') -> xr.DataArray:
    """Calculates scalar strain rate magnitude."""
    du_dx = ds[u].differentiate("x")
    du_dy = ds[u].differentiate("y")
    dv_dx = ds[v].differentiate("x")
    dv_dy = ds[v].differentiate("y")

    # Definition of scalar strain rate often varies, this is from the PIV code
    s = du_dx**2 + dv_dy**2 + 0.5 * (du_dy + dv_dx)**2
    s.attrs["standard_name"] = "strain_rate"
    s.name = "strain"
    return s


def acceleration(ds: xr.Dataset, u: str = 'u', v: str = 'v') -> xr.DataArray:
    """Calculates material derivative (convective acceleration)."""
    du_dx = ds[u].differentiate("x")
    du_dy = ds[u].differentiate("y")
    dv_dx = ds[v].differentiate("x")
    dv_dy = ds[v].differentiate("y")

    acc_x = ds[u] * du_dx + ds[v] * du_dy
    acc_y = ds[u] * dv_dx + ds[v] * dv_dy

    acc = np.sqrt(acc_x**2 + acc_y**2)
    acc.attrs["standard_name"] = "acceleration"
    return acc