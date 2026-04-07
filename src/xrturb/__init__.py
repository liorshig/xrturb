"""Turbulence Analysis accessor for xarray Datasets."""

from typing import List, Optional, Literal, Tuple, Union

import numpy as np

import xarray as xr

from . import spatial
from . import statistics
from . import filtering
from . import utils
from . import quadrant_analysis


@xr.register_dataset_accessor("turb")
class TurbulenceAccessor:
    """
    A unified accessor for Turbulence Analysis (PIV, LDV, CFD).
    Combines spatial analysis (PIV) with weighted statistical analysis (LDV).
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    # =========================================================================
    # 1. CORE STATISTICAL HELPERS (Weighted)
    # =========================================================================

    def mean(self, weights: Optional[xr.DataArray] = None, dim: str = 'time') -> xr.Dataset:
        """Returns the (weighted) mean flow field."""
        return statistics.mean(self._obj, weights, dim)

    def fluct(self, var_key: str, weights: Optional[xr.DataArray] = None,
              dim: str = 'time') -> xr.DataArray:
        """
        Calculate fluctuations: u' = u - mean(u)
        Supports weighted means for LDV.
        """
        return statistics.fluct(self._obj, var_key, weights, dim)

    # =========================================================================
    # 2. SPATIAL DERIVATIVES (PIV / CFD)
    # =========================================================================

    def vorticity(self, u: str = 'u', v: str = 'v') -> xr.DataArray:
        """Calculates vertical vorticity (dv/dx - du/dy)."""
        return spatial.vorticity(self._obj, u, v)

    def divergence(self, u: str = 'u', v: str = 'v') -> xr.DataArray:
        """Calculates 2D divergence (du/dx + dv/dy)."""
        return spatial.divergence(self._obj, u, v)

    def strain(self, u: str = 'u', v: str = 'v') -> xr.DataArray:
        """Calculates scalar strain rate magnitude."""
        return spatial.strain(self._obj, u, v)

    def acceleration(self, u: str = 'u', v: str = 'v') -> xr.DataArray:
        """Calculates material derivative (convective acceleration)."""
        return spatial.acceleration(self._obj, u, v)

    # =========================================================================
    # 3. TURBULENCE STATISTICS (Reynolds Stress, TKE)
    # =========================================================================

    def calculate_product(self, product_key: str,
                         weights: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Calculates the product of fluctuating variables.
        Example: 'uv' calculates u' * v'
        """
        return statistics.calculate_product(self._obj, product_key, weights)

    def reynolds_stress(self, weights: Optional[xr.DataArray] = None) -> xr.Dataset:
        """
        Calculates Reynolds Shear Stress components: -rho * <u_i'u_j'>.

        Here returning just <u_i'u_j'> to be density independent.
        Automatically detects available velocity components (u, v, w) and
        computes all possible shear stress components:
        - For 3D data (u,v,w): uv, uw, vw
        - For 2D data (u,v): uv
        """
        return statistics.reynolds_stress(self._obj, weights)

    def tke(self, weights: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Calculates Turbulent Kinetic Energy.

        0.5 * sum(<comp'comp'> for available velocity components)
        Automatically detects available velocity components (u, v, w) and
        computes TKE based on what's present in the dataset.
        """
        return statistics.tke(self._obj, weights)

    def calc_moments(self, variables: Optional[List[str]] = None, order: int = 2,
                    weights: Optional[xr.DataArray] = None,
                    standardized: bool = False) -> xr.Dataset:
        """
        Calculate high-order moments for given variables.

        If standardized=True, computes standardized moments (normalized by std).
        If False, computes raw central moments.
        If variables=None, automatically detects available velocity components.
        """
        return statistics.calc_moments(self._obj, variables, order, weights,
                                       standardized)

    def calc_all_products(self, variables: Optional[List[str]] = None,
                         order: int = 2, weights: Optional[xr.DataArray] = None,
                         dim: str = 'time') -> xr.Dataset:
        """
        Calculate all fluctuation products up to specified order.

        Computes the element-wise products of fluctuations and adds them as
        new variables. If variables=None, automatically detects available
        velocity components.
        """
        return statistics.calc_all_products(self._obj, variables, order,
                                           weights, dim)

    def non_uniform_spectra(
        self,
        var_key: str = 'u',
        dim: str = 'time',
        weights: Optional[Union[str, xr.DataArray]] = None,
        dt: Optional[float] = None,
        K2: Optional[int] = None,
        meanrem: bool = True,
        atw: bool = False,
        locnor: bool = False,
        selfproducts: bool = False,
    ) -> xr.Dataset:
        """
        Estimate autocorrelation and PSD for a non-uniformly sampled 1D signal.

        Parameters
        ----------
        var_key : str, default 'u'
            Variable name in the dataset.
        dim : str, default 'time'
            Coordinate/dimension containing sample time.
        weights : str or xr.DataArray, optional
            Weight variable name in the dataset, or a 1D weight DataArray aligned
            with ``dim``. If None, uniform weights are used.
        dt : float, optional
            Quantization time step. If None, uses median time increment.
        K2 : int, optional
            Maximum lag index. If None, uses ``max(1, n_samples // 4)``.
        meanrem, atw, locnor, selfproducts : bool
            Options forwarded to ``statistics.non_uniform_spectra``.

        Returns
        -------
        xr.Dataset
            Dataset with ``autocorrelation`` over ``lag`` and ``psd`` over
            ``frequency``.
        """
        if var_key not in self._obj:
            raise ValueError(f"Variable '{var_key}' not found.")

        da = self._obj[var_key]
        if dim not in da.dims:
            raise ValueError(f"Dimension '{dim}' is not present in variable '{var_key}'.")

        if da.ndim != 1 or da.dims != (dim,):
            raise ValueError(
                "non_uniform_spectra accessor currently supports only 1D signals "
                f"with dims ('{dim}',)."
            )

        t = np.asarray(da[dim].values, dtype=float)
        u = np.asarray(da.values, dtype=float)

        if weights is None:
            w = np.ones_like(u, dtype=float)
        else:
            if isinstance(weights, str):
                if weights not in self._obj:
                    raise ValueError(f"Weight variable '{weights}' not found.")
                w_da = self._obj[weights]
            elif isinstance(weights, xr.DataArray):
                w_da = weights
            else:
                raise TypeError("weights must be a variable name (str), an xr.DataArray, or None.")

            if w_da.ndim != 1 or w_da.dims != (dim,):
                raise ValueError(
                    "weights must be 1D and aligned with the signal dimension."
                )
            w = np.asarray(w_da.values, dtype=float)

        if dt is None:
            diffs = np.diff(t)
            positive_diffs = diffs[diffs > 0]
            if positive_diffs.size == 0:
                raise ValueError("Cannot infer dt from time coordinate; provide dt explicitly.")
            dt = float(np.median(positive_diffs))

        if K2 is None:
            K2 = max(1, len(t) // 4)

        tau, R, f, S = statistics.non_uniform_spectra(
            t=t,
            u=u,
            w=w,
            dt=dt,
            K2=K2,
            meanrem=meanrem,
            atw=atw,
            locnor=locnor,
            selfproducts=selfproducts,
        )

        return xr.Dataset(
            data_vars={
                'autocorrelation': xr.DataArray(np.real(R), dims=('lag',), coords={'lag': tau}),
                'psd': xr.DataArray(np.real(S), dims=('frequency',), coords={'frequency': f}),
            },
            attrs={
                'method': 'non_uniform_spectra',
                'source_variable': var_key,
                'source_dim': dim,
                'dt': float(dt),
                'K2': int(K2),
            },
        )

    # =========================================================================
    # 4. SIGNAL PROCESSING (Smoothing, Splines)
    # =========================================================================

    def filter_gaussian(self, sigma: Optional[List[float]] = None,
                       variables: Optional[List[str]] = None, **kwargs) -> xr.Dataset:
        """Gaussian filtering of velocity fields."""
        if sigma is None:
            sigma = [1.0, 1.0, 0.0]
        if variables is None:
            variables = ['u', 'v']
        return filtering.filter_gaussian(self._obj, sigma, variables, **kwargs)

    def smooth_savgol(self, var_key: str, dim: str = 'z', window_length: int = 5,
                     polyorder: int = 2, **kwargs) -> xr.DataArray:
        """
        Smooth a DataArray using Savitzky-Golay filter.

        Interpolates to uniform grid if necessary.
        """
        return filtering.smooth_savgol(self._obj, var_key, dim, window_length,
                                       polyorder, **kwargs)

    def smooth_spline(self, var_key: str, dim: str = 'z', s_factor: Optional[float] = None,
                     order: int = 3, deriv: int = 0) -> xr.DataArray:
        """Smooth using B-Splines (Robust for non-uniform data)."""
        return filtering.smooth_spline(self._obj, var_key, dim, s_factor, order,
                                       deriv)
    
    def smooth(self, var_key: str, method: str = 'savgol', **kwargs) -> xr.DataArray:
        """
        General smoothing function that selects method.

        Parameters
        ----------
        var_key : str
            Variable to smooth
        method : str
            'savgol' or 'spline'
        kwargs : dict
            Additional parameters for the chosen method

        Returns
        -------
        xr.DataArray
            Smoothed variable
        """
        if method == 'savgol':
            return self.smooth_savgol(var_key, **kwargs)
        elif method == 'spline':
            return self.smooth_spline(var_key, **kwargs)
        else:
            raise ValueError(f"Unsupported smoothing method: {method}")

    # =========================================================================
    # 5. UTILITIES (Crop, Fill, Rotate)
    # =========================================================================

    def crop(self, vector: Optional[List[Optional[float]]] = None) -> xr.Dataset:
        """Crops dataset by coordinates: [xmin, xmax, ymin, ymax]."""
        if vector is None:
            vector = [None, None, None, None]
        return utils.crop(self._obj, vector)

    def pan(self, dx: float = 0.0, dy: float = 0.0) -> xr.Dataset:
        """Shifts coordinates."""
        return utils.pan(self._obj, dx, dy)

    def fill_nans(self,
                 method: Literal["linear", "nearest", "cubic"] = "nearest"
                 ) -> xr.Dataset:
        """2D spatial interpolation to fill NaNs."""
        return utils.fill_nans(self._obj, method)

    def rotate(self, theta_deg: float) -> xr.Dataset:
        """Rotates the spatial grid AND the velocity vectors."""
        return utils.rotate(self._obj, theta_deg)

    # =========================================================================
    # 6. QUADRANT ANALYSIS
    # =========================================================================

    def add_quadrants(self, x_var: str = 'u', y_var: str = 'w',
                     z_var: str = 'c', analysis_type: str = 'quadrant',
                     hole_size: float = 0,
                     hole_vars: Optional[Tuple[str, str]] = None,
                     weights: Optional[xr.DataArray] = None,
                     dim: str = 'time') -> xr.Dataset:
        """
        Add quadrant/octant classification and hole filtering to dataset.

        Parameters
        ----------
        x_var, y_var : str
            Primary variables for classification
        z_var : str
            Third variable for octant analysis
        analysis_type : str
            'quadrant' or 'octant'
        hole_size : float
            Hole size parameter for filtering
        hole_vars : tuple of str, optional
            Variables for hole calculation
        weights : str, optional
            Weighting variable for mean calculations
        dim : str
            Dimension for averaging

        Returns
        -------
        xr.Dataset
            Dataset with quadrant and hole coordinates
        """
        return quadrant_analysis.add_quadrants(self._obj, x_var, y_var, z_var,
                                              analysis_type, hole_size,
                                              hole_vars, weights, dim)

    def calc_quadrant_mean(self, analysis_type: str = 'quadrant',
                          x_var: str = 'u', y_var: str = 'w',
                          z_var: str = 'c', hole_size: float = 0,
                          hole_vars: Optional[Tuple[str, str]] = None,
                          weights: Optional[xr.DataArray] = None, dim: str = 'time',
                          time_var: str = 'transit_time') -> xr.Dataset:
        """
        Calculate conditional averages for each quadrant/octant.

        Parameters
        ----------
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
            Conditional means for each quadrant
        """
        return quadrant_analysis.calc_quadrant_mean(self._obj, analysis_type,
                                                   x_var, y_var, z_var,
                                                   hole_size, hole_vars,
                                                   weights, dim, time_var)

    def calc_quadrant_fraction(self, analysis_type: str = 'quadrant',
                              x_var: str = 'u', y_var: str = 'w',
                              z_var: str = 'c', hole_size: float = 0,
                              hole_vars: Optional[Tuple[str, str]] = None,
                              weights: Optional[xr.DataArray] = None, dim: str = 'time',
                              time_var: str = 'transit_time') -> xr.Dataset:
        """
        Calculate the fraction of total contribution from each quadrant.

        Parameters
        ----------
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
        return quadrant_analysis.calc_quadrant_fraction(self._obj, analysis_type,
                                                       x_var, y_var, z_var,
                                                       hole_size, hole_vars,
                                                       weights, dim, time_var)

    def calc_quadrant_contribution(self, analysis_type: str = 'quadrant',
                                  x_var: str = 'u', y_var: str = 'w',
                                  z_var: str = 'c', hole_size: float = 0,
                                  hole_vars: Optional[Tuple[str, str]] = None,
                                  weights: Optional[xr.DataArray] = None, dim: str = 'time',
                                  time_var: str = 'transit_time') -> xr.Dataset:
        """
        Calculate the contribution of each quadrant to the total.

        Parameters
        ----------
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
            Contribution of each quadrant
        """
        return quadrant_analysis.calc_quadrant_contribution(self._obj,
                                                           analysis_type,
                                                           x_var, y_var, z_var,
                                                           hole_size, hole_vars,
                                                           weights, dim,
                                                           time_var)

    def calc_quadrant_duration(self, analysis_type: str = 'quadrant',
                              x_var: str = 'u', y_var: str = 'w',
                              z_var: str = 'c', hole_size: float = 0,
                              hole_vars: Optional[Tuple[str, str]] = None,
                              weights: Optional[xr.DataArray] = None, dim: str = 'time',
                              time_var: str = 'transit_time') -> xr.Dataset:
        """
        Calculate the time duration spent in each quadrant.

        Parameters
        ----------
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
        return quadrant_analysis.calc_quadrant_duration(self._obj,
                                                       analysis_type, x_var,
                                                       y_var, z_var, hole_size,
                                                       hole_vars, weights, dim,
                                                       time_var)

    # =========================================================================
    # 7. GRAPHICS WRAPPERS (Optional)
    # =========================================================================

    def quiver(self, **kwargs) -> None:
        """Wrapper for external quiver plot function."""