import numpy as np
import xarray as xr
import pytest
from xrturb.filtering import filter_gaussian, smooth_savgol, smooth_spline


class TestFilterGaussian:
    def test_filter_gaussian_basic(self, sample_dataset):
        """Test Gaussian filtering."""
        filtered = filter_gaussian(sample_dataset)
        
        assert filtered.dims == sample_dataset.dims
        assert 'u' in filtered
        assert 'v' in filtered

    def test_filter_gaussian_custom_vars(self, sample_dataset):
        """Test filtering specific variables."""
        filtered = filter_gaussian(sample_dataset, variables=['u'])
        
        # u should be filtered, v should be unchanged
        assert not filtered['u'].equals(sample_dataset['u'])
        assert filtered['v'].equals(sample_dataset['v'])


class TestSmoothSavgol:
    def test_smooth_savgol_basic(self, sample_dataset_1d):
        """Test Savitzky-Golay smoothing."""
        smoothed = smooth_savgol(sample_dataset_1d, 'u', dim='x')
        
        assert set(smoothed.dims) == {'x', 't'}
        assert smoothed.name == 'u'

    def test_smooth_savgol_uniform_grid(self, sample_dataset_1d):
        """Test smoothing on uniform grid."""
        smoothed = smooth_savgol(sample_dataset_1d, 'u', dim='x')
        
        assert set(smoothed.dims) == {'x', 't'}


class TestSmoothSpline:
    def test_smooth_spline_basic(self, sample_dataset_1d):
        """Test spline smoothing."""
        smoothed = smooth_spline(sample_dataset_1d, 'u', dim='x')
        
        assert set(smoothed.dims) == {'x', 't'}
        assert smoothed.name == 'u'

    def test_smooth_spline_derivative(self, sample_dataset_1d):
        """Test spline smoothing with derivative."""
        smoothed_deriv = smooth_spline(sample_dataset_1d, 'u', dim='x', deriv=1)
        
        assert set(smoothed_deriv.dims) == {'x', 't'}