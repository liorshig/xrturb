import numpy as np
import xarray as xr
import pytest
from xrturb.spatial import vorticity, divergence, strain, acceleration


class TestVorticity:
    def test_vorticity_calculation(self, sample_dataset):
        """Test vorticity calculation."""
        vort = vorticity(sample_dataset)
        
        assert 'vorticity' in vort.attrs['standard_name']
        assert vort.attrs['units'] == '1/s'
        assert vort.dims == ('x', 'y', 'time')

    def test_vorticity_values(self, sample_dataset_single_frame):
        """Test vorticity with known values."""
        # Create a simple case: u = y, v = -x (solid body rotation)
        ds = sample_dataset_single_frame.copy()
        X, Y = np.meshgrid(ds.x, ds.y, indexing='ij')
        ds['u'] = (['x', 'y'], Y)
        ds['v'] = (['x', 'y'], -X)
        
        vort = vorticity(ds)
        
        # For solid body rotation, vorticity should be -2
        expected_vorticity = -2 * np.ones_like(vort)
        np.testing.assert_allclose(vort.values, expected_vorticity, rtol=1e-2)


class TestDivergence:
    def test_divergence_calculation(self, sample_dataset):
        """Test divergence calculation."""
        div = divergence(sample_dataset)
        
        assert 'divergence' in div.attrs['standard_name']
        assert div.attrs['units'] == '1/s'
        assert div.dims == ('x', 'y', 'time')


class TestStrain:
    def test_strain_calculation(self, sample_dataset):
        """Test strain rate calculation."""
        strain_rate = strain(sample_dataset)
        
        assert 'strain_rate' in strain_rate.attrs['standard_name']
        assert strain_rate.name == 'strain'
        assert strain_rate.dims == ('x', 'y', 'time')


class TestAcceleration:
    def test_acceleration_calculation(self, sample_dataset):
        """Test material acceleration calculation."""
        acc = acceleration(sample_dataset)
        
        assert 'acceleration' in acc.attrs['standard_name']
        assert acc.dims == ('x', 'y', 'time')