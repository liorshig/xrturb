import numpy as np
import xarray as xr
import pytest
import xrturb


class TestTurbulenceAccessor:
    def test_accessor_registration(self, sample_dataset):
        """Test that turb accessor is registered."""
        assert hasattr(sample_dataset, 'turb')
        assert isinstance(sample_dataset.turb, xrturb.TurbulenceAccessor)

    def test_accessor_init(self, sample_dataset):
        """Test accessor initialization."""
        accessor = sample_dataset.turb
        assert accessor._obj is sample_dataset

    def test_vorticity_accessor(self, sample_dataset):
        """Test vorticity through accessor."""
        vort = sample_dataset.turb.vorticity()
        
        assert 'vorticity' in vort.attrs['standard_name']
        assert vort.dims == ('x', 'y', 't')

    def test_divergence_accessor(self, sample_dataset):
        """Test divergence through accessor."""
        div = sample_dataset.turb.divergence()
        
        assert 'divergence' in div.attrs['standard_name']

    def test_mean_accessor(self, sample_dataset):
        """Test mean through accessor."""
        mean_ds = sample_dataset.turb.mean()
        
        assert 't' not in mean_ds.dims

    def test_fluct_accessor(self, sample_dataset):
        """Test fluctuations through accessor."""
        fluct_u = sample_dataset.turb.fluct('u')
        
        assert fluct_u.dims == ('x', 'y', 't')

    def test_tke_accessor(self, sample_dataset):
        """Test TKE through accessor."""
        tke_val = sample_dataset.turb.tke()
        
        assert 'TKE' in tke_val.attrs['standard_name']

    def test_reynolds_stress_accessor(self, sample_dataset):
        """Test Reynolds stress through accessor."""
        rss = sample_dataset.turb.reynolds_stress()
        
        assert isinstance(rss, xr.Dataset)
        assert "u'v'" in rss
        assert rss["u'v'"].attrs['standard_name'] == "Reynolds_stress_u'v'"

    def test_filter_gaussian_accessor(self, sample_dataset):
        """Test Gaussian filter through accessor."""
        filtered = sample_dataset.turb.filter_gaussian()
        
        assert filtered.dims == sample_dataset.dims

    def test_crop_accessor(self, sample_dataset):
        """Test crop through accessor."""
        cropped = sample_dataset.turb.crop([0.2, 0.8, 0.2, 0.8])
        
        assert cropped.x.min() >= 0.2
        assert cropped.x.max() <= 0.8

    def test_fill_nans_accessor(self, sample_dataset):
        """Test fill_nans through accessor."""
        # Add some NaNs
        ds_with_nan = sample_dataset.copy()
        ds_with_nan['u'].loc[{'x': ds_with_nan.x[0]}] = np.nan
        
        filled = ds_with_nan.turb.fill_nans()
        
        assert not np.isnan(filled['u']).any()

    def test_calc_moments_accessor(self, sample_dataset):
        """Test moment calculation through accessor."""
        moments = sample_dataset.turb.calc_moments()
        
        assert isinstance(moments, xr.Dataset)
        assert len(moments.data_vars) > 0

    def test_calc_products_accessor(self, sample_dataset):
        """Test product calculation through accessor."""
        products_ds = sample_dataset.turb.calc_all_products()
        
        assert isinstance(products_ds, xr.Dataset)
        assert len(products_ds.data_vars) > len(sample_dataset.data_vars)
        assert tuple(products_ds.dims) == tuple(sample_dataset.dims)