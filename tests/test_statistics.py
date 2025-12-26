import numpy as np
import xarray as xr
import pytest
from xrturb.statistics import (
    mean, fluct, calculate_product, reynolds_stress, 
    tke, calc_moments, calc_all_products, _weighted_mean,
    two_point_correlation
)


class TestWeightedMean:
    def test_weighted_mean_basic(self, sample_dataset):
        """Test weighted mean calculation."""
        # Create weights
        weights = np.ones_like(sample_dataset.time)
        weights_da = xr.DataArray(weights, dims=['time'])
        
        result = _weighted_mean(sample_dataset['u'], weights_da)
        
        # Should be close to unweighted mean
        expected = sample_dataset['u'].mean('time')
        xr.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_weighted_mean_none_weights(self, sample_dataset):
        """Test unweighted mean when weights=None."""
        result = _weighted_mean(sample_dataset['u'], None)
        
        expected = sample_dataset['u'].mean('time')
        xr.testing.assert_allclose(result, expected)


class TestMean:
    def test_mean_basic(self, sample_dataset):
        """Test mean calculation."""
        mean_ds = mean(sample_dataset)
        
        assert 'time' not in mean_ds.dims
        assert 'u' in mean_ds
        assert 'v' in mean_ds

    def test_mean_with_weights(self, sample_dataset):
        """Test weighted mean."""
        # Add weights
        weights = np.random.rand(len(sample_dataset.time))
        weights_da = xr.DataArray(weights, dims=['time'], name='weights')
        
        mean_ds = mean(sample_dataset, weights=weights_da)
        
        # Check that u has been averaged (no t dim)
        assert 'time' not in mean_ds['u'].dims


class TestFluct:
    def test_fluct_basic(self, sample_dataset):
        """Test fluctuation calculation."""
        fluct_u = fluct(sample_dataset, 'u')
        
        assert fluct_u.dims == ('x', 'y', 'time')
        # Mean of fluctuations should be close to zero
        mean_fluct = fluct_u.mean('time')
        np.testing.assert_allclose(mean_fluct.values, 0, atol=1e-10)


class TestCalculateProduct:
    def test_calculate_product_uv(self, sample_dataset):
        """Test product calculation for 'uv'."""
        product = calculate_product(sample_dataset, 'uv')
        
        assert product.name == 'uv'
        assert product.dims == ('x', 'y', 'time')

    def test_calculate_product_single_var(self, sample_dataset):
        """Test product calculation for single variable 'uu'."""
        product = calculate_product(sample_dataset, 'uu')
        
        assert product.name == 'uu'


class TestReynoldsStress:
    def test_reynolds_stress_basic(self, sample_dataset):
        """Test Reynolds stress calculation."""
        rss = reynolds_stress(sample_dataset)
        
        assert isinstance(rss, xr.Dataset)
        assert "u'v'" in rss
        assert rss["u'v'"].attrs['standard_name'] == "Reynolds_stress_u'v'"
        assert rss["u'v'"].dims == ('x', 'y')

    def test_reynolds_stress_3d(self, sample_dataset_3d):
        """Test Reynolds stress calculation for 3D data."""
        rss = reynolds_stress(sample_dataset_3d)
        
        assert isinstance(rss, xr.Dataset)
        # Should have uv, uw, vw
        assert "u'v'" in rss
        assert "u'w'" in rss
        assert "v'w'" in rss
        assert len(rss.data_vars) == 3


class TestTKE:
    def test_tke_basic(self, sample_dataset):
        """Test TKE calculation."""
        tke_val = tke(sample_dataset)
        
        assert 'TKE' in tke_val.attrs['standard_name']
        assert tke_val.attrs['units'] == 'm^2/s^2'
        assert tke_val.dims == ('x', 'y')

    def test_tke_single_frame(self, sample_dataset_single_frame):
        """Test TKE for single frame (instantaneous KE)."""
        tke_val = tke(sample_dataset_single_frame)
        
        assert 'kinetic_energy' in tke_val.attrs['standard_name']
        assert tke_val.dims == ('x', 'y')

    def test_tke_with_w_component(self, sample_dataset):
        """Test TKE with w component."""
        w_data = 0.1 * np.random.randn(*sample_dataset['u'].shape)
        ds_with_w = sample_dataset.assign(w=(['x', 'y', 'time'], w_data))
        
        tke_val = tke(ds_with_w)
        
        assert tke_val.dims == ('x', 'y')

    def test_tke_u_and_w_only(self, sample_dataset):
        """Test TKE with only u and w components (no v)."""
        # Remove v from sample_dataset and add w
        ds_u_w = sample_dataset.drop_vars('v')
        w_data = 0.1 * np.random.randn(*sample_dataset['u'].shape)
        ds_u_w = ds_u_w.assign(w=(['x', 'y', 'time'], w_data))
        
        tke_val = tke(ds_u_w)
        
        assert tke_val.dims == ('x', 'y')
        assert '2 velocity components' in tke_val.attrs['note']


class TestCalcMoments:
    def test_calc_moments_basic(self, sample_dataset):
        """Test moment calculation."""
        moments = calc_moments(sample_dataset)
        
        assert isinstance(moments, xr.Dataset)
        # Should have some moment variables
        assert len(moments.data_vars) > 0

    def test_calc_moments_higher_order(self, sample_dataset):
        """Test higher order moments."""
        moments = calc_moments(sample_dataset, order=3)
        
        # Should have more variables for order 3
        assert len(moments.data_vars) > len(calc_moments(sample_dataset, order=2).data_vars)

    def test_calc_moments_standardized(self, sample_dataset):
        """Test standardized moment calculation."""
        moments_std = calc_moments(sample_dataset, standardized=True)
        
        assert isinstance(moments_std, xr.Dataset)
        assert len(moments_std.data_vars) > 0
        # For standardized, "u'u'" should be approximately 1 (variance / variance)
        if "Mu'u'" in moments_std:
            assert abs(moments_std["Mu'u'"].mean().values - 1.0) < 0.1  # rough check


class TestCalcProducts:
    def test_calc_all_products_basic(self, sample_dataset):
        """Test product calculation."""
        products_ds = calc_all_products(sample_dataset)
        
        assert isinstance(products_ds, xr.Dataset)
        # Should have original variables plus new product variables
        assert len(products_ds.data_vars) > len(sample_dataset.data_vars)
        # Check that products have the same dimensions as input
        for var in products_ds.data_vars:
            if var not in sample_dataset.data_vars:
                assert products_ds[var].dims == tuple(sample_dataset.dims)

    def test_calc_all_products_higher_order(self, sample_dataset):
        """Test higher order products."""
        products_low = calc_all_products(sample_dataset, order=2)
        products_high = calc_all_products(sample_dataset, order=3)
        
        # Should have more variables for order 3
        assert len(products_high.data_vars) > len(products_low.data_vars)

    def test_calc_all_products_with_weights(self, sample_dataset):
        """Test product calculation with weights."""
        # Add weights to the dataset
        weights = np.random.rand(len(sample_dataset.time))
        weights_da = xr.DataArray(weights, dims=['time'], name='weights')
        
        products_ds = calc_all_products(sample_dataset, weights=weights_da)
        
        assert isinstance(products_ds, xr.Dataset)
        assert len(products_ds.data_vars) > len(sample_dataset.data_vars)


class TestTwoPointCorrelation:
    def test_two_point_correlation_basic(self, sample_dataset):
        """Test the basic 2-point correlation calculation."""
        corr = two_point_correlation(sample_dataset, var_name='u')
        
        # Check shape and dims
        assert corr.dims == ('x', 'y')
        assert corr.shape == (sample_dataset.x.size, sample_dataset.y.size)
        
        # The correlation at the reference point should be 1.0
        ref_x = sample_dataset.x[0].item()
        ref_y = sample_dataset.y[0].item()
        np.testing.assert_allclose(corr.sel(x=ref_x, y=ref_y, method='nearest'), 1.0, atol=1e-5)

    def test_two_point_correlation_reference_point(self, sample_dataset):
        """Test correlation with a specified reference point."""
        x_ref, y_ref = 1.5, 2.5
        corr = two_point_correlation(sample_dataset, var_name='u', x_ref=x_ref, y_ref=y_ref)

        # Check shape and dims
        assert corr.dims == ('x', 'y')
        
        # The correlation at the specified reference point should be 1.0
        np.testing.assert_allclose(corr.sel(x=x_ref, y=y_ref, method='nearest'), 1.0, atol=1e-5)

    def test_two_point_correlation_missing_variable(self, sample_dataset):
        """Test that it raises ValueError for a missing variable."""
        with pytest.raises(ValueError, match="Variable 'non_existent_var' not found"):
            two_point_correlation(sample_dataset, var_name='non_existent_var')