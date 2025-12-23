import numpy as np
import xarray as xr
import pytest
from xrturb.utils import fill_nans, crop, pan, rotate


class TestFillNans:
    def test_fill_nans_basic(self, sample_dataset):
        """Test basic NaN filling."""
        ds = sample_dataset.copy()
        # Introduce some NaNs
        ds['u'].loc[{'x': ds.x[0], 'y': ds.y[0]}] = np.nan
        
        filled = fill_nans(ds)
        
        assert not np.isnan(filled['u']).any()
        assert not np.isnan(filled['v']).any()

    def test_fill_nans_no_nans(self, sample_dataset):
        """Test that function works when no NaNs present."""
        filled = fill_nans(sample_dataset)
        
        assert filled.equals(sample_dataset)


class TestCrop:
    def test_crop_basic(self, sample_dataset):
        """Test basic cropping."""
        cropped = crop(sample_dataset, [0.2, 0.8, 0.2, 0.8])
        
        assert cropped.x.min() >= 0.2
        assert cropped.x.max() <= 0.8
        assert cropped.y.min() >= 0.2
        assert cropped.y.max() <= 0.8

    def test_crop_none_values(self, sample_dataset):
        """Test cropping with None values (use full range)."""
        cropped = crop(sample_dataset, [0.2, None, None, 0.8])
        
        assert cropped.x.min() >= 0.2
        assert cropped.x.max() == sample_dataset.x.max()
        assert cropped.y.min() == sample_dataset.y.min()
        assert cropped.y.max() <= 0.8


class TestPan:
    def test_pan_basic(self, sample_dataset):
        """Test coordinate shifting."""
        panned = pan(sample_dataset, dx=0.1, dy=-0.05)
        
        assert panned.x.min() == sample_dataset.x.min() + 0.1
        assert panned.y.min() == sample_dataset.y.min() - 0.05


class TestRotate:
    def test_rotate_90_degrees(self, sample_dataset):
        """Test 90-degree rotation."""
        rotated = rotate(sample_dataset, 90)
        
        # After 90-degree rotation, coordinates should swap
        # This is a basic check; full validation would require more complex assertions
        assert 'rotation_angle' in rotated.attrs
        assert rotated.attrs['rotation_angle'] == 90

    def test_rotate_zero_degrees(self, sample_dataset):
        """Test zero rotation returns similar dataset."""
        rotated = rotate(sample_dataset, 0)
        
        # Should be approximately equal (allowing for floating point precision)
        xr.testing.assert_allclose(rotated[['u', 'v']], sample_dataset[['u', 'v']], rtol=1e-10)