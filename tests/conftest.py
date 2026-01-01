import numpy as np
import xarray as xr
import pytest


@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset for testing."""
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 5)
    
    # Create synthetic velocity fields
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    u = 1 + 0.1 * np.sin(2 * np.pi * X) + 0.05 * np.random.randn(*X.shape)
    v = 0.1 * np.cos(2 * np.pi * Y) + 0.05 * np.random.randn(*X.shape)
    
    ds = xr.Dataset(
        {
            'u': (['x', 'y', 'time'], u),
            'v': (['x', 'y', 'time'], v),
        },
        coords={
            'x': x,
            'y': y,
            'time': t,
        }
    )
    return ds


@pytest.fixture
def sample_dataset_1d():
    """Create a sample 1D xarray Dataset for testing."""
    x = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 5)
    
    # Create synthetic velocity field
    X, T = np.meshgrid(x, t, indexing='ij')
    
    u = 1 + 0.1 * np.sin(2 * np.pi * X) + 0.05 * np.random.randn(*X.shape)
    
    ds = xr.Dataset(
        {
            'u': (['x', 'time'], u),
        },
        coords={
            'x': x,
            'time': t,
        }
    )
    return ds


@pytest.fixture
def sample_dataset_3d():
    """Create a sample 3D xarray Dataset for testing."""
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 5)
    
    # Create synthetic velocity fields
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    u = 1 + 0.1 * np.sin(2 * np.pi * X) + 0.05 * np.random.randn(*X.shape)
    v = 0.1 * np.cos(2 * np.pi * Y) + 0.05 * np.random.randn(*X.shape)
    w = 0.05 * np.sin(2 * np.pi * X + np.pi/4) + 0.05 * np.random.randn(*X.shape)
    
    ds = xr.Dataset(
        {
            'u': (['x', 'y', 'time'], u),
            'v': (['x', 'y', 'time'], v),
            'w': (['x', 'y', 'time'], w),
        },
        coords={
            'x': x,
            'y': y,
            'time': t,
        }
    )
    return ds




@pytest.fixture
def correlation_data_array():
    """Create a sample DataArray with exponential correlation for testing length scale."""
    lag_x = np.linspace(0, 50, 100)
    y = np.array([1, 2])
    L_true = 10.0  # True length scale
    
    # Create an exponential decay correlation
    correlation = np.exp(-lag_x / L_true)
    
    # Add a second dimension to test broadcasting
    corr_2d, _ = xr.broadcast(xr.DataArray(correlation, dims=['lag_x'], coords={'lag_x': lag_x}),
                              xr.DataArray(y, dims=['y'], coords={'y': y}))

    return corr_2d
