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
def sample_dataset_single_frame():
    """Create a sample dataset with single time frame."""
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = 1 + 0.1 * np.sin(2 * np.pi * X)
    v = 0.1 * np.cos(2 * np.pi * Y)
    
    ds = xr.Dataset(
        {
            'u': (['x', 'y'], u),
            'v': (['x', 'y'], v),
        },
        coords={
            'x': x,
            'y': y,
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