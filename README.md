# xrturb

A Python package for turbulence analysis using xarray, providing comprehensive tools for processing and analyzing turbulent flow data from PIV/LDV measurements.

## Features

- **Turbulence Statistics**: Calculate TKE, Reynolds stresses, and higher-order moments
- **Spatial Derivatives**: Compute vorticity, divergence, strain rates
- **Filtering**: Apply Gaussian, Savitzky-Golay, and other filters
- **Data Utilities**: Fill NaNs, crop, rotate, and manipulate datasets
- **xarray Integration**: Seamless integration with xarray's Dataset and DataArray

## Installation

```bash
pip install xrturb
```

For development:

```bash
git clone https://github.com/yourusername/xrturb.git
cd xrturb
uv sync --dev
uv pip install -e .
```

## Quick Start

```python
import xarray as xr
import xrturb

# Load your PIV data
ds = xr.open_dataset('your_data.nc')

# Calculate turbulent kinetic energy
tke = ds.turb.tke()

# Compute Reynolds stresses
stresses = ds.turb.reynolds_stress()

# Apply Gaussian filter
filtered = ds.turb.filter_gaussian(sigma=2)
```

## Documentation

Full documentation is available at: [https://xrturb.readthedocs.io/](https://xrturb.readthedocs.io/)

To build documentation locally:

```bash
cd docs
sphinx-build -b html . _build/html
```

Then open `docs/_build/html/index.html` in your browser.

## License

MIT License
