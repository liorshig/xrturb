Usage
=====

Basic Usage
-----------

Import xrturb and load your data:

.. code-block:: python

   import xarray as xr
   import xrturb

   # Load your PIV/LDV data
   ds = xr.open_dataset('your_data.nc')

   # Access turbulence methods
   ds.turb.mean()
   ds.turb.fluct('u')
   ds.turb.tke()
   ds.turb.reynolds_stress()

Turbulence Statistics
=====================

Calculate various turbulence statistics:

.. code-block:: python

   # Turbulent Kinetic Energy
   tke = ds.turb.tke()

   # Reynolds stress tensor
   rss = ds.turb.reynolds_stress()

   # Higher-order moments
   moments = ds.turb.calc_moments(order=3)

   # Raw fluctuation products (without averaging)
   ds_with_products = ds.turb.calc_all_products(order=3)

Spatial Derivatives
===================

Compute vorticity, divergence, and strain rates:

.. code-block:: python

   # Vorticity components
   vorticity = ds.turb.vorticity()

   # Strain rate tensor
   strain = ds.turb.strain()

   # Divergence
   div = ds.turb.divergence()

Filtering and Smoothing
=======================

Apply various filters to your data:

.. code-block:: python

   # Gaussian filtering
   filtered = ds.turb.filter_gaussian(sigma=2)

   # Savitzky-Golay smoothing
   smoothed = ds.turb.filter_savgol(window_length=5, polyorder=2)

Utility Functions
=================

Data manipulation utilities:

.. code-block:: python

   # Fill NaN values
   filled = ds.turb.fill_nan(method='linear')

   # Crop data
   cropped = ds.turb.crop(x_slice=slice(10, 50), y_slice=slice(20, 80))

   # Rotate coordinates
   rotated = ds.turb.rotate(angle=45)