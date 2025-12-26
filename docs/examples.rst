Examples
========

Basic Turbulence Analysis
-------------------------

This example shows how to perform basic turbulence analysis on PIV data:

.. code-block:: python

   import numpy as np
   import xarray as xr
   import xrturb

   # Create sample PIV data
   x = np.linspace(0, 1, 50)
   y = np.linspace(0, 1, 50)
   t = np.linspace(0, 10, 100)

   X, Y, T = np.meshgrid(x, y, t, indexing='ij')

   # Generate synthetic turbulent velocity field
   u = 1.0 + 0.1 * np.sin(2 * np.pi * X) + 0.05 * np.random.randn(*X.shape)
   v = 0.1 * np.cos(2 * np.pi * Y) + 0.05 * np.random.randn(*X.shape)

   # Create xarray Dataset
   ds = xr.Dataset(
       {
           'u': (['x', 'y', 'time'], u),
           'v': (['x', 'y', 'time'], v),
       },
       coords={'x': x, 'y': y, 'time': t}
   )

   # Calculate mean flow
   mean_flow = ds.turb.mean()

   # Calculate fluctuations
   u_fluct = ds.turb.fluct('u')
   v_fluct = ds.turb.fluct('v')

   # Calculate TKE
   tke = ds.turb.tke()
   print(f"Mean TKE: {tke.mean().values:.3f} m²/s²")

   # Add all fluctuation products to dataset
   ds_products = ds.turb.calc_all_products(order=3)
   print(f"Added {len(ds_products.data_vars) - len(ds.data_vars)} product variables")

Reynolds Stress Analysis
=======================

Calculate and visualize Reynolds stresses:

.. code-block:: python

   # Calculate Reynolds stress tensor
   rss = ds.turb.reynolds_stress()

   # Access individual components
   uv_stress = rss['uv']
   uu_stress = rss['uu']  # This will be variance of u

   # Plot contours
   import matplotlib.pyplot as plt

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   uv_stress.plot(ax=ax1, cmap='RdBu_r')
   ax1.set_title('Reynolds Shear Stress <u\'v\'>')

   tke.plot(ax=ax2, cmap='viridis')
   ax2.set_title('Turbulent Kinetic Energy')

   plt.tight_layout()
   plt.show()

Higher-Order Statistics
=======================

Compute skewness and kurtosis:

.. code-block:: python

   # Calculate standardized moments up to order 4
   moments = ds.turb.calc_moments(order=4, standardized=True)

   # Skewness of u (should be ~0 for Gaussian)
   skewness_u = moments['uuu'].mean()
   print(f"Skewness of u: {skewness_u.values:.3f}")

   # Kurtosis of u (should be ~3 for Gaussian)
   kurtosis_u = moments['uuuu'].mean()
   print(f"Kurtosis of u: {kurtosis_u.values:.3f}")

Spatial Analysis
===============

Compute vorticity and strain rates:

.. code-block:: python

   # Calculate vorticity
   vorticity = ds.turb.vorticity()

   # Calculate strain rate components
   strain = ds.turb.strain()

   # Plot vorticity field
   vorticity.plot(cmap='RdBu_r', vmin=-5, vmax=5)
   plt.title('Vorticity Field')
   plt.show()

Filtering Data
==============

Apply filters to reduce noise:

.. code-block:: python

   # Apply Gaussian filter
   sigma = 1.0  # filter width
   filtered_ds = ds.turb.filter_gaussian(sigma=sigma)

   # Compare raw vs filtered TKE
   tke_raw = ds.turb.tke()
   tke_filtered = filtered_ds.turb.tke()

   # Plot comparison
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   tke_raw.plot(ax=ax1, vmin=0, vmax=0.01)
   ax1.set_title('Raw TKE')

   tke_filtered.plot(ax=ax2, vmin=0, vmax=0.01)
   ax2.set_title(f'Filtered TKE (σ={sigma})')

   plt.tight_layout()
   plt.show()