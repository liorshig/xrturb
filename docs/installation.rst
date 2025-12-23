Installation
============

Requirements
------------

xrturb requires Python 3.8+ and the following packages:

* xarray
* numpy
* scipy
* dask (optional, for parallel processing)

Installation from source
------------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/yourusername/xrturb.git
   cd xrturb
   pip install -e .

Using uv (recommended)
----------------------

.. code-block:: bash

   uv pip install -e .

For development with all dependencies:

.. code-block:: bash

   uv sync --dev