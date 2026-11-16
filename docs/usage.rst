Usage
=====

.. _usage:

Quick start (Python API)
------------------------

The library has **one entry point**, :func:`cwaves.compute_cwaves`. Give it the
spectrum as an ``xarray.DataArray`` carrying its wavenumber coordinates; the
grid is detected and the spectrum is used **as sampled** (no resampling):

.. code-block:: python

   import cwaves

   # xs: cross-spectrum with 1-D coordinates k_rg, k_az (rad/m)
   params = cwaves.compute_cwaves(xs)                    # 4 x 5, 25-600 m
   params = cwaves.compute_cwaves(xs, cwave_Nk=8, cwave_Nphi=9)
   print(params.attrs["grid"], params.attrs["coverage"])  # e.g. cartesian half

Supported inputs
----------------

.. list-table::
   :header-rows: 1

   * - grid
     - coordinates (1-D each, own dimension)
     - examples
   * - cartesian, uniform or not
     - ``k_rg`` (range), ``k_az`` (azimuth), rad/m
     - SARWave L1B XSP (FFT grid), OCN ``oswCartSpec`` (sinh grid, convert
       cycles/m to rad/m)
   * - polar, linear or log ``k``
     - ``k`` (rad/m), ``phi`` (**radians**, ``k_rg = k cos(phi)``)
     - OCN ``oswPolSpec`` (convert ``oswPhi`` from degrees), model spectra

Each grid may cover the **half plane** ``k_rg >= 0`` (``phi`` in
:math:`[-\pi/2, \pi/2]`) or the **full plane** (``k_rg`` symmetric around 0,
or the full circle). On the full plane the result is the CWAVE parameters of
the centrally-symmetric part of the spectrum (identical to the half-plane
result for a Hermitian cross-spectrum) -- see :ref:`theory`.

Any extra dimension (``pol``, ``time``, ``tile``...) is broadcast; a wavenumber
axis that varies along an extra dimension (one ``k_rg`` per tile) is handled
slice by slice.

Explicit errors are raised for: an unsupported grid (no ``k_rg``/``k_az`` nor
``k``/``phi`` coordinates), ``phi`` in degrees, NaN or non-monotonic
coordinates, a partial plane coverage (e.g. a few negative ``k_rg`` only), or
a spectrum with zero mass in the wavenumber annulus.

Polar example (ESA OCN WV ``oswPolSpec``):

.. code-block:: python

   import numpy as np
   import xarray as xr
   import cwaves

   with xr.open_dataset("s1a-wv1-ocn-vv.nc") as ds:
       spec = xr.DataArray(
           ds["oswPolSpec"].isel(oswAzSize=0, oswRaSize=0).values,  # (phi, k)
           dims=("phi", "k"),
           coords={"phi": np.deg2rad(ds["oswPhi"].values), "k": ds["oswK"].values},
       )
   params = cwaves.compute_cwaves(spec)   # native 72 x 60 polar grid

(:func:`cwaves.client.polar_spectrum` does exactly this.)

Complex spectra
---------------

``part`` selects what is projected (see :ref:`complex-cwave`):

.. code-block:: python

   cwaves.compute_cwaves(xs)                    # |xs| on even harmonics (paper)
   cwaves.compute_cwaves(xs, part="real")       # Re xs on even harmonics
   cwaves.compute_cwaves(xs, part="imag")       # Im xs on odd harmonics
   c = cwaves.compute_cwaves(xs, part="complex")  # Re -> even, Im -> odd
   c.real, c.imag

Parameters
----------

- ``cwave_wlmax`` / ``cwave_wlmin`` : maximum / minimum wavelength in meters
  (default 600 / 25). They set :math:`k_{min}=2\pi/\lambda_{max}`,
  :math:`k_{max}=2\pi/\lambda_{min}`.
- ``cwave_Nk`` : number of Gegenbauer (radial) functions (default 4).
- ``cwave_Nphi`` : number of harmonic (angular) functions (default 5).
- ``part`` : ``"modulus"`` (default), ``"real"``, ``"imag"``, ``"complex"``.
- ``save_cwave_kernel`` : if ``True``, write the kernels to ``cwaves_kernel.nc``.

:func:`cwaves.compute_cwave_parameters` keeps the historical ``xsar_slc``
signature and calls :func:`cwaves.compute_cwaves`.

Configuration
-------------

The ``cwave_*`` defaults live in :file:`cwaves/default_config.yml` and can be
loaded with :func:`cwaves.load_config` / :func:`cwaves.get_default_config` :

.. code-block:: python

   import cwaves

   cwave_block = cwaves.get_default_config()["cwave"]
   params = cwaves.compute_cwaves(xs, **cwave_block)

.. _axis-convention-usage:

Axis convention
---------------

:math:`k_x` is the range wavenumber (``k_rg``) and :math:`k_y` the azimuth
wavenumber (``k_az``); polar angles are measured from the range axis
(``k_rg = k cos(phi)``, ``k_az = k sin(phi)``). See :ref:`axis-convention`.

Command-line interfaces
-----------------------

``cwaves`` : CWAVE of the spectrum stored in a NetCDF file
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

One input, ``--input-nc``; the schema is detected from the variables of the
file (never from its name):

* ``xspectra_{lag}tau_Re/Im`` on ``k_rg``/``k_az`` (e.g. SARWave L1B XSP WV):
  one parameter set per ``time`` slice (``--lag``, ``--pol``);
* ``oswPolSpec`` on ``oswK``/``oswPhi`` (e.g. ESA OCN WV): native polar grid.

Tiled products whose spectra live in NetCDF groups (IW ``intraburst`` /
``interburst``) are rejected with a pointer to the IW map script below.

.. code-block:: bash

   cwaves --input-nc s1a_wv1_xsp.nc --lag 2 --pol 0 --output-nc cwave.nc
   cwaves --input-nc s1a-wv1-ocn-vv.nc --cwave-nk 8 --cwave-nphi 9
   # complex CWAVE (written as cwave_params_re / cwave_params_im)
   cwaves --input-nc s1a_wv1_xsp.nc --part complex --output-nc cwave_c.nc

``cwaves-kernels`` : render the kernel basis
+++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: bash

   cwaves-kernels --output-png kernels.png              # full basis
   cwaves-kernels --k-gp 1 --phi-hf 1 --output-png k1_phi1.png

IW subswath map (``python -m cwaves.scripts.iw_cwaves_map``)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Compute CWAVE tile by tile on the IW subswaths (``iw1``/``iw2``/``iw3``) of a
Sentinel-1 ``.SAFE`` (OCN IW ``oswCartSpec`` on its native sinh grid, or
SARWAVE L1B IW XSP) and render a georeferenced map; the output size follows
``--cwave-nk`` x ``--cwave-nphi``:

.. code-block:: bash

   python -m cwaves.scripts.iw_cwaves_map --input-safe product.SAFE \
       --group intraburst --lag 2 --pol VV --k-gp 1 --phi-hf 1

The map figure requires the ``scripts`` extra (``cartopy``) :
``pip install "cwaves[scripts]"``.
