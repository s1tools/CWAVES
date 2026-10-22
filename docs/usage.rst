Usage
=====

Quick start
-----------

Compute the CWAVE parameters from a one-sided (positive range) complex
cross-spectrum :

.. code-block:: python

   import xarray as xr
   import cwaves

   xs = xr.open_dataset("measurement.nc")["cross_spectra"]

   cwave_params = cwaves.compute_cwave_parameters(
       xs, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5
   )

Parameters
----------

- ``cwave_wlmax`` / ``cwave_wlmin`` : maximum / minimum wavelength in meters.
- ``cwave_Nk`` : number of Gegenbauer (wavenumber) basis functions.
- ``cwave_Nphi`` : number of harmonic (angle) basis functions.

Axis convention
---------------

``kx = range`` (``k_rg``, dimension ``freq_sample``), ``ky = azimuth`` (``k_az``,
dimension ``freq_line``). The major axis of the kernel ellipses is in the range
direction.
