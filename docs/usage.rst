Usage
=====

.. _usage:

Quick start (Python API)
------------------------

Compute the CWAVE parameters from a **one-sided (positive range)** complex
cross-spectrum :

.. code-block:: python

   import xarray as xr
   import cwaves

   # one-sided (k_rg >= 0) cross-spectrum with k_rg / k_az coordinates
   xs = xr.open_dataset("measurement.nc")["cross_spectra"]

   cwave_params = cwaves.compute_cwave_parameters(
       xs, cwave_wlmax=600, cwave_wlmin=25, cwave_Nk=4, cwave_Nphi=5
   )

   print(cwave_params)  # DataArray, dims (k_gp, phi_hf)

Input contract
--------------

As in the 2007 paper, the input must be a **one-sided (half)** cross-spectrum :

* **one-sided in range** : :math:`k_{rg} \ge 0` (a full / symmetrized spectrum
  is rejected with a ``ValueError``),
* **two-sided in azimuth** : :math:`k_{az}` spans negative and positive,
* **real modulus** : only :math:`|xs|` is used (the complex phase is discarded),
* **uniform (regular) wavenumber grid** in both :math:`k_{rg}` and :math:`k_{az}`
  (the projection uses a constant cell area, exact only for a uniform grid).

See :ref:`theory` and ``repository_specifications.md`` (section 6) for the full
rationale.

Parameters
----------

- ``cwave_wlmax`` / ``cwave_wlmin`` : maximum / minimum wavelength in meters
  (default 600 / 25). They set the wavenumber bounds
  :math:`k_{min}=2\pi/\lambda_{max}`, :math:`k_{max}=2\pi/\lambda_{min}`.
- ``cwave_Nk`` : number of Gegenbauer (wavenumber) basis functions (default 4).
- ``cwave_Nphi`` : number of harmonic (angle) basis functions (default 5).
- ``save_cwave_kernel`` : if ``True``, write the kernels to ``cwaves_kernel.nc``.

Configuration
-------------

The ``cwave_*`` defaults live in :file:`cwaves/default_config.yml` and can be
loaded with :func:`cwaves.load_config` / :func:`cwaves.get_default_config` :

.. code-block:: python

   import cwaves

   cfg = cwaves.get_default_config()
   cwave_block = cfg["cwave"]          # cwave_wlmax, cwave_wlmin, cwave_Nk, ...
   params = cwaves.compute_cwave_parameters(xs, **cwave_block)

.. _axis-convention-usage:

Axis convention
---------------

:math:`k_x` is the range wavenumber (``k_rg``, dimension ``freq_sample``) and
:math:`k_y` is the azimuth wavenumber (``k_az``, dimension ``freq_line``). The
major axis of the kernel ellipses is in the **range** direction. See
:ref:`axis-convention`.

Command-line interfaces
-----------------------

``cwaves`` ships three console scripts.

``cwaves-osw`` : CWAVE from a Sentinel-1 product
++++++++++++++++++++++++++++++++++++++++++++++++

Two mutually-exclusive inputs :

* ``--input-nc-l1b`` : SARWave L1B XSP cross-spectrum product (primary, no
  ``s1_int2.xml``). One CWAVE parameter set per time slice.
* ``--input-nc-osw`` : ESA OCN WV Sentinel-1 measurement file (the polar
  ``oswPolSpec`` is resampled to a cartesian cross-spectrum).

.. code-block:: bash

   cwaves-osw --input-nc-l1b l1b-s1a-iw2-xsp.nc --lag 2 --pol 0
   cwaves-osw --input-nc-osw s1d-wv1-ocn-vv.nc --output-nc cwave.nc

``cwaves-kernels`` : render the kernel basis
+++++++++++++++++++++++++++++++++++++++++++++

Render the orthonormal kernel basis in the (:math:`k_x`, :math:`k_y`)
wavenumber domain (elliptic domain :math:`A` masked outside), as a grid of
:math:`N_k` x :math:`N_\varphi` panels :

.. code-block:: bash

   # full basis
   cwaves-kernels --output-png kernels.png
   # a single kernel
   cwaves-kernels --k-gp 1 --phi-hf 1 --output-png k1_phi1.png

IW subswath map (``python -m cwaves.scripts.iw_cwaves_map``)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Compute CWAVE on the IW subswaths (``iw1``/``iw2``/``iw3``) of a Sentinel-1 L2
product (``.SAFE``) and render a single georeferenced map. Both the OCN IW
measurement and the SARWAVE L1B XSP families are auto-detected :

.. code-block:: bash

   python -m cwaves.scripts.iw_cwaves_map --input-safe product.SAFE \
       --group intraburst --lag 2 --pol VV --k-gp 1 --phi-hf 1

The map figure requires the ``scripts`` extra (``cartopy``) :
``pip install "cwaves[scripts]"``.
