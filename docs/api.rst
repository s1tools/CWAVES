API reference
=============

The public API is exposed at the top level of the :mod:`cwaves` package.

Core
----

.. autofunction:: cwaves.compute_cwave_parameters

.. autofunction:: cwaves.compute_kernel

.. autofunction:: cwaves.gegenbauer_polynoms

.. autofunction:: cwaves.harmonic_functions

Clients (Sentinel-1 products)
-----------------------------

.. autofunction:: cwaves.compute_cwaves_from_l1b

.. autofunction:: cwaves.compute_cwaves_from_osw

.. autofunction:: cwaves.client.polar_to_cartesian

Utilities
---------

.. autofunction:: cwaves.symmetrize_xspectrum

.. autofunction:: cwaves.get_default_config

.. autofunction:: cwaves.load_config

Scripts (console entry points)
------------------------------

The scripts are importable modules with an :func:`~entrypoint` callable.

* :mod:`cwaves.scripts.plot_kernels` — console ``cwaves-kernels``. Render the
  orthonormal CWAVE kernel basis in the (:math:`k_x`, :math:`k_y`) wavenumber
  domain.

  .. autofunction:: cwaves.scripts.plot_kernels.plot_kernels

  .. autofunction:: cwaves.scripts.plot_kernels.build_kernel_grid

  .. autofunction:: cwaves.scripts.plot_kernels.domain_mask

* :mod:`cwaves.scripts.iw_cwaves_map` — console via ``python -m
  cwaves.scripts.iw_cwaves_map``. Compute CWAVE on the IW subswaths of a
  Sentinel-1 L2 product and render a georeferenced map.

  .. autofunction:: cwaves.scripts.iw_cwaves_map.compute_ocn_subswath

  .. autofunction:: cwaves.scripts.iw_cwaves_map.compute_sarwave_subswath

  .. autofunction:: cwaves.scripts.iw_cwaves_map.resample_sinh_to_regular

  .. autofunction:: cwaves.scripts.iw_cwaves_map.plot_map
