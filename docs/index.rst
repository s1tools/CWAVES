.. CWAVES documentation master file

Welcome to CWAVES
=================

``cwaves`` is a Python library to compute **CWAVE parameters** from ocean wave
and SAR spectra (cartesian or polar grids, half or full plane, real or
complex), implementing the orthonormal kernel decomposition of :

.. epigraph::

   J. Schulz-Stellenfleth, T. Konig, and S. Lehner, "An empirical approach for
   the retrieval of integral ocean wave parameters from synthetic aperture
   radar data", **J. Geophys. Res., 112, C03019, 2007.**
   `doi:10.1029/2006JC003970
   <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JC003970>`_

The CWAVE method retrieves integral ocean wave parameters from SAR data without
the intermediate step of retrieving the 2D ocean wave spectrum : the spectrum
is projected onto a set of orthonormal kernels built as tensor products of
Gegenbauer polynomials (wavenumber direction) and harmonic functions (angle
direction), weighted by the Jacobian of the wavenumber mapping. What the
decomposition can and cannot do is summarised in :ref:`limits`.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   theory
   limits
   proofs
   installation
   usage
   certification
   performance
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
