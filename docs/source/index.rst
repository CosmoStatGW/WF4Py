.. WF4Py documentation master file, created by
   sphinx-quickstart on Tue Jan  3 00:12:49 2023.

Welcome to WF4Py's documentation!
=================================

**WF4Py** is a user-friendly package implementing state-of-the-art frequency-domain gravitational wave waveform models in pure Python, thus enabling parallelisation over multiple events at a time.
All the waveforms are accurately checked with their implementation in the `LIGO Algorithm Library <https://wiki.ligo.org/Computing/LALSuite>`_, ``LAL``.

For further documentation refer to the release papers `arXiv:2207.02771 <https://arxiv.org/abs/2207.02771>`_ and `arXiv:2207.06910 <https://arxiv.org/abs/2207.06910>`_.

Have a look at `gwfast <https://github.com/CosmoStatGW/gwfast>`_ for a Fisher matrix code implementing the ``WF4Py`` models.

.. toctree::
   :maxdepth: 1
   :caption: Package description:

   installation
   events_parameters
   waveforms
   utilities

.. toctree::
 :maxdepth: 1
 :caption: Tutorial:

 notebooks/WF4Py_tutorial

.. toctree::
 :maxdepth: 1
 :caption: References:

 citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
