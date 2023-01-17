.. _wf_models:

Waveform models
===============

Waveform models give the prediction for a GW signal emitted by a coalescing binary as a function of the :ref:`parameters of the source <parameters_names>`.

``WF4Py`` provides Fourier domain waveform models.
In particular, it provides a Python implementation for some selected state-of-the-art models, written following the original implementations in the `LIGO Algorithm Library <https://wiki.ligo.org/Computing/LALSuite>`_, ``LAL``.

.. _wf_model_class:

The WaveFormModel class
-----------------------

Waveforms in ``WF4Py`` are built as classes, initialised as follows

.. autoclass:: WF4Py.waveform_models.WFclass_definition.WaveFormModel

Each waveform includes four fundamental methods, to compute, given the parameter of the source(s), the *phase* of the signal, :math:`\Phi (f)`, the *amplitude* of the signal, :math:`A (f)`, the *time to coalescence* as a function of the frequency, :math:`\tau^* (f)`, and the *cut frequency*.

.. automethod:: WF4Py.waveform_models.WFclass_definition.WaveFormModel.Phi

.. automethod:: WF4Py.waveform_models.WFclass_definition.WaveFormModel.Ampl

.. automethod:: WF4Py.waveform_models.WFclass_definition.WaveFormModel.tau_star

.. note::
  For :py:class:`WF4Py.waveform_models.WFclass_definition.NewtInspiral` we use the :math:`\tau^* (f)` expression in `M. Maggiore -- Gravitational Waves Vol. 1 <https://global.oup.com/academic/product/gravitational-waves-9780198570745?q=Michele%20Maggiore&lang=en&cc=it>`_ eq. (4.21).
  For the other models instead we use the :math:`\tau^* (f)` expression in `arXiv:0907.0700 <https://arxiv.org/abs/0907.0700>`_ eq. (3.8b), valid up to 3.5 PN.

.. automethod:: WF4Py.waveform_models.WFclass_definition.WaveFormModel.fcut

Some models also have a method denoted as :py:class:`hphc` to compute directly the :math:`\tilde{h}_+` and :math:`\tilde{h}_{\times}` polarisations of the gravitational wave, see below.

.. _wf_models_py:

Waveform models
---------------

We here report and briefly describe the waveform models implemented in pure Python in ``WF4Py``. We carefully checked that our Python implementation accurately reproduces the original ``LAL`` waveforms.

Leading order inspiral
""""""""""""""""""""""

This model includes only the leading order term in the inspiral.

.. autoclass:: WF4Py.waveform_models.WFclass_definition.NewtInspiral

.. _TF2:

TaylorF2
""""""""

.. versionadded:: 1.0.1
  Added the possibility to terminate the waveform at the ISCO frequency of a remnant Kerr BH.
  Added the spin-induced quadrupole due to tidal effects.

This model includes contributions to the phase up to 3.5 order in the *Post Newtonian*, PN, expansion, and can thus be used to describe the inspiral. The amplitude is the same as in Newtonian approximation.
Our implementation can include both the tidal terms at 5 and 6 PN (see `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_) and a moderate eccentricity in the orbit :math:`e_0\lesssim 0.1` up to 3 PN (see `arXiv:1605.00304 <https://arxiv.org/abs/1605.00304>`_).
There is no limitation in the parameters range, but, being an inspiral-only model, :py:class:`WF4Py.waveform_models.TaylorF2_RestrictedPN.TaylorF2_RestrictedPN` is better suited for BNS systems.

.. autoclass:: WF4Py.waveform_models.TaylorF2_RestrictedPN.TaylorF2_RestrictedPN

.. _IMRPhenomD:

IMRPhenomD
""""""""""

This is a full inspiral–merger-ringdown model tuned with NR simulations, which can be used to simulate signals coming from BBH mergers, with non–precessing spins up to :math:`|\chi_z|\sim 0.85` and mass ratios up to :math:`q = m_1/m_2 \sim 18`.

.. autoclass:: WF4Py.waveform_models.IMRPhenomD.IMRPhenomD

.. _IMRPhenomD_NRTidalv2:

IMRPhenomD_NRTidalv2
""""""""""""""""""""

This is a full inspiral-merger-ringdown model tuned with NR simulations, which extends the :ref:`IMRPhenomD` model to include tidal effects, and can thus be used to accurately describe signals coming from BNS mergers. The validity has been assessed for masses between :math:`1\,{\rm M}_{\odot}` and :math:`3\,{\rm M}_{\odot}`, spins up to :math:`|\chi_z|\sim 0.6` and tidal deformabilities up to :math:`\Lambda_i\sim 5000`.

.. autoclass:: WF4Py.waveform_models.IMRPhenomD_NRTidalv2.IMRPhenomD_NRTidalv2

The model includes a Planck taper filter to terminate the waveform after merger, we thus the cut frequency slightly before the end of the filter.

.. _IMRPhenomHM:

IMRPhenomHM
"""""""""""

This is a full inspiral–merger-ringdown model tuned with NR simulations, which takes into account not only the :math:`(2,2)` quadrupole of the signal, but also the sub–dominant multipoles :math:`(l,m) = (2,1),\, (3,2),\, (3,3),\, (4,3)`, and :math:`(4,4)`, that can be particularly relevant to better describe the signal coming from BBH systems. The calibration range is the same of the :ref:`IMRPhenomD` model.

.. autoclass:: WF4Py.waveform_models.IMRPhenomHM.IMRPhenomHM

To avoid for loops and recomputing coefficients (given that in this case the amplitude cannot be computed separately from the phase), this model features a :py:class:`WF4Py.waveform_models.IMRPhenomHM.IMRPhenomHM.hphc` method, to compute directly :math:`\tilde{h}_+` and :math:`\tilde{h}_{\times}`

.. automethod:: WF4Py.waveform_models.IMRPhenomHM.IMRPhenomHM.hphc

Also, in this case the :py:class:`WF4Py.waveform_models.IMRPhenomHM.IMRPhenomHM.Phi` and :py:class:`WF4Py.waveform_models.IMRPhenomHM.IMRPhenomHM.Ampl` methods return dictionaries containing the phases and amplitudes of the various modes, respectively. The dictionaries have keys ``'21'``, ``'22'``, ``'32'``, ``'33'``, ``'43'`` and ``'44'``.

.. automethod:: WF4Py.waveform_models.IMRPhenomHM.IMRPhenomHM.Phi

.. automethod:: WF4Py.waveform_models.IMRPhenomHM.IMRPhenomHM.Ampl

To combine the various modes and obtain the full amplitude and phase from these outputs, it is possible to use the function

.. autofunction:: WF4Py.WFutils.Add_Higher_Modes

.. _IMRPhenomNSBH:

IMRPhenomNSBH
"""""""""""""

This is a full inspiral–merger-ringdown model tuned with NR simulations, which can describe the signal coming from the merger of a NS and a BH, with mass ratios up to :math:`q = m_1/m_2 \sim 100`, also taking into account tidal effects and the impact of the possible tidal disruption of the NS.

.. autoclass:: WF4Py.waveform_models.IMRPhenomNSBH.IMRPhenomNSBH

.. note::

  In ``LAL``, to compute the parameter :math:`\xi_{\rm tide}` in `arXiv:1509.00512 <https://arxiv.org/abs/1509.00512>`_ eq. (8), the roots are extracted.
  In Python this would break the possibility to vectorise so, to circumvent the issue, we compute a grid of :math:`\xi_{\rm tide}` as a function of the compactness, mass ratio and BH spin, and then use a 3D interpolator.
  The first time the code runs, if this interpolator is not already present, it will be computed.
  The base resolution of the grid is 200 points per parameter, that we find sufficient to reproduce the ``LAL`` implementation with good precision, given the smooth behaviour of the function, but this can be raised if needed.
  In this case, it is necessary to change the name of the file assigned to ``self.path_xiTide_tab`` and the ``res`` input passed to the function that loads the grid.

  .. automethod:: WF4Py.waveform_models.IMRPhenomNSBH.IMRPhenomNSBH._make_xiTide_interpolator

.. _IMRPhenomD_NRTidalv2_Lorentzian:

.. _IMRPhenomXAS:

IMRPhenomXAS
""""""""""""

This is a full inspiral–merger-ringdown model tuned with NR simulations, which can be used to simulate signals coming from BBH mergers, with non–precessing spins up to :math:`|\chi_z|\sim 0.9` and mass ratios :math:`q = m_1/m_2` from 1 to 1000, among the last to be released.

.. autoclass:: WF4Py.waveform_models.IMRPhenomXAS.IMRPhenomXAS
