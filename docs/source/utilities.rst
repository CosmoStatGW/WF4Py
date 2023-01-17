.. _utilities:

Utility functions
=================

We here list and describe the utility functions and definitions provided in the :py:class:`WF4Py.WFutils` module.

Convert between parameters
--------------------------

``WF4Py`` provides some useful functions to convert between different parameters. All of them are vectorised, and can thus be used on arrays containing the parameters of multiple events.

Tidal deformability parameters
""""""""""""""""""""""""""""""

Conversions between the individual tidal deformabilities of the two objects :math:`\Lambda_1` and :math:`\Lambda_2` and the combinations :math:`\tilde{\Lambda}` and :math:`\delta\tilde{\Lambda}` (see the :ref:`definition <note_labdatildeDef>`)

.. autofunction:: WF4Py.WFutils.Lamt_delLam_from_Lam12

.. autofunction:: WF4Py.WFutils.Lam12_from_Lamt_delLam

Masses
""""""

Conversions between the component masses and the chirp mass and symmetric mass ratio.

.. autofunction:: WF4Py.WFutils.m1m2_from_Mceta

.. autofunction:: WF4Py.WFutils.Mceta_from_m1m2

Constants
---------

We here list the constants defined in the :py:class:`WF4Py.WFutils` module

.. autodata:: WF4Py.WFutils.uGpc

.. autodata:: WF4Py.WFutils.uMsun

.. autodata:: WF4Py.WFutils.clight

.. autodata:: WF4Py.WFutils.clightGpc

.. autodata:: WF4Py.WFutils.GMsun_over_c3

.. autodata:: WF4Py.WFutils.GMsun_over_c2

.. autodata:: WF4Py.WFutils.GMsun_over_c2_Gpc

.. autodata:: WF4Py.WFutils.REarth

.. autodata:: WF4Py.WFutils.f_isco

.. autodata:: WF4Py.WFutils.f_qK
