#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import numpy as np

from abc import ABC, abstractmethod
import os
import sys

from WF4Py import WFutils as utils

##############################################################################
# WaveFormModel CLASS DEFINITION
##############################################################################

class WaveFormModel(ABC):
    """
    Abstract class to compute waveforms
    
    :param str objType: The kind of system the wf model is made for, can be ``'BBH'``, ``'BNS'`` or ``'NSBH'``.
    :param float fcutPar: The cut frequency factor of the waveform. This can either be given in :math:`\\rm Hz`, as for :py:class:`WF4Py.waveform_models.TaylorF2_RestrictedPN.TaylorF2_RestrictedPN`, or as an adimensional frequency (Mf), as for the IMR models.
    :param bool, optional is_tidal: Boolean specifying if the waveform includes tidal effects.
    :param bool, optional is_HigherModes: Boolean specifying if the waveform includes the contribution of sub-dominant (higher-order) modes.
    :param bool, optional is_Precessing: Boolean specifying if the waveform includes spin-precession effects.
    :param bool, optional is_eccentric: Boolean specifying if the waveform includes orbital eccentricity.
    
    """
    
    def __init__(self, objType, fcutPar, is_tidal=False, is_HigherModes=False, is_Precessing=False, is_eccentric=False):
        """
        Constructor method
        """
        # The kind of system the wf model is made for, can be 'BBH', 'BNS' or 'NSBH'
        self.objType = objType
        # The cut frequency factor of the waveform, in Hz, to be divided by Mtot (in units of Msun). The method fcut can be redefined, as e.g. in the IMRPhenomD implementation, and fcutPar can be passed as an adimensional frequency (Mf)
        self.fcutPar = fcutPar
        self.is_tidal = is_tidal
        self.is_HigherModes = is_HigherModes
        self.is_Precessing = is_Precessing
        self.is_eccentric = is_eccentric
        
    @abstractmethod
    def Phi(self, f, **kwargs):
        """
        Compute the phase of the GW as a function of frequency, given the events parameters.

        We compute here only the GW phase, not the full phase of the signal, which also includes the reference phase and the time of coalescence.
        
        :param numpy.ndarray f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(numpy.ndarray, numpy.ndarray, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: GW phase for the chosen events evaluated on the frequency grid.
        :rtype: numpy.ndarray
        
        """
        # The frequency of the GW, as a function of frequency
        # With reference to the book M. Maggiore - Gravitational Waves Vol. 1, with Phi we mean only
        # the GW frequency, not the full phase of the signal, given by
        # Psi+(f) = 2 pi f t_c - Phi0 - pi/4 - Phi(f)
        pass
    
    @abstractmethod
    def Ampl(self, f, **kwargs):
        """
        Compute the amplitude of the GW as a function of frequency, given the events parameters.
        
        :param numpy.ndarray f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(numpy.ndarray, numpy.ndarray, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the amplitude of, as in :py:data:`events`.
        :return: GW amplitude for the chosen events evaluated on the frequency grid.
        :rtype: numpy.ndarray
        
        """
        pass
        
    def tau_star(self, f, **kwargs):
        # The relation among the time to coalescence (in seconds) and the frequency (in Hz). We use as default
        # the expression in M. Maggiore - Gravitational Waves Vol. 1 eq. (4.21), valid in Newtonian and restricted PN approximation
        """
        Compute the time to coalescence (in seconds) as a function of frequency (in :math:`\\rm Hz`), given the events parameters.
        
        :param numpy.ndarray f: Frequency grid on which the time to coalescence will be computed, in :math:`\\rm Hz`.
        :param dict(numpy.ndarray, numpy.ndarray, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
        :return: time to coalescence for the chosen events evaluated on the frequency grid, in seconds.
        :rtype: numpy.ndarray
        
        """
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        return 2.18567 * ((1.21/kwargs['Mc'])**(5./3.)) * ((100/f)**(8./3.))
    
    def fcut(self, **kwargs):
        # The cut frequency of the waveform. In general this can be approximated as 2f_ISCO, but for complete waveforms
        # the expression is different
        """
        Compute the cut frequency of the waveform as a function of the events parameters, in :math:`\\rm Hz`.
        
        :param dict(numpy.ndarray, numpy.ndarray, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the cut frequency of, as in :py:data:`events`.
        :return: Cut frequency of the waveform for the chosen events, in :math:`\\rm Hz`.
        :rtype: numpy.ndarray
        
        """
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        return self.fcutPar/(kwargs['Mc']/(kwargs['eta']**(3./5.)))

##############################################################################
# NEWTONIAN INSPIRAL WAVEFORM
##############################################################################

class NewtInspiral(WaveFormModel):
    """
    Leading order inspiral only waveform model.
    
    Relevant references: `M. Maggiore -- Gravitational Waves Vol. 1 <https://global.oup.com/academic/product/gravitational-waves-9780198570745?q=Michele%20Maggiore&lang=en&cc=it>`_, chapter 4.
        
    :param kwargs: Optional arguments to be passed to the parent class :py:class:`WaveFormModel`.
    
    """
    
    def __init__(self, **kwargs):
        """
        Constructor method
        """
        # Cut from M. Maggiore - Gravitational Waves Vol. 2 eq. (14.106)
        # From T. Dietrich et al. Phys. Rev. D 99, 024029, 2019, below eq. (4) (also look at Fig. 1) it seems be that, for BNS in the non-tidal case, the cut frequency should be lowered to (0.04/(2.*pi*G*Msun/c3))/Mtot.
        super().__init__('BBH', 1./(6.*np.pi*np.sqrt(6.)*utils.GMsun_over_c3), **kwargs)
    
    def Phi(self, f, **kwargs):
        """
        Compute the phase of the GW as a function of frequency, given the events parameters.

        We compute here only the GW phase, not the full phase of the signal, which also includes the reference phase and the time of coalescence.
        
        :param numpy.ndarray f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(numpy.ndarray, numpy.ndarray, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: GW phase for the chosen events evaluated on the frequency grid.
        :rtype: numpy.ndarray
        
        """
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        phase = 3.*0.25*(utils.GMsun_over_c3*kwargs['Mc']*8.*np.pi*f)**(-5./3.)
        return phase - np.pi*0.25
    
    def Ampl(self, f, **kwargs):
        """
        Compute the amplitude of the GW as a function of frequency, given the events parameters.
        
        :param numpy.ndarray f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(numpy.ndarray, numpy.ndarray, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the amplitude of, as in :py:data:`events`.
        :return: GW amplitude for the chosen events evaluated on the frequency grid.
        :rtype: numpy.ndarray
        
        """
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        amplitude = np.sqrt(5./24.) * (np.pi**(-2./3.)) * utils.clightGpc/kwargs['dL'] * (utils.GMsun_over_c3*kwargs['Mc'])**(5./6.) * (f**(-7./6.))
        return amplitude
