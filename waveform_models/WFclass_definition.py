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

import WFutils as utils

##############################################################################
# WaveFormModel CLASS DEFINITION
##############################################################################

class WaveFormModel(ABC):
    '''
    Abstract class to compute waveforms.
    '''
    
    def __init__(self, objType, fcutPar, is_tidal=False, is_HigherModes=False):
        # The kind of system the wf model is made for, can be 'BBH', 'BNS' or 'NSBH'
        self.objType = objType
        # The cut frequency factor of the waveform, in Hz, to be divided by Mtot (in units of Msun). The method fcut can be redefined, as e.g. in the IMRPhenomD implementation, and fcutPar can be passed as an adimensional frequency (Mf)
        self.fcutPar = fcutPar
        self.is_tidal=is_tidal
        self.is_HigherModes = is_HigherModes
        
    @abstractmethod
    def Phi(self, f, **kwargs):
        # The frequency of the GW, as a function of frequency
        # With reference to the book M. Maggiore - Gravitational Waves Vol. 1, with Phi we mean only
        # the GW frequency, not the full phase of the signal, given by
        # Psi+(f) = 2 pi f t_c - Phi0 - pi/4 - Phi(f)
        pass
    
    @abstractmethod
    def Ampl(self, f, **kwargs):
        # The amplitude of the signal as a function of frequency
        pass
        
    def tau_star(self, f, **kwargs):
        # The relation among the time to coalescence (in seconds) and the frequency (in Hz). We use as default
        # the expression in M. Maggiore - Gravitational Waves Vol. 1 eq. (4.21), valid in Newtonian and restricted PN approximation
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        return 2.18567 * ((1.21/kwargs['Mc'])**(5./3.)) * ((100/f)**(8./3.))
    
    def fcut(self, **kwargs):
        # The cut frequency of the waveform. In general this can be approximated as 2f_ISCO, but for complete waveforms
        # the expression is different
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        return self.fcutPar/(kwargs['Mc']/(kwargs['eta']**(3./5.)))

##############################################################################
# NEWTONIAN INSPIRAL WAVEFORM
##############################################################################

class NewtInspiral(WaveFormModel):
    '''
    Leading order (inspiral only) waveform model
    '''
    
    def __init__(self, **kwargs):
        # Cut from M. Maggiore - Gravitational Waves Vol. 2 eq. (14.106)
        # From T. Dietrich et al. Phys. Rev. D 99, 024029, 2019, below eq. (4) (also look at Fig. 1) it seems be that, for BNS in the non-tidal case, the cut frequency should be lowered to (0.04/(2.*pi*G*Msun/c3))/Mtot.
        super().__init__('BBH', 1./(6.*np.pi*np.sqrt(6.)*utils.GMsun_over_c3), **kwargs)
    
    def Phi(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        phase = 3.*0.25*(utils.GMsun_over_c3*kwargs['Mc']*8.*np.pi*f)**(-5./3.)
        return phase - np.pi*0.25
    
    def Ampl(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        amplitude = np.sqrt(5./24.) * (np.pi**(-2./3.)) * utils.clightGpc/kwargs['dL'] * (utils.GMsun_over_c3*kwargs['Mc'])**(5./6.) * (f**(-7./6.))
        return amplitude
