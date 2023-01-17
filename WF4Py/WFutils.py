#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import numpy as np
import os

##############################################################################
# PATH TO WFfiles FOLDER
##############################################################################

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
"""
Path to the ``WF4Py`` directory.

:type: str
"""
WFfilesPATH = os.path.join(SCRIPT_DIR, 'WFfiles')
"""
Path to the ``WF4Py/WFfiles`` directory, containing files needed for the waveform evaluation.

:type: str
"""
##############################################################################
# PHYSICAL CONSTANTS
##############################################################################
# See http://asa.hmnao.com/static/files/2021/Astronomical_Constants_2021.pdf

GMsun_over_c3 = 4.925491025543575903411922162094833998e-6 # seconds
"""
Geometrized solar mass :math:`G \, {\\rm M}_{\odot} / c^3`, in seconds (:math:`\\rm s`).

:type: float
"""
GMsun_over_c2 = 1.476625061404649406193430731479084713e3 # meters
"""
Geometrized solar mass :math:`G \, {\\rm M}_{\odot} / c^2`, in meters (:math:`\\rm m`).

:type: float
"""
uGpc = 3.085677581491367278913937957796471611e25 # meters
"""
Gigaparsec (:math:`\\rm Gpc`) in meters (:math:`\\rm m`).

:type: float
"""
uMsun = 1.988409902147041637325262574352366540e30 # kg
"""
Solar mass (:math:`{\\rm M}_{\odot}`) in kilograms (:math:`\\rm kg`).

:type: float
"""
GMsun_over_c2_Gpc = GMsun_over_c2/uGpc # Gpc
"""
Geometrized solar mass :math:`G \, {\\rm M}_{\odot} / c^2`, in gigaparsec (:math:`\\rm Gpc`).

:type: float
"""
REarth = 6371.00 #km
"""
Average Earth radius, in kilometers (:math:`\\rm km`).

:type: float
"""
clight = 2.99792458*(10**5) #km/s
"""
Speed of light in vacuum (:math:`c`), in kilometers per second (:math:`\\rm km / s`).

:type: float
"""
clightGpc = clight/3.0856778570831e+22
"""
Speed of light in vacuum (:math:`c`), in gigaparsecs per second (:math:`\\rm Gpc / s`).

:type: float
"""

# ISCO frequency coefficient for a Schwarzschild BH
f_isco=1./(np.sqrt(6.)*6.*2.*np.pi*GMsun_over_c3)
"""
ISCO frequency coefficient for a Schwarzschild BH, in :math:`\\rm Hz`.

:type: float
"""
# limit of the quasi-Keplerian approximation, as in arXiv:2108.05861 (see also arXiv:1605.00304), more conservative than the Schwarzschild ISCO
f_qK = 2585. # Hz
"""
Coefficient for the limit of the quasi-Keplerian approximation, in :math:`\\rm Hz`, as in `arXiv:2108.05861 <https://arxiv.org/abs/2108.05861>`_ (see also `arXiv:1605.00304 <https://arxiv.org/abs/1605.00304>`_). This is more conservative than two times the Schwarzschild ISCO.

:type: float
"""

##############################################################################
# TIDAL PARAMETERS
##############################################################################

def Lamt_delLam_from_Lam12(Lambda1, Lambda2, eta):
    """
    Compute the dimensionless tidal deformability combinations :math:`\\tilde{\Lambda}` and :math:`\delta\\tilde{\Lambda}`, defined in `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_ eq. (5) and (6), as a function of the dimensionless tidal deformabilities of the two objects and the symmetric mass ratio.
    
    :param numpy.ndarray or float Lambda1: Tidal deformability of object 1, :math:`\Lambda_1`.
    :param numpy.ndarray or float Lambda2: Tidal deformability of object 2, :math:`\Lambda_2`.
    :param numpy.ndarray or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`\\tilde{\Lambda}` and :math:`\delta\\tilde{\Lambda}`.
    :rtype: tuple(numpy.ndarray, numpy.ndarray) or tuple(float, float)
    
    """
    eta2 = eta*eta
    # This is needed to stabilize JAX derivatives
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        
    Lamt = (8./13.)*((1. + 7.*eta - 31.*eta2)*(Lambda1 + Lambda2) + Seta*(1. + 9.*eta - 11.*eta2)*(Lambda1 - Lambda2))
    
    delLam = 0.5*(Seta*(1. - 13272./1319.*eta + 8944./1319.*eta2)*(Lambda1 + Lambda2) + (1. - 15910./1319.*eta + 32850./1319.*eta2 + 3380./1319.*eta2*eta)*(Lambda1 - Lambda2))
    
    return Lamt, delLam
    
def Lam12_from_Lamt_delLam(Lamt, delLam, eta):
    """
    Compute the dimensionless tidal deformabilities of the two objects as a function of the dimensionless tidal deformability combinations :math:`\\tilde{\Lambda}` and :math:`\delta\\tilde{\Lambda}`, defined in `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_ eq. (5) and (6), and the symmetric mass ratio.
    
    :param numpy.ndarray or float Lamt: Tidal deformability combination :math:`\\tilde{\Lambda}`.
    :param numpy.ndarray or float delLam: Tidal deformability combination :math:`\delta\\tilde{\Lambda}`.
    :param numpy.ndarray or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`\Lambda_1` and :math:`\Lambda_2`.
    :rtype: tuple(numpy.ndarray, numpy.ndarray) or tuple(float, float)
    
    """
    eta2 = eta*eta
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    
    mLp=(8./13.)*(1.+ 7.*eta-31.*eta2)
    mLm=(8./13.)*Seta*(1.+ 9.*eta-11.*eta2)
    mdp=Seta*(1.-(13272./1319.)*eta+(8944./1319.)*eta2)*0.5
    mdm=(1.-(15910./1319.)*eta+(32850./1319.)*eta2+(3380./1319.)*(eta2*eta))*0.5

    det=(306656./1319.)*(eta**5)-(5936./1319.)*(eta**4)

    Lambda1 = ((mdp-mdm)*Lamt+(mLm-mLp)*delLam)/det
    Lambda2 = ((-mdm-mdp)*Lamt+(mLm+mLp)*delLam)/det
    
    return Lambda1, Lambda2

##############################################################################
# MASSES
##############################################################################

def m1m2_from_Mceta(Mc, eta):
    """
    Compute the component masses of a binary given its chirp mass and symmetric mass ratio.
    
    :param numpy.ndarray or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`.
    :param numpy.ndarray or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`m_1` and :math:`m_2`.
    :rtype: tuple(numpy.ndarray, numpy.ndarray) or tuple(float, float)
    
    """
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    m1 = 0.5*(Mc/(eta**(3./5.)))*(1. + Seta)
    m2 = 0.5*(Mc/(eta**(3./5.)))*(1. - Seta)

    return m1, m2
    
def Mceta_from_m1m2(m1, m2):
    """
    Compute the chirp mass and symmetric mass ratio of a binary given its component masses.
    
    :param numpy.ndarray or float m1: Mass of the primary object, :math:`m_1`.
    :param numpy.ndarray or float m2: Mass of the secondary object, :math:`m_2`.
    :return: :math:`{\cal M}_c` and :math:`\eta`.
    :rtype: tuple(numpy.ndarray, numpy.ndarray) or tuple(float, float)
    
    """
    Mc  = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
    eta = (m1*m2)/((m1+m2)*(m1+m2))
    
    return Mc, eta
    
##############################################################################
# SPHERICAL HARMONICS
##############################################################################

def Add_Higher_Modes(Ampl, Phi, iota, phi=0.):
    """
    Compute the total signal from a collection of different modes.
    
    :param dict(numpy.ndarray, numpy.ndarray, ...) Ampl: Dictionary containing the amplitudes for each mode computed on a grid of frequencies. The keys are expected to be stings made up of :math:`l` and :math:`m`, e.g. for :math:`(2,2)` --> key= ``'22'``.
    :param dict(numpy.ndarray, numpy.ndarray, ...) Phi: Dictionary containing the phases for each mode computed on a grid of frequencies.
    :param numpy.ndarray or float iota: The inclination angle(s) of the system(s) with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`.
    :param numpy.ndarray or float phi: The second angular direction of the spherical coordinate system.
    :return: Plus and cross polarisations of the GW for the chosen events evaluated on the frequency grid.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    
    """
    # Function to compute the total signal from a collection of different modes
    # Ampl and Phi have to be dictionaries containing the amplitudes and phases, computed on a grid of frequencies, for
    # each mode. The keys are expected to be stings made up of l and m, e.g. for (2,2) -> key='22'
    
    def SpinWeighted_SphericalHarmonic(theta, phi, l, m, s=-2):
        # Taken from arXiv:0709.0093v3 eq. (II.7), (II.8) and LALSimulation for the s=-2 case and up to l=4
        
        if s != -2:
            raise ValueError('The only spin-weight implemented for the moment is s = -2.')
            
        if (2 == l):
            if (-2 == m):
                res = np.sqrt( 5.0 / ( 64.0 * np.pi ) ) * ( 1.0 - np.cos( theta ))*( 1.0 - np.cos( theta ))
            elif (-1 == m):
                res = np.sqrt( 5.0 / ( 16.0 * np.pi ) ) * np.sin( theta )*( 1.0 - np.cos( theta ))
            elif (0 == m):
                res = np.sqrt( 15.0 / ( 32.0 * np.pi ) ) * np.sin( theta )*np.sin( theta )
            elif (1 == m):
                res = np.sqrt( 5.0 / ( 16.0 * np.pi ) ) * np.sin( theta )*( 1.0 + np.cos( theta ))
            elif (2 == m):
                res = np.sqrt( 5.0 / ( 64.0 * np.pi ) ) * ( 1.0 + np.cos( theta ))*( 1.0 + np.cos( theta ))
            else:
                raise ValueError('Invalid m for l = 2.')
                
        elif (3 == l):
            if (-3 == m):
                res = np.sqrt(21.0/(2.0*np.pi))*np.cos(theta*0.5)*((np.sin(theta*0.5))**(5.))
            elif (-2 == m):
                res = np.sqrt(7.0/(4.0*np.pi))*(2.0 + 3.0*np.cos(theta))*((np.sin(theta*0.5))**(4.0))
            elif (-1 == m):
                res = np.sqrt(35.0/(2.0*np.pi))*(np.sin(theta) + 4.0*np.sin(2.0*theta) - 3.0*np.sin(3.0*theta))/32.0
            elif (0 == m):
                res = (np.sqrt(105.0/(2.0*np.pi))*np.cos(theta)*(np.sin(theta)*np.sin(theta)))*0.25
            elif (1 == m):
                res = -np.sqrt(35.0/(2.0*np.pi))*(np.sin(theta) - 4.0*np.sin(2.0*theta) - 3.0*np.sin(3.0*theta))/32.0
            elif (2 == m):
                res = np.sqrt(7.0/np.pi)*((np.cos(theta*0.5))**(4.0))*(-2.0 + 3.0*np.cos(theta))*0.5
            elif (3 == m):
                res = -np.sqrt(21.0/(2.0*np.pi))*((np.cos(theta/2.0))**(5.0))*np.sin(theta*0.5)
            else:
                raise ValueError('Invalid m for l = 3.')
                
        elif (4 == l):
            if (-4 == m):
                res = 3.0*np.sqrt(7.0/np.pi)*(np.cos(theta*0.5)*np.cos(theta*0.5))*((np.sin(theta*0.5))**6.0)
            elif (-3 == m):
                res = 3.0*np.sqrt(7.0/(2.0*np.pi))*np.cos(theta*0.5)*(1.0 + 2.0*np.cos(theta))*((np.sin(theta*0.5))**5.0)
            elif (-2 == m):
                res = (3.0*(9.0 + 14.0*np.cos(theta) + 7.0*np.cos(2.0*theta))*((np.sin(theta/2.0))**4.0))/(4.0*np.sqrt(np.pi))
            elif (-1 == m):
                res = (3.0*(3.0*np.sin(theta) + 2.0*np.sin(2.0*theta) + 7.0*np.sin(3.0*theta) - 7.0*np.sin(4.0*theta)))/(32.0*np.sqrt(2.0*np.pi))
            elif (0 == m):
                res = (3.0*np.sqrt(5.0/(2.0*np.pi))*(5.0 + 7.0*np.cos(2.0*theta))*(np.sin(theta)*np.sin(theta)))/16.
            elif (1 == m):
                res = (3.0*(3.0*np.sin(theta) - 2.0*np.sin(2.0*theta) + 7.0*np.sin(3.0*theta) + 7.0*np.sin(4.0*theta)))/(32.0*np.sqrt(2.0*np.pi))
            elif (2 == m):
                res = (3.0*((np.cos(theta*0.5))**4.0)*(9.0 - 14.0*np.cos(theta) + 7.0*np.cos(2.0*theta)))/(4.0*np.sqrt(np.pi))
            elif (3 == m):
                res = -3.0*np.sqrt(7.0/(2.0*np.pi))*((np.cos(theta*0.5))**5.0)*(-1.0 + 2.0*np.cos(theta))*np.sin(theta*0.5)
            elif (4 == m):
                res = 3.0*np.sqrt(7.0/np.pi)*((np.cos(theta*0.5))**6.0)*(np.sin(theta*0.5)*np.sin(theta*0.5))
            else:
                raise ValueError('Invalid m for l = 4.')
                
        else:
            raise ValueError('Multipoles with l > 4 not implemented yet.')
        
        return res*np.exp(1j*m*phi)
    
    hp = np.zeros(Ampl[list(Ampl)[0]].shape)
    hc = np.zeros(Ampl[list(Ampl)[0]].shape)
    
    for key in Ampl.keys():
        if key in Phi.keys():
            l, m = int(key[:2//2]), int(key[2//2:])
            Y = SpinWeighted_SphericalHarmonic(iota, phi, l, m)
            if m:
                Ymstar = np.conj(SpinWeighted_SphericalHarmonic(iota, phi, l, -m))
            else:
                Ymstar = 0.
            
            hp = hp + Ampl[key]*np.exp(-1j*Phi[key])*(0.5*(Y + ((-1)**l)*Ymstar))
            hc = hc + Ampl[key]*np.exp(-1j*Phi[key])*(-1j* 0.5 * (Y - ((-1)**l)* Ymstar))
    
    return hp, hc

##############################################################################
# OTHERS
##############################################################################

def check_evparams(evParams, checktidal=False):
    """
    Check the format of the events parameters and make the needed conversions.
    
    :param dict(numpy.ndarray, numpy.ndarray, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    :param bool, optional checktidal: Boolean specifying if the tidal parameters have to be included in the checks.
    
    """
    # Function to check the format and limits of the events' parameters and make the needed conversions
    try:
        evParams['dL']
    except KeyError:
        raise ValueError('The luminosity distance has to be provided.')
        
    swapBool = np.full(evParams['dL'].shape, False)
    
    if np.any(evParams['dL']<=0.):
        raise ValueError('The luminosity distance cannot be null or negative.')
        
    try:
        evParams['eta']
    except KeyError:
        try:
            if np.any(evParams['m1']<=0.):
                raise ValueError('The mass of the first object cannot be null or negative.')
            if np.any(evParams['m2']<=0.):
                raise ValueError('The mass of the second object cannot be null or negative.')
            if np.any(evParams['m1']<evParams['m2']):
                swapBool = np.where(evParams['m1']<evParams['m2'], True, False)
                
                print('WARNING: for one or more events the mass of the first object is smaller than the mass of the second, we swap them and all the corresponding parameters')
            evParams['eta'] = evParams['m1']*evParams['m2']/((evParams['m1']+evParams['m2'])**2)
            evParams['Mc'] = ((evParams['m1']*evParams['m2'])**(3./5.))/((evParams['m1']+evParams['m2'])**(1./5.))
        except KeyError:
            raise ValueError('Two among Mc--eta and m1--m2 have to be provided.')
    
    if np.any(evParams['eta']<=0.):
        raise ValueError('The symmetric mass ratio cannot be null or negative.')
    if np.any(evParams['eta']>0.25):
        raise ValueError('The symmetric mass ratio cannot be greater than 1/4.')
    if np.any(evParams['Mc']<=0.):
        raise ValueError('The chirp mass cannot be null or negative.')
        
    try:
        evParams['chi1z']
    except KeyError:
        try:
            if np.any(abs(evParams['chiS'])>1.):
                raise ValueError('The symmetric spin component cannot have modulus greater than 1.')
            if np.any(abs(evParams['chiA'])>1.):
                raise ValueError('The antisymmetric spin component cannot have modulus greater than 1.')
            evParams['chi1z'] = evParams['chiS'] + evParams['chiA']
            evParams['chi2z'] = evParams['chiS'] - evParams['chiA']
        except KeyError:
            raise ValueError('Two among chi1z, chi2z and chiS and chiA have to be provided.')
    
    if np.any(abs(evParams['chi1z'])>1.):
        raise ValueError('The spin of the first object cannot have modulus greater than 1.')
    if np.any(abs(evParams['chi2z'])>1.):
        raise ValueError('The spin of the second object cannot have modulus greater than 1.')
    
    if np.any(swapBool):
        chi1tmp = evParams['chi1z']
        evParams['chi1z'] = np.where(swapBool, evParams['chi2z'], evParams['chi1z'])
        evParams['chi2z'] = np.where(swapBool, chi1tmp, evParams['chi2z'])
        print(evParams['chi1z'])
        print(evParams['chi2z'])
    
    if checktidal:
        try:
            evParams['Lambda1']
        except KeyError:
            try:
                evParams['Lambda1'], evParams['Lambda2'] = Lam12_from_Lamt_delLam(evParams['LambdaTilde'], evParams['deltaLambda'], evParams['eta'])
            except KeyError:
                raise ValueError('Two among Lambda1, Lambda2 and Lambda_Tilde and delta Lambda have to be provided.')
        
        if np.any(evParams['Lambda1']<0.):
            raise ValueError('The adimensional tidal deformability of the first object cannot be null or negative.')
        if np.any(evParams['Lambda2']<0.):
            raise ValueError('The adimensional tidal deformability of the second object cannot be null or negative.')
            
        if np.any(swapBool):
            Lam1tmp = evParams['Lambda1']
            evParams['Lambda1'] = np.where(swapBool, evParams['Lambda2'], evParams['Lambda1'])
            evParams['Lambda2'] = np.where(swapBool, Lam1tmp, evParams['Lambda2'])
