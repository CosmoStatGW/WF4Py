#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from abc import ABC, abstractmethod
import os
import sys
import h5py

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(SCRIPT_DIR)
WFfilesPATH = os.path.join(SCRIPT_DIR, 'WFfiles')

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

##############################################################################
# TAYLORF2 3.5 RESTRICTED PN WAVEFORM
##############################################################################

class TaylorF2_RestrictedPN(WaveFormModel):
    '''
    TaylorF2 restricted PN waveform model
    '''
    
    # This waveform model is restricted PN (the amplitude stays as in Newtonian approximation) up to 3.5 PN
    def __init__(self, fHigh=None, is_tidal=False, use_3p5PN_SpinHO=False, phiref_vlso=False, **kwargs):
        
        # Setting use_3p5PN_SpinHO=True SS and SSS contributions at 3.5PN are added (not present in LAL)
        # Setting is_tidal=True tidal contributions to the waveform at 10 and 12 PN are added
        # Setting phiref_vlso=True the LSO frequency is used as reference frequency
        
        if fHigh is None:
            fHigh = 1./(6.*np.pi*np.sqrt(6.)*utils.GMsun_over_c3) #Hz
        if is_tidal:
            objectT = 'BNS'
        else:
            objectT = 'BBH'
        self.use_3p5PN_SpinHO = use_3p5PN_SpinHO
        self.phiref_vlso = phiref_vlso
        super().__init__(objectT, fHigh, is_tidal=is_tidal, **kwargs)
    
    def Phi(self, f, **kwargs):
        # From A. Buonanno, B. Iyer, E. Ochsner, Y. Pan, B.S. Sathyaprakash - arXiv:0907.0700 - eq. (3.18) plus spins as in arXiv:1107.1267 eq. (5.3) up to 2.5PN and PhysRevD.93.084054 eq. (6) for 3PN and 3.5PN
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        
        Mtot_sec = kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        Seta = np.sqrt(1.0 - 4.0*eta)
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi_s, chi_a   = 0.5*(chi1 + chi2), 0.5*(chi1 - chi2)
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi_sdotchi_a  = chi_s*chi_a
        # flso = 1/6^(3/2)/(pi*M) -> vlso = (pi*M*flso)^(1/3) = (1/6^(3/2))^(1/3)
        vlso = 1./np.sqrt(6.)
        
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        TF2coeffs['four'] = 15293365./508032. + (27145.*eta)/504.+ (3085.*eta2)/72. + (-405./8. + 200.*eta)*chi_a2 - (405.*Seta*chi_sdotchi_a)/4. + (-405./8. + (5.*eta)/2.)*chi_s2
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        if self.phiref_vlso:
            TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*(1.-3.*np.log(vlso))
            phiR = 0.
        else:
            TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
            # This pi factor is needed to include LAL fRef rescaling, so to end up with the exact same waveform
            phiR = np.pi
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        TF2coeffs['six'] = 11583231236531./4694215680. - 640./3.*np.pi**2 - 6848./21.*np.euler_gamma + eta*(-15737765635./3048192. + 2255./12.*np.pi**2) + eta2*76055./1728. - eta2*eta*127825./1296. - (6848./21.)*np.log(4.) + np.pi*(2270.*Seta*chi_a/3. + (2270./3. - 520.*eta)*chi_s) + (75515./144. - 8225.*eta/18.)*Seta*chi_sdotchi_a + (75515./288. - 263245.*eta/252. - 480.*eta2)*chi_a2 + (75515./288. - 232415.*eta/504. + 1255.*eta2/9.)*chi_s2
        TF2coeffs['six_log'] = -(6848./21.)
        if self.use_3p5PN_SpinHO:
        # This part includes SS and SSS contributions at 3.5PN, which are not included in LAL
            TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36. + (14585./8. - 7270.*eta + 80.*eta2)*chi_a2)*chi_s + (14585./24. - 475.*eta/6. + 100.*eta2/3.)*chi_s2*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a + (14585./24. - 2380.*eta)*chi_a2*chi_a + (14585./8. - 215.*eta/2.)*chi_a*chi_s2)
        else:
            TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)

        if self.is_tidal:
            # Add tidal contribution if needed, as in PhysRevD.89.103012
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
            Lam_t, delLam    = utils.Lamt_delLam_from_Lam12(Lambda1, Lambda2, eta)
            
            phi_Tidal = (-0.5*39.*Lam_t)*(v**10.) + (-3115./64.*Lam_t + 6595./364.*Seta*delLam)*(v**12.)
            
            phase = TF2OverallAmpl*(TF2coeffs['zero'] + TF2coeffs['one']*v + TF2coeffs['two']*v*v + TF2coeffs['three']*v**3 + TF2coeffs['four']*v**4 + (TF2coeffs['five'] + TF2coeffs['five_log']*np.log(v))*v**5 + (TF2coeffs['six'] + TF2coeffs['six_log']*np.log(v))*v**6 + TF2coeffs['seven']*v**7 + phi_Tidal)/(v**5.)
            
        else:
            phase = TF2OverallAmpl*(TF2coeffs['zero'] + TF2coeffs['one']*v + TF2coeffs['two']*v*v + TF2coeffs['three']*v**3 + TF2coeffs['four']*v**4 + (TF2coeffs['five'] + TF2coeffs['five_log']*np.log(v))*v**5 + (TF2coeffs['six'] + TF2coeffs['six_log']*np.log(v))*v**6 + TF2coeffs['seven']*v**7)/(v**5.)
        
        return phase + phiR - np.pi*0.25

    def Ampl(self, f, **kwargs):
        # In the restricted PN approach the amplitude is the same as for the Newtonian approximation, so this function is equivalent
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        amplitude = np.sqrt(5./24.) * (np.pi**(-2./3.)) * utils.clightGpc/kwargs['dL'] * (utils.GMsun_over_c3*kwargs['Mc'])**(5./6.) * (f**(-7./6.))
        return amplitude
    
    def tau_star(self, f, **kwargs):
        # We use the expression in arXiv:0907.0700 eq. (3.8b)
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        Mtot_sec = kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (- 10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)

##############################################################################
# IMRPhenomD WAVEFORM
##############################################################################

class IMRPhenomD(WaveFormModel):
    '''
    IMRPhenomD waveform model
    '''
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253
    def __init__(self, **kwargs):
        # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        self.AMP_fJoin_INS = 0.014
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        self.PHI_fJoin_INS = 0.018
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.2
         
        super().__init__('BBH', fcutPar, **kwargs)
        
        self.QNMgrid_a     = np.loadtxt(os.path.join(WFfilesPATH, 'QNMData_a.txt'))
        self.QNMgrid_fring = np.loadtxt(os.path.join(WFfilesPATH, 'QNMData_fring.txt'))
        self.QNMgrid_fdamp = np.loadtxt(os.path.join(WFfilesPATH, 'QNMData_fdamp.txt'))
        
    def Phi(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        
        QuadMon1, QuadMon2 = np.ones(M.shape), np.ones(M.shape)
        
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi1dotchi2    = chi1*chi2
        chi_sdotchi_a  = chi_s*chi_a
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        
        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoin = self.PHI_fJoin_INS
        fMRDJoin = 0.5*fring
        
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoin)/3.))*((np.pi*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoin)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoin - beta3/(3.*fInsJoin*fInsJoin*fInsJoin) + beta2*np.log(fInsJoin)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoin
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoin - beta3/(3.*fMRDJoin*fMRDJoin*fMRDJoin) + beta2*np.log(fMRDJoin))/eta + C1Int + C2Int*fMRDJoin
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoin**4) + beta2/fMRDJoin)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * np.arctan((fMRDJoin - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoin
        
        # Time shift so that peak amplitude is approximately at t=0
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), np.fabs(fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2))
        
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        # LAL sets fRef as the minimum frequency, do the same
        fRef   = np.amin(fgrid, axis=0)
        phiRef = np.where(fRef < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fRef**(2./3.)) + PhiInspcoeffs['third']*(fRef**(1./3.)) + PhiInspcoeffs['third_log']*(fRef**(1./3.))*np.log(np.pi*fRef)/3. + PhiInspcoeffs['log']*np.log(np.pi*fRef)/3. + PhiInspcoeffs['min_third']*(fRef**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fRef**(-2./3.)) + PhiInspcoeffs['min_one']/fRef + PhiInspcoeffs['min_four_thirds']*(fRef**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fRef**(-5./3.)) + (PhiInspcoeffs['one']*fRef + PhiInspcoeffs['four_thirds']*(fRef**(4./3.)) + PhiInspcoeffs['five_thirds']*(fRef**(5./3.)) + PhiInspcoeffs['two']*fRef*fRef)/eta, np.where(fRef<fMRDJoin, (beta1*fRef - beta3/(3.*fRef*fRef*fRef) + beta2*np.log(fRef))/eta + C1Int + C2Int*fRef, np.where(fRef < self.fcutPar, (-(alpha2/fRef) + (4.0/3.0) * (alpha3 * (fRef**(3./4.))) + alpha1 * fRef + alpha4 * np.arctan((fRef - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fRef,0.)))

        phis = np.where(fgrid < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fgrid**(2./3.)) + PhiInspcoeffs['third']*(fgrid**(1./3.)) + PhiInspcoeffs['third_log']*(fgrid**(1./3.))*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['log']*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['min_third']*(fgrid**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fgrid**(-2./3.)) + PhiInspcoeffs['min_one']/fgrid + PhiInspcoeffs['min_four_thirds']*(fgrid**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fgrid**(-5./3.)) + (PhiInspcoeffs['one']*fgrid + PhiInspcoeffs['four_thirds']*(fgrid**(4./3.)) + PhiInspcoeffs['five_thirds']*(fgrid**(5./3.)) + PhiInspcoeffs['two']*fgrid*fgrid)/eta, np.where(fgrid<fMRDJoin, (beta1*fgrid - beta3/(3.*fgrid*fgrid*fgrid) + beta2*np.log(fgrid))/eta + C1Int + C2Int*fgrid, np.where(fgrid < self.fcutPar, (-(alpha2/fgrid) + (4.0/3.0) * (alpha3 * (fgrid**(3./4.))) + alpha1 * fgrid + alpha4 * np.arctan((fgrid - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fgrid,0.)))
        
        return phis + np.where(fgrid < self.fcutPar, - t0*(fgrid - fRef) - phiRef, 0.)
        
    def Ampl(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # This can speed up a bit, we call it multiple times
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        chi_s     = 0.5 * (chi1 + chi2)
        chi_a     = 0.5 * (chi1 - chi2)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = -1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
        gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
        # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
        rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
        rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
        rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
        # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
        f1Interm = self.AMP_fJoin_INS
        f3Interm = fpeak
        dfInterm = 0.5*(f3Interm - f1Interm)
        f2Interm = f1Interm + dfInterm
        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3
        # v1 is the inspiral model evaluated at f1Interm
        v1 = 1. + (f1Interm**(2./3.))*Acoeffs['two_thirds'] + (f1Interm**(4./3.)) * Acoeffs['four_thirds'] + (f1Interm**(5./3.)) *  Acoeffs['five_thirds'] + (f1Interm**(7./3.)) * Acoeffs['seven_thirds'] + (f1Interm**(8./3.)) * Acoeffs['eight_thirds'] + f1Interm * (Acoeffs['one'] + f1Interm * Acoeffs['two'] + f1Interm*f1Interm * Acoeffs['three'])
        # d1 is the derivative of the inspiral model evaluated at f1
        d1 = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/(1008.*(f1Interm**(1./3.))) + ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48. + ((-27312085. - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta + 35371056.*eta2)*(f1Interm**(1./3.))*(np.pi**(4./3.)))/6.096384e6 + (5.*(f1Interm**(2./3.))*(np.pi**(5./3.))*(chi2*(-285197.*(-1 + Seta)+ 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1- 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1 + 4*eta)*np.pi))/96768.- (f1Interm*np.pi*np.pi*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta+ 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi)+ 12.*eta*(-545384828789.0 - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta)- 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320*np.pi*np.pi)))/3.0042980352e10+ (7.0/3.0)*(f1Interm**(4./3.))*rho1 + (8.0/3.0)*(f1Interm**(5./3.))*rho2 + 3.*(f1Interm*f1Interm)*rho3
        # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
        v3 = np.exp(-(f3Interm - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3)
        # d2 is the derivative of the merger-ringdown model evaluated at f3
        d2 = ((-2.*fdamp*(f3Interm - fring)*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3) - (gamma2*gamma1))/(np.exp((f3Interm - fring)*gamma2/(fdamp*gamma3)) * ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3))
        # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
        v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
        # Now some definitions to speed up
        f1  = f1Interm
        f2  = f2Interm
        f3  = f3Interm
        f12 = f1Interm*f1Interm
        f13 = f1Interm*f12;
        f14 = f1Interm*f13;
        f15 = f1Interm*f14;
        f22 = f2Interm*f2Interm;
        f23 = f2Interm*f22;
        f24 = f2Interm*f23;
        f32 = f3Interm*f3Interm;
        f33 = f3Interm*f32;
        f34 = f3Interm*f33;
        f35 = f3Interm*f34;
        # Finally cnpute the deltas
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2.*d1*f13*f22*f33 - 2.*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2.*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4.*f12*f23*f32*v1 - 3.*f1*f24*f32*v1 - 8.*f12*f22*f33*v1 + 4.*f1*f23*f33*v1 + f24*f33*v1 + 4.*f12*f2*f34*v1 + f1*f22*f34*v1 - 2.*f23*f34*v1 - 2.*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3.*f14*f33*v2 - 3.*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2.*f14*f23*v3 - f13*f24*v3 + 2.*f15*f2*f3*v3 - f14*f22*f3*v3 - 4.*f13*f23*f3*v3 + 3.*f12*f24*f3*v3 - 4.*f14*f2*f32*v3 + 8.*f13*f22*f32*v3 - 4.*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3-f2)*(f3-f2)))
        delta1 = -((-(d2*f15*f22) + 2.*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2.*d1*f13*f23*f3 + 2.*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2.*d2*f12*f24 - d2*f15*f3 + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta3 = -((-2.*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 - 2.*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2.*d1*f12*f2*f3 + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        
        # Defined as in LALSimulation - LALSimIMRPhenomD.c line 332. Final units are correctly Hz^-1
        Overallamp = 2. * np.sqrt(5./(64.*np.pi)) * M * utils.GMsun_over_c2_Gpc * M * utils.GMsun_over_c3 / kwargs['dL']
        
        amplitudeIMR = np.where(fgrid < self.AMP_fJoin_INS, 1. + (fgrid**(2./3.))*Acoeffs['two_thirds'] + (fgrid**(4./3.)) * Acoeffs['four_thirds'] + (fgrid**(5./3.)) *  Acoeffs['five_thirds'] + (fgrid**(7./3.)) * Acoeffs['seven_thirds'] + (fgrid**(8./3.)) * Acoeffs['eight_thirds'] + fgrid * (Acoeffs['one'] + fgrid * Acoeffs['two'] + fgrid*fgrid * Acoeffs['three']), np.where(fgrid < fpeak, delta0 + fgrid*delta1 + fgrid*fgrid*(delta2 + fgrid*delta3 + fgrid*fgrid*delta4), np.where(fgrid < self.fcutPar,np.exp(-(fgrid - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((fgrid - fring)*(fgrid - fring) + fdamp*gamma3*fdamp*gamma3), 0.)))
        
        return Overallamp*amp0*(fgrid**(-7./6.))*amplitudeIMR
        
    def _finalspin(self, eta, chi1, chi2):
        # Compute the spin of the final object, as in LALSimIMRPhenomD_internals.c line 161 and 142,
        # which is taken from arXiv:1508.07250 eq. (3.6)
        Seta = np.sqrt(1.0 - 4.0*eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = (m1*m1 * chi1 + m2*m2 * chi2)
        
        af1 = eta*(3.4641016151377544 - 4.399247300629289*eta + 9.397292189321194*eta*eta - 13.180949901606242*eta*eta*eta)
        af2 = eta*(s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) + (0.1014665242971878 - 2.0967746996832157*eta)*s))
        af3 = eta*(s*((-1.3546806617824356 + 4.108962025369336*eta)*s*s + (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
        return af1 + af2 + af3
        
    def _radiatednrg(self, eta, chi1, chi2):
        # Predict the total radiated energy, from arXiv:1508.07250 eq (3.7) and (3.8)
        Seta = np.sqrt(1.0 - 4.0*eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def tau_star(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # For complex waveforms we use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        return self.fcutPar/(kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.)))
    
##############################################################################
# IMRPhenomD_NRTidalv2 WAVEFORM
##############################################################################

class IMRPhenomD_NRTidalv2(WaveFormModel):
    '''
    IMRPhenomD_NRTidal waveform model
    '''
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253, arXiv:1905.06011
    def __init__(self, **kwargs):
        # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        self.AMP_fJoin_INS = 0.014
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        self.PHI_fJoin_INS = 0.018
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.2
         
        super().__init__('BNS', fcutPar, is_tidal=True, **kwargs)
        
        self.QNMgrid_a     = np.loadtxt(os.path.join(WFfilesPATH, 'QNMData_a.txt'))
        self.QNMgrid_fring = np.loadtxt(os.path.join(WFfilesPATH, 'QNMData_fring.txt'))
        self.QNMgrid_fdamp = np.loadtxt(os.path.join(WFfilesPATH, 'QNMData_fdamp.txt'))
        
    def Phi(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        if 'Lambda1' in kwargs:
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
            # A non-zero tidal deformability induces a quadrupole moment (for BBH it is 1).
            # The relation between the two is given in arxiv:1608.02582 eq. (15) with coefficients from third row of Table I
            # We also extend the range to 0 <= Lam < 1, as done in LALSimulation in LALSimUniversalRelations.c line 123
            QuadMon1 = np.where(Lambda1 < 1., 1. + Lambda1*(0.427688866723244 + Lambda1*(-0.324336526985068 + Lambda1*0.1107439432180572)), np.exp(0.1940 + 0.09163 * np.log(Lambda1) + 0.04812 * np.log(Lambda1) * np.log(Lambda1) -4.283e-3 * np.log(Lambda1) * np.log(Lambda1) * np.log(Lambda1) + 1.245e-4 * np.log(Lambda1) * np.log(Lambda1) * np.log(Lambda1) * np.log(Lambda1)))
            QuadMon2 = np.where(Lambda2 < 1., 1. + Lambda2*(0.427688866723244 + Lambda2*(-0.324336526985068 + Lambda2*0.1107439432180572)), np.exp(0.1940 + 0.09163 * np.log(Lambda2) + 0.04812 * np.log(Lambda2) * np.log(Lambda2) -4.283e-3 * np.log(Lambda2) * np.log(Lambda2) * np.log(Lambda2) + 1.245e-4 * np.log(Lambda2) * np.log(Lambda2) * np.log(Lambda2) * np.log(Lambda2)))
            
        else:
            Lambda1, Lambda2   = np.zeros(M.shape), np.zeros(M.shape)
            QuadMon1, QuadMon2 = np.ones(M.shape), np.ones(M.shape)
            
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi_sdotchi_a  = chi_s*chi_a
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        
        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5 PN)
        # First the nonspinning part
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoin = self.PHI_fJoin_INS
        fMRDJoin = 0.5*fring
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoin)/3.))*((np.pi*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoin)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoin - beta3/(3.*fInsJoin*fInsJoin*fInsJoin) + beta2*np.log(fInsJoin)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoin
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoin - beta3/(3.*fMRDJoin*fMRDJoin*fMRDJoin) + beta2*np.log(fMRDJoin))/eta + C1Int + C2Int*fMRDJoin
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoin**4) + beta2/fMRDJoin)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * np.arctan((fMRDJoin - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoin
        
        # Time shift so that peak amplitude is approximately at t=0
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), np.fabs(fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2))
        
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        # LAL sets fRef as the minimum frequency, do the same
        fRef   = np.amin(fgrid, axis=0)
        phiRef = np.where(fRef < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fRef**(2./3.)) + PhiInspcoeffs['third']*(fRef**(1./3.)) + PhiInspcoeffs['third_log']*(fRef**(1./3.))*np.log(np.pi*fRef)/3. + PhiInspcoeffs['log']*np.log(np.pi*fRef)/3. + PhiInspcoeffs['min_third']*(fRef**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fRef**(-2./3.)) + PhiInspcoeffs['min_one']/fRef + PhiInspcoeffs['min_four_thirds']*(fRef**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fRef**(-5./3.)) + (PhiInspcoeffs['one']*fRef + PhiInspcoeffs['four_thirds']*(fRef**(4./3.)) + PhiInspcoeffs['five_thirds']*(fRef**(5./3.)) + PhiInspcoeffs['two']*fRef*fRef)/eta, np.where(fRef<fMRDJoin, (beta1*fRef - beta3/(3.*fRef*fRef*fRef) + beta2*np.log(fRef))/eta + C1Int + C2Int*fRef, np.where(fRef < self.fcutPar, (-(alpha2/fRef) + (4.0/3.0) * (alpha3 * (fRef**(3./4.))) + alpha1 * fRef + alpha4 * np.arctan((fRef - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fRef,0.)))
        
        phis = np.where(fgrid < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fgrid**(2./3.)) + PhiInspcoeffs['third']*(fgrid**(1./3.)) + PhiInspcoeffs['third_log']*(fgrid**(1./3.))*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['log']*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['min_third']*(fgrid**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fgrid**(-2./3.)) + PhiInspcoeffs['min_one']/fgrid + PhiInspcoeffs['min_four_thirds']*(fgrid**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fgrid**(-5./3.)) + (PhiInspcoeffs['one']*fgrid + PhiInspcoeffs['four_thirds']*(fgrid**(4./3.)) + PhiInspcoeffs['five_thirds']*(fgrid**(5./3.)) + PhiInspcoeffs['two']*fgrid*fgrid)/eta, np.where(fgrid<fMRDJoin, (beta1*fgrid - beta3/(3.*fgrid*fgrid*fgrid) + beta2*np.log(fgrid))/eta + C1Int + C2Int*fgrid, np.where(fgrid < self.fcutPar, (-(alpha2/fgrid) + (4.0/3.0) * (alpha3 * (fgrid**(3./4.))) + alpha1 * fgrid + alpha4 * np.arctan((fgrid - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fgrid,0.)))
        
        # Add the tidal contribution to the phase, as in arXiv:1905.06011
        # Compute the tidal coupling constant, arXiv:1905.06011 eq. (8) using Lambda = 2/3 k_2/C^5 (eq. (10))

        kappa2T = (3.0/13.0) * ((1.0 + 12.0*m2ByM/m1ByM)*(m1ByM**5)*Lambda1 + (1.0 + 12.0*m1ByM/m2ByM)*(m2ByM**5)*Lambda2)
        
        c_Newt   = 2.4375
        n_1      = -12.615214237993088
        n_3over2 =  19.0537346970349
        n_2      = -21.166863146081035
        n_5over2 =  90.55082156324926
        n_3      = -60.25357801943598
        d_1      = -15.11120782773667
        d_3over2 =  22.195327350624694
        d_2      =   8.064109635305156
    
        numTidal = 1.0 + (n_1 * ((np.pi*fgrid)**(2./3.))) + (n_3over2 * np.pi*fgrid) + (n_2 * ((np.pi*fgrid)**(4./3.))) + (n_5over2 * ((np.pi*fgrid)**(5./3.))) + (n_3 * np.pi*fgrid*np.pi*fgrid)
        denTidal = 1.0 + (d_1 * ((np.pi*fgrid)**(2./3.))) + (d_3over2 * np.pi*fgrid) + (d_2 * ((np.pi*fgrid)**(4./3.)))
        
        tidal_phase = - kappa2T * c_Newt / (m1ByM * m2ByM) * ((np.pi*fgrid)**(5./3.)) * numTidal / denTidal
        
        # In the NRTidalv2 extension also 3.5PN spin-squared and 3.5PN spin-cubed terms are included, see eq. (27) of arXiv:1905.06011
        # This is needed to account for spin-induced quadrupole moments
        # Compute octupole moment and emove -1 to account for BBH baseline
        
        OctMon1 = - 1. + np.exp(0.003131 + 2.071 * np.log(QuadMon1)  - 0.7152 * np.log(QuadMon1) * np.log(QuadMon1) + 0.2458 * np.log(QuadMon1) * np.log(QuadMon1) * np.log(QuadMon1) - 0.03309 * np.log(QuadMon1) * np.log(QuadMon1) * np.log(QuadMon1) * np.log(QuadMon1))
        OctMon2 = - 1. + np.exp(0.003131 + 2.071 * np.log(QuadMon2)  - 0.7152 * np.log(QuadMon2) * np.log(QuadMon2) + 0.2458 * np.log(QuadMon2) * np.log(QuadMon2) * np.log(QuadMon2) - 0.03309 * np.log(QuadMon2) * np.log(QuadMon2) * np.log(QuadMon2) * np.log(QuadMon2))

        SS_3p5PN  = - 400.*np.pi*(QuadMon1-1.)*chi12*m1ByM*m1ByM - 400.*np.pi*(QuadMon2-1.)*chi22*m2ByM*m2ByM
        SSS_3p5PN = 10.*((m1ByM*m1ByM+308./3.*m1ByM)*chi1+(m2ByM*m2ByM-89./3.*m2ByM)*chi2)*(QuadMon1-1.)*m1ByM*m1ByM*chi12 + 10.*((m2ByM*m2ByM+308./3.*m2ByM)*chi2+(m1ByM*m1ByM-89./3.*m1ByM)*chi1)*(QuadMon2-1.)*m2ByM*m2ByM*chi22 - 440.*OctMon1*m1ByM*m1ByM*m1ByM*chi12*chi1 - 440.*OctMon2*m2ByM*m2ByM*m2ByM*chi22*chi2
        
        return phis + np.where(fgrid < self.fcutPar, - t0*(fgrid - fRef) - phiRef + tidal_phase + (SS_3p5PN + SSS_3p5PN)*TF2OverallAmpl*((np.pi*fgrid)**(2./3.)), 0.)
        
    def Ampl(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # This can speed up a bit, we call it multiple times
        chi1, chi2   = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        if 'Lambda1' in kwargs:
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
        else:
            Lambda1, Lambda2 = np.zeros(M.shape), np.zeros(M.shape)
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = -1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
        gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
        # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
        rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
        rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
        rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
        # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
        f1Interm = self.AMP_fJoin_INS
        f3Interm = fpeak
        dfInterm = 0.5*(f3Interm - f1Interm)
        f2Interm = f1Interm + dfInterm
        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3
        # v1 is the inspiral model evaluated at f1Interm
        v1 = 1. + (f1Interm**(2./3.))*Acoeffs['two_thirds'] + (f1Interm**(4./3.)) * Acoeffs['four_thirds'] + (f1Interm**(5./3.)) *  Acoeffs['five_thirds'] + (f1Interm**(7./3.)) * Acoeffs['seven_thirds'] + (f1Interm**(8./3.)) * Acoeffs['eight_thirds'] + f1Interm * (Acoeffs['one'] + f1Interm * Acoeffs['two'] + f1Interm*f1Interm * Acoeffs['three'])
        # d1 is the derivative of the inspiral model evaluated at f1
        d1 = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/(1008.*(f1Interm**(1./3.))) + ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48. + ((-27312085. - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta + 35371056.*eta2)*(f1Interm**(1./3.))*(np.pi**(4./3.)))/6.096384e6 + (5.*(f1Interm**(2./3.))*(np.pi**(5./3.))*(chi2*(-285197.*(-1 + Seta)+ 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1- 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1 + 4*eta)*np.pi))/96768.- (f1Interm*np.pi*np.pi*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta+ 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi)+ 12.*eta*(-545384828789.0 - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta)- 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320*np.pi*np.pi)))/3.0042980352e10+ (7.0/3.0)*(f1Interm**(4./3.))*rho1 + (8.0/3.0)*(f1Interm**(5./3.))*rho2 + 3.*(f1Interm*f1Interm)*rho3
        # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
        v3 = np.exp(-(f3Interm - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3)
        # d2 is the derivative of the merger-ringdown model evaluated at f3
        d2 = ((-2.*fdamp*(f3Interm - fring)*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3) - (gamma2*gamma1))/(np.exp((f3Interm - fring)*gamma2/(fdamp*gamma3)) * ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3))
        # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
        v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
        # Now some definitions to speed up
        f1  = f1Interm
        f2  = f2Interm
        f3  = f3Interm
        f12 = f1Interm*f1Interm
        f13 = f1Interm*f12;
        f14 = f1Interm*f13;
        f15 = f1Interm*f14;
        f22 = f2Interm*f2Interm;
        f23 = f2Interm*f22;
        f24 = f2Interm*f23;
        f32 = f3Interm*f3Interm;
        f33 = f3Interm*f32;
        f34 = f3Interm*f33;
        f35 = f3Interm*f34;
        # Finally cnpute the deltas
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2.*d1*f13*f22*f33 - 2.*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2.*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4.*f12*f23*f32*v1 - 3.*f1*f24*f32*v1 - 8.*f12*f22*f33*v1 + 4.*f1*f23*f33*v1 + f24*f33*v1 + 4.*f12*f2*f34*v1 + f1*f22*f34*v1 - 2.*f23*f34*v1 - 2.*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3.*f14*f33*v2 - 3.*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2.*f14*f23*v3 - f13*f24*v3 + 2.*f15*f2*f3*v3 - f14*f22*f3*v3 - 4.*f13*f23*f3*v3 + 3.*f12*f24*f3*v3 - 4.*f14*f2*f32*v3 + 8.*f13*f22*f32*v3 - 4.*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3-f2)*(f3-f2)))
        delta1 = -((-(d2*f15*f22) + 2.*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2.*d1*f13*f23*f3 + 2.*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2.*d2*f12*f24 - d2*f15*f3 + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta3 = -((-2.*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 - 2.*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2.*d1*f12*f2*f3 + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        
        # Defined as in LALSimulation - LALSimIMRPhenomD.c line 332. Final units are correctly Hz^-1
        Overallamp = 2. * np.sqrt(5./(64.*np.pi)) * M * utils.GMsun_over_c2_Gpc * M * utils.GMsun_over_c3 / kwargs['dL']
        
        amplitudeIMR = np.where(fgrid < self.AMP_fJoin_INS, 1. + (fgrid**(2./3.))*Acoeffs['two_thirds'] + (fgrid**(4./3.)) * Acoeffs['four_thirds'] + (fgrid**(5./3.)) *  Acoeffs['five_thirds'] + (fgrid**(7./3.)) * Acoeffs['seven_thirds'] + (fgrid**(8./3.)) * Acoeffs['eight_thirds'] + fgrid * (Acoeffs['one'] + fgrid * Acoeffs['two'] + fgrid*fgrid * Acoeffs['three']), np.where(fgrid < fpeak, delta0 + fgrid*delta1 + fgrid*fgrid*(delta2 + fgrid*delta3 + fgrid*fgrid*delta4), np.where(fgrid < self.fcutPar,np.exp(-(fgrid - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((fgrid - fring)*(fgrid - fring) + fdamp*gamma3*fdamp*gamma3), 0.)))
        
        # Now add the tidal amplitude as in arXiv:1905.06011
        # Compute the tidal coupling constant, arXiv:1905.06011 eq. (8) using Lambda = 2/3 k_2/C^5 (eq. (10))
        Xa = 0.5 * (1.0 + Seta)
        Xb = 0.5 * (1.0 - Seta)
        kappa2T = (3.0/13.0) * ((1.0 + 12.0*Xb/Xa)*(Xa**5)*Lambda1 + (1.0 + 12.0*Xa/Xb)*(Xb**5)*Lambda2)
        
        # Now compute the amplitude modification as in arXiv:1905.06011 eq. (24)
        xTidal = (np.pi * fgrid)**(2./3.)
        n1T    = 4.157407407407407
        n289T  = 2519.111111111111
        dTidal = 13477.8073677
        polyTidal = (1.0 + n1T*xTidal + n289T*(xTidal**(2.89)))/(1.+dTidal*(xTidal**4))
        ampTidal = -9.0*kappa2T*(xTidal**3.25)*polyTidal
        
        # Compute the dimensionless merger frequency (Mf) for the Planck taper filtering
        a_0 = 0.3586
        n_1 = 3.35411203e-2
        n_2 = 4.31460284e-5
        d_1 = 7.54224145e-2
        d_2 = 2.23626859e-4
        kappa2T2 = kappa2T*kappa2T
        
        numPT = 1.0 + n_1*kappa2T + n_2*kappa2T2
        denPT = 1.0 + d_1*kappa2T + d_2*kappa2T2
        Q_0 = a_0 / np.sqrt(q)
        f_merger = Q_0 * (numPT / denPT) / (2.*np.pi)
        
        def planck_taper_fun(x, y):
            # Terminate the waveform at 1.2 times the merger frequency
            a=1.2
            yp = a*y
            planck_taper = np.where(x < y, 1., np.where(x > yp, 0., 1. - 1./(np.exp((yp - y)/(x - y) + (yp - y)/(x - yp)) + 1.)))
            return planck_taper
        # Now compute tha Planck taper series
        planck_taper = planck_taper_fun(fgrid, f_merger)
        
        return Overallamp*(amp0*(fgrid**(-7./6.))*amplitudeIMR + 2*np.sqrt(np.pi/5.)*ampTidal)*planck_taper
    
        
    def _finalspin(self, eta, chi1, chi2):
        # Compute the spin of the final object, as in LALSimIMRPhenomD_internals.c line 161 and 142,
        # which is taken from arXiv:1508.07250 eq. (3.6)
        Seta = np.sqrt(1.0 - 4.0*eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2)
        af1 = eta*(3.4641016151377544 - 4.399247300629289*eta + 9.397292189321194*eta*eta - 13.180949901606242*eta*eta*eta)
        af2 = eta*(s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) + (0.1014665242971878 - 2.0967746996832157*eta)*s))
        af3 = eta*(s*((-1.3546806617824356 + 4.108962025369336*eta)*s*s + (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
        return af1 + af2 + af3
        
    def _radiatednrg(self, eta, chi1, chi2):
        # Predict the total radiated energy, from arXiv:1508.07250 eq (3.7) and (3.8)
        Seta = np.sqrt(1.0 - 4.0*eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def tau_star(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # For complex waveforms we use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        Seta = np.sqrt(1.0 - 4.0*eta)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        if 'Lambda1' in kwargs:
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
        else:
            Lambda1, Lambda2 = np.zeros(M.shape), np.zeros(M.shape)
        Xa = 0.5 * (1.0 + Seta)
        Xb = 0.5 * (1.0 - Seta)
        kappa2T = (3.0/13.0) * ((1.0 + 12.0*Xb/Xa)*(Xa**5)*Lambda1 + (1.0 + 12.0*Xa/Xb)*(Xb**5)*Lambda2)
        
        # Compute the dimensionless merger frequency (Mf) for the Planck taper filtering
        a_0 = 0.3586
        n_1 = 3.35411203e-2
        n_2 = 4.31460284e-5
        d_1 = 7.54224145e-2
        d_2 = 2.23626859e-4
        kappa2T2 = kappa2T*kappa2T
        
        numPT = 1.0 + n_1*kappa2T + n_2*kappa2T2
        denPT = 1.0 + d_1*kappa2T + d_2*kappa2T2
        Q_0 = a_0 / np.sqrt(q)
        f_merger = Q_0 * (numPT / denPT) / (2.*np.pi)
        # Terminate the waveform at 1.2 times the merger frequency
        f_end_taper = 1.2*f_merger

        return f_end_taper/(M*utils.GMsun_over_c3)

##############################################################################
# IMRPhenomHM WAVEFORM
##############################################################################

class IMRPhenomHM(WaveFormModel):
    '''
    IMRPhenomHM waveform model
    '''
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253, arXiv:1708.00404, arXiv:1909.10010
    def __init__(self, **kwargs):
        # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        self.AMP_fJoin_INS = 0.014
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        self.PHI_fJoin_INS = 0.018
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.2
         
        super().__init__('BBH', fcutPar, is_HigherModes=True, **kwargs)
        
        # List of phase shifts: the index is the azimuthal number m
        self.complShiftm = np.array([0., np.pi*0.5, 0., -np.pi*0.5, np.pi, np.pi*0.5, 0.])
        
    def Phi(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        
        QuadMon1, QuadMon2 = np.ones(M.shape), np.ones(M.shape)
        
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi1dotchi2    = chi1*chi2
        chi_sdotchi_a  = chi_s*chi_a
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        # This is MfRef, needed to recover LAL, which sets fRef to f_min if fRef=0
        fRef  = np.amin(fgrid, axis=0)
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin, radiated energy and mass
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        finMass = 1. - Erad
        
        fring, fdamp = self._RDfreqCalc(finMass, aeff, 2, 2)
        
        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoin = self.PHI_fJoin_INS
        fMRDJoin = 0.5*fring
        
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoin)/3.))*((np.pi*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoin)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoin - beta3/(3.*fInsJoin*fInsJoin*fInsJoin) + beta2*np.log(fInsJoin)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoin
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoin - beta3/(3.*fMRDJoin*fMRDJoin*fMRDJoin) + beta2*np.log(fMRDJoin))/eta + C1Int + C2Int*fMRDJoin
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoin**4) + beta2/fMRDJoin)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * np.arctan((fMRDJoin - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoin
        
        # Time shift so that peak amplitude is approximately at t=0
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), np.fabs(fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2))
        
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        def completePhase(infreqs, C1MRDuse, C2MRDuse, RhoUse, TauUse):
            
            return np.where(infreqs < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(infreqs**(2./3.)) + PhiInspcoeffs['third']*(infreqs**(1./3.)) + PhiInspcoeffs['third_log']*(infreqs**(1./3.))*np.log(np.pi*infreqs)/3. + PhiInspcoeffs['log']*np.log(np.pi*infreqs)/3. + PhiInspcoeffs['min_third']*(infreqs**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(infreqs**(-2./3.)) + PhiInspcoeffs['min_one']/infreqs + PhiInspcoeffs['min_four_thirds']*(infreqs**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(infreqs**(-5./3.)) + (PhiInspcoeffs['one']*infreqs + PhiInspcoeffs['four_thirds']*(infreqs**(4./3.)) + PhiInspcoeffs['five_thirds']*(infreqs**(5./3.)) + PhiInspcoeffs['two']*infreqs*infreqs)/eta, np.where(infreqs<fMRDJoin, (beta1*infreqs - beta3/(3.*infreqs*infreqs*infreqs) + beta2*np.log(infreqs))/eta + C1Int + C2Int*infreqs, np.where(infreqs < self.fcutPar, (-(alpha2/infreqs) + (4.0/3.0) * (alpha3 * (infreqs**(3./4.))) + alpha1 * infreqs + alpha4 * RhoUse * np.arctan((infreqs - alpha5 * fring)/(fdamp * RhoUse * TauUse)))/eta + C1MRDuse + C2MRDuse*infreqs,0.)))
 
        phiRef = completePhase(fRef, C1MRD, C2MRD, 1., 1.)
        phi0   = 0.5*phiRef
        
        # Now compute the other modes, they are 5, we loop
        phislmp = {}
        
        for ell in (2,3,4):
            for mm in (ell-1, ell):
                # Compute ringdown and damping frequencies from fits for the various modes
                fringlm, fdamplm = self._RDfreqCalc(finMass, aeff, ell, mm)
                Rholm, Taulm = fring/fringlm, fdamplm/fdamp
                
                # Rholm and Taulm only figure in the MRD part, the rest of the coefficients is the same, recompute only this
                DPhiMRDVal    = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*Taulm*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*Taulm*Rholm*fdamp*Taulm*Rholm))))/eta
                PhiMRJoinTemp = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * Rholm* np.arctan((fMRDJoin - alpha5 * fring)/(fdamp*Rholm*Taulm))
                C2MRDHM = DPhiIntTempVal - DPhiMRDVal
                C1MRDHM = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRDHM*fMRDJoin
                    
                # Compute mapping coefficinets
                Map_fl = self.PHI_fJoin_INS
                Map_fi = Map_fl / Rholm
                Map_fr = fringlm
                
                Map_ai, Map_bi = 2./mm, 0.
                    
                Map_Trd = Map_fr * Rholm
                Map_Ti  = 2. * Map_fi / mm
                Map_am  = (Map_Trd - Map_Ti) / (Map_fr - Map_fi)
                Map_bm  = Map_Ti - Map_fi * Map_am
                    
                Map_ar, Map_br = Rholm, 0.
                # Now compute the needed constants
                tmpMf = Map_am * Map_fi + Map_bm
                PhDBconst = completePhase(tmpMf, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_am
                    
                tmpMf = Map_ar * Map_fr + Map_br
                PhDCconst = completePhase(tmpMf, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_ar
            
                tmpMf = Map_ai * Map_fi + Map_bi
                PhDBAterm = completePhase(tmpMf, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_ai
                
                tmpMf = Map_am * Map_fr + Map_bm
                tmpphaseC = - PhDBconst + PhDBAterm + completePhase(tmpMf, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_am
                
                phitmp = np.where(fgrid < Map_fi, completePhase(fgrid*Map_ai + Map_bi, C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_ai, np.where(fgrid < Map_fr, - PhDBconst + PhDBAterm + completePhase(fgrid*Map_am + Map_bm, C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_am, - PhDCconst + tmpphaseC + completePhase(fgrid*Map_ar + Map_br, C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_ar))
                
                phislmp[str(ell)+str(mm)] =  phitmp - t0*(fgrid - fRef) - mm*phi0 + self.complShiftm[mm]
                
        return phislmp
        
    
    def Ampl(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # This can speed up a bit, we call it multiple times
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = -1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        finMass = 1. - Erad
        
        fring, fdamp = self._RDfreqCalc(finMass, aeff, 2, 2)
        
        # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
        gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
        # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
        rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
        rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
        rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
        # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
        f1Interm = self.AMP_fJoin_INS
        f3Interm = fpeak
        dfInterm = 0.5*(f3Interm - f1Interm)
        f2Interm = f1Interm + dfInterm
        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3
        # v1 is the inspiral model evaluated at f1Interm
        v1 = 1. + (f1Interm**(2./3.))*Acoeffs['two_thirds'] + (f1Interm**(4./3.)) * Acoeffs['four_thirds'] + (f1Interm**(5./3.)) *  Acoeffs['five_thirds'] + (f1Interm**(7./3.)) * Acoeffs['seven_thirds'] + (f1Interm**(8./3.)) * Acoeffs['eight_thirds'] + f1Interm * (Acoeffs['one'] + f1Interm * Acoeffs['two'] + f1Interm*f1Interm * Acoeffs['three'])
        # d1 is the derivative of the inspiral model evaluated at f1
        d1 = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/(1008.*(f1Interm**(1./3.))) + ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48. + ((-27312085. - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta + 35371056.*eta2)*(f1Interm**(1./3.))*(np.pi**(4./3.)))/6.096384e6 + (5.*(f1Interm**(2./3.))*(np.pi**(5./3.))*(chi2*(-285197.*(-1 + Seta)+ 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1- 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1 + 4*eta)*np.pi))/96768.- (f1Interm*np.pi*np.pi*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta+ 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi)+ 12.*eta*(-545384828789.0 - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta)- 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320*np.pi*np.pi)))/3.0042980352e10+ (7.0/3.0)*(f1Interm**(4./3.))*rho1 + (8.0/3.0)*(f1Interm**(5./3.))*rho2 + 3.*(f1Interm*f1Interm)*rho3
        # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
        v3 = np.exp(-(f3Interm - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3)
        # d2 is the derivative of the merger-ringdown model evaluated at f3
        d2 = ((-2.*fdamp*(f3Interm - fring)*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3) - (gamma2*gamma1))/(np.exp((f3Interm - fring)*gamma2/(fdamp*gamma3)) * ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3))
        # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
        v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
        # Now some definitions to speed up
        f1  = f1Interm
        f2  = f2Interm
        f3  = f3Interm
        f12 = f1Interm*f1Interm
        f13 = f1Interm*f12;
        f14 = f1Interm*f13;
        f15 = f1Interm*f14;
        f22 = f2Interm*f2Interm;
        f23 = f2Interm*f22;
        f24 = f2Interm*f23;
        f32 = f3Interm*f3Interm;
        f33 = f3Interm*f32;
        f34 = f3Interm*f33;
        f35 = f3Interm*f34;
        # Finally cnpute the deltas
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2.*d1*f13*f22*f33 - 2.*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2.*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4.*f12*f23*f32*v1 - 3.*f1*f24*f32*v1 - 8.*f12*f22*f33*v1 + 4.*f1*f23*f33*v1 + f24*f33*v1 + 4.*f12*f2*f34*v1 + f1*f22*f34*v1 - 2.*f23*f34*v1 - 2.*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3.*f14*f33*v2 - 3.*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2.*f14*f23*v3 - f13*f24*v3 + 2.*f15*f2*f3*v3 - f14*f22*f3*v3 - 4.*f13*f23*f3*v3 + 3.*f12*f24*f3*v3 - 4.*f14*f2*f32*v3 + 8.*f13*f22*f32*v3 - 4.*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3-f2)*(f3-f2)))
        delta1 = -((-(d2*f15*f22) + 2.*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2.*d1*f13*f23*f3 + 2.*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2.*d2*f12*f24 - d2*f15*f3 + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta3 = -((-2.*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 - 2.*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2.*d1*f12*f2*f3 + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        
        # Defined as in LALSimulation - LALSimIMRPhenomUtils.c line 70. Final units are correctly Hz^-1
        # there is a 2 * sqrt(5/(64*pi)) missing w.r.t the standard coefficient, which comes from the (2,2) shperical harmonic
        Overallamp = M * utils.GMsun_over_c2_Gpc * M * utils.GMsun_over_c3 / kwargs['dL']
        
        def completeAmpl(infreqs):
            
            return Overallamp*amp0*(infreqs**(-7./6.))*np.where(infreqs < self.AMP_fJoin_INS, 1. + (infreqs**(2./3.))*Acoeffs['two_thirds'] + (infreqs**(4./3.)) * Acoeffs['four_thirds'] + (infreqs**(5./3.)) *  Acoeffs['five_thirds'] + (infreqs**(7./3.)) * Acoeffs['seven_thirds'] + (infreqs**(8./3.)) * Acoeffs['eight_thirds'] + infreqs * (Acoeffs['one'] + infreqs * Acoeffs['two'] + infreqs*infreqs * Acoeffs['three']), np.where(infreqs < fpeak, delta0 + infreqs*delta1 + infreqs*infreqs*(delta2 + infreqs*delta3 + infreqs*infreqs*delta4), np.where(infreqs < self.fcutPar,np.exp(-(infreqs - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((infreqs - fring)*(infreqs - fring) + fdamp*gamma3*fdamp*gamma3), 0.)))
        
        def OnePointFiveSpinPN(infreqs, l, m, ChiS, ChiA):
            # PN amplitudes function, needed to scale
            
            v  = (2.*np.pi*infreqs/m)**(1./3.)
            v2 = v*v
            v3 = v2*v
            
            if (2 == l) and (2 == m):
                Hlm = 1.
            
            elif (2 == l) and (1 == m):
                Hlm = (np.sqrt(2.0) / 3.0) * (v * Seta - v2 * 1.5 * (ChiA + Seta * ChiS) + v3 * Seta * ((335.0 / 672.0) + (eta * 117.0 / 56.0)) + v3*v * (ChiA * (3427.0 / 1344. - eta * 2101.0 / 336.) + Seta * ChiS * (3427.0 / 1344 - eta * 965 / 336) + Seta * (-1j * 0.5 - np.pi - 2 * 1j * 0.69314718056)))
            
            elif (3 == l) and (3 == m):
                Hlm = 0.75 * np.sqrt(5.0 / 7.0) * (v * Seta)
            
            elif (3 == l) and (2 == m):
                Hlm = (1.0 / 3.0) * np.sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta))
            
            elif (4 == l) and (4 == m):
                Hlm = (4.0 / 9.0) * np.sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta)
            
            elif (4 == l) and (3 == m):
                Hlm = 0.75 * np.sqrt(3.0 / 35.0) * v3 * Seta * (1.0 - 2.0 * eta)
            
            else:
                raise ValueError('Mode not present in IMRPhenomHM waveform model.')
            
            # Compute the final PN Amplitude at Leading Order in Mf
            
            return np.pi * np.sqrt(eta * 2. / 3.) * (v**(-3.5)) * abs(Hlm)
            
        ampllm = {}
        
        for ell in (2,3,4):
            for mm in (ell-1, ell):
                # Compute ringdown and damping frequencies from fits for the various modes
                fringlm, fdamplm = self._RDfreqCalc(finMass, aeff, ell, mm)
                Rholm, Taulm = fring/fringlm, fdamplm/fdamp
                
                # Scale input frequencies according to PhenomHM model
                # Compute mapping coefficinets
                Map_fl = self.AMP_fJoin_INS
                Map_fi = Map_fl / Rholm
                Map_fr = fringlm
                
                Map_ai, Map_bi = 2./mm, 0.
                    
                Map_Trd = Map_fr - fringlm + fring
                Map_Ti  = 2. * Map_fi / mm
                Map_am  = (Map_Trd - Map_Ti) / (Map_fr - Map_fi)
                Map_bm  = Map_Ti - Map_fi * Map_am
                    
                Map_ar, Map_br = 1., - Map_fr + fring
                
                # Now scale as f -> f*a+b for each regime
                fgridScaled = np.where(fgrid < Map_fi, fgrid*Map_ai + Map_bi, np.where(fgrid < Map_fr, fgrid*Map_am + Map_bm, fgrid*Map_ar + Map_br))
                
                # Map the ampliude's range
                # We divide by the leading order l=m=2 behavior, and then scale in the expected PN behavior for the multipole of interest.
                
                beta_term1  = OnePointFiveSpinPN(fgrid, ell, mm, chi_s, chi_a)
                beta_term2  = OnePointFiveSpinPN(2.*fgrid/mm, ell, mm, chi_s, chi_a)
                
                HMamp_term1 = OnePointFiveSpinPN(fgridScaled, ell, mm, chi_s, chi_a)
                HMamp_term2 = OnePointFiveSpinPN(fgridScaled, 2, 2, 0., 0.)
                
                # The (3,3) and (4,3) modes vanish if eta=0.25 (equal mass case) and the (2,1) mode vanishes if both eta=0.25 and chi1z=chi2z
                # This results in NaNs having 0/0, correct for this using np.nan_to_num()
                
                ampllm[str(ell)+str(mm)] = np.nan_to_num(completeAmpl(fgridScaled) * (beta_term1 / beta_term2) * HMamp_term1 / HMamp_term2)
                
        return ampllm
    
    def hphc(self, f, **kwargs):
        # This function retuns directly the full plus and cross polarisations, avoiding for loops over the modes
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        iota = kwargs['iota']
        QuadMon1, QuadMon2 = np.ones(M.shape), np.ones(M.shape)
        
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi1dotchi2    = chi1*chi2
        chi_sdotchi_a  = chi_s*chi_a
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        # This is MfRef, needed to recover LAL, which sets fRef to f_min if fRef=0
        fRef  = np.amin(fgrid, axis=0)
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin, radiated energy and mass
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        finMass = 1. - Erad
    
        # Compute the real and imag parts of the complex ringdown frequency for the (l,m) mode as in LALSimIMRPhenomHM.c line 189
        # These are all fits of the different modes. We directly exploit the fact that the relevant HM in this WF are 6
        modes = np.array([21,22,32,33,43,44])
        ells = np.floor(modes/10).astype('int')
        mms = modes - ells*10
        # Domain mapping for dimnesionless BH spin
        alphaRDfr = np.log(2. - aeff) / np.log(3.)
        # beta = 1. / (2. + l - abs(m))
        betaRDfr = np.where(modes==21, 1./3., np.where(modes==22, 0.5, np.where(modes==32, 1./3., np.where(modes==33, 0.5, np.where(modes==43, 1./3., 0.5)))))
        kappaRDfr  = np.expand_dims(alphaRDfr,len(alphaRDfr.shape))**betaRDfr
        kappaRDfr2 = kappaRDfr*kappaRDfr
        kappaRDfr3 = kappaRDfr*kappaRDfr2
        kappaRDfr4 = kappaRDfr*kappaRDfr3
        
        tmpRDfr = np.where(modes==21, 0.589113 * np.exp(0.043525 * 1j) + 0.18896353 * np.exp(2.289868 * 1j) * kappaRDfr + 1.15012965 * np.exp(5.810057 * 1j) * kappaRDfr2 + 6.04585476 * np.exp(2.741967 * 1j) * kappaRDfr3 + 11.12627777 * np.exp(5.844130 * 1j) * kappaRDfr4 + 9.34711461 * np.exp(2.669372 * 1j) * kappaRDfr4*kappaRDfr + 3.03838318 * np.exp(5.791518 * 1j) * kappaRDfr4*kappaRDfr2, np.where(modes==22, 1.0 + kappaRDfr * (1.557847 * np.exp(2.903124 * 1j) + 1.95097051 * np.exp(5.920970 * 1j) * kappaRDfr + 2.09971716 * np.exp(2.760585 * 1j) * kappaRDfr2 + 1.41094660 * np.exp(5.914340 * 1j) * kappaRDfr3 + 0.41063923 * np.exp(2.795235 * 1j) * kappaRDfr4), np.where(modes==32, 1.022464 * np.exp(0.004870 * 1j) + 0.24731213 * np.exp(0.665292 * 1j) * kappaRDfr + 1.70468239 * np.exp(3.138283 * 1j) * kappaRDfr2 + 0.94604882 * np.exp(0.163247 * 1j) * kappaRDfr3 + 1.53189884 * np.exp(5.703573 * 1j) * kappaRDfr4 + 2.28052668 * np.exp(2.685231 * 1j) * kappaRDfr4*kappaRDfr + 0.92150314 * np.exp(5.841704 * 1j) * kappaRDfr4*kappaRDfr2, np.where(modes==33, 1.5 + kappaRDfr * (2.095657 * np.exp(2.964973 * 1j) + 2.46964352 * np.exp(5.996734 * 1j) * kappaRDfr + 2.66552551 * np.exp(2.817591 * 1j) * kappaRDfr2 + 1.75836443 * np.exp(5.932693 * 1j) * kappaRDfr3 + 0.49905688 * np.exp(2.781658 * 1j) * kappaRDfr4), np.where(modes==43, 1.5 + kappaRDfr * (0.205046 * np.exp(0.595328 * 1j) + 3.10333396 * np.exp(3.016200 * 1j) * kappaRDfr + 4.23612166 * np.exp(6.038842 * 1j) * kappaRDfr2 + 3.02890198 * np.exp(2.826239 * 1j) * kappaRDfr3 + 0.90843949 * np.exp(5.915164 * 1j) * kappaRDfr4), 2.0 + kappaRDfr * (2.658908 * np.exp(3.002787 * 1j) + 2.97825567 * np.exp(6.050955 * 1j) * kappaRDfr + 3.21842350 * np.exp(2.877514 * 1j) * kappaRDfr2 + 2.12764967 * np.exp(5.989669 * 1j) * kappaRDfr3 + 0.60338186 * np.exp(2.830031 * 1j) * kappaRDfr4))))))

        fringlm = (np.real(tmpRDfr)/(2.*np.pi*np.expand_dims(finMass, len(finMass.shape))))
        fdamplm = (np.imag(tmpRDfr)/(2.*np.pi*np.expand_dims(finMass, len(finMass.shape))))
        
        # This recomputation is included for clarity
        betaRDfr = 0.5
        kappaRDfr  = alphaRDfr**betaRDfr
        kappaRDfr2 = kappaRDfr*kappaRDfr
        kappaRDfr3 = kappaRDfr*kappaRDfr2
        kappaRDfr4 = kappaRDfr*kappaRDfr3
        
        tmpRDfr = 1.0 + kappaRDfr * (1.557847 * np.exp(2.903124 * 1j) + 1.95097051 * np.exp(5.920970 * 1j) * kappaRDfr + 2.09971716 * np.exp(2.760585 * 1j) * kappaRDfr2 + 1.41094660 * np.exp(5.914340 * 1j) * kappaRDfr3 + 0.41063923 * np.exp(2.795235 * 1j) * kappaRDfr4)
        
        fring = (np.real(tmpRDfr)/(2.*np.pi*finMass))
        fdamp = (np.imag(tmpRDfr)/(2.*np.pi*finMass))

        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoinPh = self.PHI_fJoin_INS
        fMRDJoinPh = 0.5*fring
        
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoinPh)/3.))*((np.pi*fInsJoinPh)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoinPh) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoinPh)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoinPh)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoinPh**(1./3.)) + sigma3*(fInsJoinPh**(2./3.)) + sigma4*fInsJoinPh)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoinPh**4) + beta2/fInsJoinPh)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoinPh**(2./3.)) + PhiInspcoeffs['third']*(fInsJoinPh**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoinPh**(1./3.))*np.log(np.pi*fInsJoinPh)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoinPh)/3. + PhiInspcoeffs['min_third']*(fInsJoinPh**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoinPh**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoinPh + PhiInspcoeffs['min_four_thirds']*(fInsJoinPh**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoinPh**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoinPh + PhiInspcoeffs['four_thirds']*(fInsJoinPh**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoinPh**(5./3.)) + PhiInspcoeffs['two']*fInsJoinPh*fInsJoinPh)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoinPh - beta3/(3.*fInsJoinPh*fInsJoinPh*fInsJoinPh) + beta2*np.log(fInsJoinPh)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoinPh
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoinPh - beta3/(3.*fMRDJoinPh*fMRDJoinPh*fMRDJoinPh) + beta2*np.log(fMRDJoinPh))/eta + C1Int + C2Int*fMRDJoinPh
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoinPh**4) + beta2/fMRDJoinPh)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoinPh*fMRDJoinPh) + alpha3/(fMRDJoinPh**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoinPh - alpha5*fring)*(fMRDJoinPh - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoinPh) + (4.0/3.0) * (alpha3 * (fMRDJoinPh**(3./4.))) + alpha1 * fMRDJoinPh + alpha4 * np.arctan((fMRDJoinPh - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoinPh
        
        # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
        gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
        # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
        rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
        rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
        rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
        # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
        f1Interm = self.AMP_fJoin_INS
        f3Interm = fpeak
        dfInterm = 0.5*(f3Interm - f1Interm)
        f2Interm = f1Interm + dfInterm
        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3
        # v1 is the inspiral model evaluated at f1Interm
        v1 = 1. + (f1Interm**(2./3.))*Acoeffs['two_thirds'] + (f1Interm**(4./3.)) * Acoeffs['four_thirds'] + (f1Interm**(5./3.)) *  Acoeffs['five_thirds'] + (f1Interm**(7./3.)) * Acoeffs['seven_thirds'] + (f1Interm**(8./3.)) * Acoeffs['eight_thirds'] + f1Interm * (Acoeffs['one'] + f1Interm * Acoeffs['two'] + f1Interm*f1Interm * Acoeffs['three'])
        # d1 is the derivative of the inspiral model evaluated at f1
        d1 = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/(1008.*(f1Interm**(1./3.))) + ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48. + ((-27312085. - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta + 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta + 35371056.*eta2)*(f1Interm**(1./3.))*(np.pi**(4./3.)))/6.096384e6 + (5.*(f1Interm**(2./3.))*(np.pi**(5./3.))*(chi2*(-285197.*(-1 + Seta)+ 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1- 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1 + 4*eta)*np.pi))/96768.- (f1Interm*np.pi*np.pi*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta+ 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi)+ 12.*eta*(-545384828789.0 - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta)- 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320*np.pi*np.pi)))/3.0042980352e10+ (7.0/3.0)*(f1Interm**(4./3.))*rho1 + (8.0/3.0)*(f1Interm**(5./3.))*rho2 + 3.*(f1Interm*f1Interm)*rho3
        # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
        v3 = np.exp(-(f3Interm - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3)
        # d2 is the derivative of the merger-ringdown model evaluated at f3
        d2 = ((-2.*fdamp*(f3Interm - fring)*gamma3*gamma1) / ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3) - (gamma2*gamma1))/(np.exp((f3Interm - fring)*gamma2/(fdamp*gamma3)) * ((f3Interm - fring)*(f3Interm - fring) + fdamp*gamma3*fdamp*gamma3))
        # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
        v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
        # Now some definitions to speed up
        f1  = f1Interm
        f2  = f2Interm
        f3  = f3Interm
        f12 = f1Interm*f1Interm
        f13 = f1Interm*f12;
        f14 = f1Interm*f13;
        f15 = f1Interm*f14;
        f22 = f2Interm*f2Interm;
        f23 = f2Interm*f22;
        f24 = f2Interm*f23;
        f32 = f3Interm*f3Interm;
        f33 = f3Interm*f32;
        f34 = f3Interm*f33;
        f35 = f3Interm*f34;
        # Finally cnpute the deltas
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2.*d1*f13*f22*f33 - 2.*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2.*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4.*f12*f23*f32*v1 - 3.*f1*f24*f32*v1 - 8.*f12*f22*f33*v1 + 4.*f1*f23*f33*v1 + f24*f33*v1 + 4.*f12*f2*f34*v1 + f1*f22*f34*v1 - 2.*f23*f34*v1 - 2.*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3.*f14*f33*v2 - 3.*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2.*f14*f23*v3 - f13*f24*v3 + 2.*f15*f2*f3*v3 - f14*f22*f3*v3 - 4.*f13*f23*f3*v3 + 3.*f12*f24*f3*v3 - 4.*f14*f2*f32*v3 + 8.*f13*f22*f32*v3 - 4.*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta0 = -((d2*f15*f22*f3 - 2.*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3-f2)*(f3-f2)))
        delta1 = -((-(d2*f15*f22) + 2.*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2.*d1*f13*f23*f3 + 2.*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2.*d2*f12*f24 - d2*f15*f3 + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta3 = -((-2.*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 - 2.*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2.*d1*f12*f2*f3 + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) / ((f1 - f2)*(f1 - f2)*(f1 - f3)*(f1 - f3)*(f1 - f3)*(f3 - f2)*(f3 - f2)))
        
        # Defined as in LALSimulation - LALSimIMRPhenomUtils.c line 70. Final units are correctly Hz^-1
        # there is a 2 * sqrt(5/(64*pi)) missing w.r.t the standard coefficient, which comes from the (2,2) shperical harmonic
        Overallamp = M * utils.GMsun_over_c2_Gpc * M * utils.GMsun_over_c3 / kwargs['dL']
        
        def completeAmpl(infreqs):
            
            return Overallamp*amp0*(infreqs**(-7./6.))*np.where(infreqs < self.AMP_fJoin_INS, 1. + (infreqs**(2./3.))*Acoeffs['two_thirds'] + (infreqs**(4./3.)) * Acoeffs['four_thirds'] + (infreqs**(5./3.)) *  Acoeffs['five_thirds'] + (infreqs**(7./3.)) * Acoeffs['seven_thirds'] + (infreqs**(8./3.)) * Acoeffs['eight_thirds'] + infreqs * (Acoeffs['one'] + infreqs * Acoeffs['two'] + infreqs*infreqs * Acoeffs['three']), np.where(infreqs < fpeak, delta0 + infreqs*delta1 + infreqs*infreqs*(delta2 + infreqs*delta3 + infreqs*infreqs*delta4), np.where(infreqs < self.fcutPar,np.exp(-(infreqs - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((infreqs - fring)*(infreqs - fring) + fdamp*gamma3*fdamp*gamma3), 0.)))
        
        def completePhase(infreqs, C1MRDuse, C2MRDuse, RhoUse, TauUse):
            
            return np.where(infreqs < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(infreqs**(2./3.)) + PhiInspcoeffs['third']*(infreqs**(1./3.)) + PhiInspcoeffs['third_log']*(infreqs**(1./3.))*np.log(np.pi*infreqs)/3. + PhiInspcoeffs['log']*np.log(np.pi*infreqs)/3. + PhiInspcoeffs['min_third']*(infreqs**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(infreqs**(-2./3.)) + PhiInspcoeffs['min_one']/infreqs + PhiInspcoeffs['min_four_thirds']*(infreqs**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(infreqs**(-5./3.)) + (PhiInspcoeffs['one']*infreqs + PhiInspcoeffs['four_thirds']*(infreqs**(4./3.)) + PhiInspcoeffs['five_thirds']*(infreqs**(5./3.)) + PhiInspcoeffs['two']*infreqs*infreqs)/eta, np.where(infreqs<fMRDJoinPh, (beta1*infreqs - beta3/(3.*infreqs*infreqs*infreqs) + beta2*np.log(infreqs))/eta + C1Int + C2Int*infreqs, np.where(infreqs < self.fcutPar, (-(alpha2/infreqs) + (4.0/3.0) * (alpha3 * (infreqs**(3./4.))) + alpha1 * infreqs + alpha4 * RhoUse * np.arctan((infreqs - alpha5 * fring)/(fdamp * RhoUse * TauUse)))/eta + C1MRDuse + C2MRDuse*infreqs,0.)))
 
        def OnePointFiveSpinPN(infreqs, ChiS, ChiA):
            # PN amplitudes function, needed to scale
            
            v  = np.moveaxis((2.*np.pi*infreqs/mms)**(1./3.), len(infreqs.shape)-1, len(infreqs.shape) - 2)
            v2 = v*v
            v3 = v2*v
            
            reshModes = np.expand_dims(modes, len(modes.shape))
            Hlm = np.where(reshModes==21, (np.sqrt(2.0) / 3.0) * (v * Seta - v2 * 1.5 * (ChiA + Seta * ChiS) + v3 * Seta * ((335.0 / 672.0) + (eta * 117.0 / 56.0)) + v3*v * (ChiA * (3427.0 / 1344. - eta * 2101.0 / 336.) + Seta * ChiS * (3427.0 / 1344 - eta * 965 / 336) + Seta * (-1j * 0.5 - np.pi - 2 * 1j * 0.69314718056))), np.where(reshModes==22, 1., np.where(reshModes==32, (1.0 / 3.0) * np.sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta)), np.where(reshModes==33, 0.75 * np.sqrt(5.0 / 7.0) * (v * Seta), np.where(reshModes==43, 0.75 * np.sqrt(3.0 / 35.0) * v3 * Seta * (1.0 - 2.0 * eta), (4.0 / 9.0) * np.sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta))))))
            
            # Compute the final PN Amplitude at Leading Order in Mf
            
            return np.pi * np.sqrt(eta * 2. / 3.) * (v**(-3.5)) * abs(Hlm)
        
        def SpinWeighted_SphericalHarmonic(theta, phi=0.):
            # Taken from arXiv:0709.0093v3 eq. (II.7), (II.8) and LALSimulation for the s=-2 case and up to l=4.
            # We assume already phi=0 and s=-2 to simplify the function
            
            Ylm    = np.where(modes==21, np.sqrt( 5.0 / ( 16.0 * np.pi ) ) * np.sin( theta )*( 1.0 + np.cos( theta )), np.where(modes==22, np.sqrt( 5.0 / ( 64.0 * np.pi ) ) * ( 1.0 + np.cos( theta ))*( 1.0 + np.cos( theta )), np.where(modes==32, np.sqrt(7.0/np.pi)*((np.cos(theta*0.5))**(4.0))*(-2.0 + 3.0*np.cos(theta))*0.5, np.where(modes==33, -np.sqrt(21.0/(2.0*np.pi))*((np.cos(theta/2.0))**(5.0))*np.sin(theta*0.5), np.where(modes==43, -3.0*np.sqrt(7.0/(2.0*np.pi))*((np.cos(theta*0.5))**5.0)*(-1.0 + 2.0*np.cos(theta))*np.sin(theta*0.5), 3.0*np.sqrt(7.0/np.pi)*((np.cos(theta*0.5))**6.0)*(np.sin(theta*0.5)*np.sin(theta*0.5)))))))
            Ylminm = np.where(modes==21, np.sqrt( 5.0 / ( 16.0 * np.pi ) ) * np.sin( theta )*( 1.0 - np.cos( theta )), np.where(modes==22, np.sqrt( 5.0 / ( 64.0 * np.pi ) ) * ( 1.0 - np.cos( theta ))*( 1.0 - np.cos( theta )), np.where(modes==32, np.sqrt(7.0/(4.0*np.pi))*(2.0 + 3.0*np.cos(theta))*((np.sin(theta*0.5))**(4.0)), np.where(modes==33, np.sqrt(21.0/(2.0*np.pi))*np.cos(theta*0.5)*((np.sin(theta*0.5))**(5.)), np.where(modes==43, 3.0*np.sqrt(7.0/(2.0*np.pi))*np.cos(theta*0.5)*(1.0 + 2.0*np.cos(theta))*((np.sin(theta*0.5))**5.0), 3.0*np.sqrt(7.0/np.pi)*(np.cos(theta*0.5)*np.cos(theta*0.5))*((np.sin(theta*0.5))**6.0))))))
            
            return Ylm, Ylminm
        
        # Time shift so that peak amplitude is approximately at t=0
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        phiRef = completePhase(fRef, C1MRD, C2MRD, 1., 1.)
        phi0   = 0.5*phiRef
        
        # Now compute all the modes, they are 6, we parallelize
        
        Rholm, Taulm = (fring/fringlm.T), (fdamplm.T/fdamp)
        # Rholm and Taulm only figure in the MRD part, the rest of the coefficients is the same, recompute only this
        DPhiMRDVal    = (alpha1 + alpha2/(fMRDJoinPh*fMRDJoinPh) + alpha3/(fMRDJoinPh**(1./4.)) + alpha4/(fdamp*Taulm*(1. + (fMRDJoinPh - alpha5*fring)*(fMRDJoinPh - alpha5*fring)/(fdamp*Taulm*Rholm*fdamp*Taulm*Rholm))))/eta
        PhiMRJoinTemp = -(alpha2/fMRDJoinPh) + (4.0/3.0) * (alpha3 * (fMRDJoinPh**(3./4.))) + alpha1 * fMRDJoinPh + alpha4 * Rholm* np.arctan((fMRDJoinPh - alpha5 * fring)/(fdamp*Rholm*Taulm))
        C2MRDHM = DPhiIntTempVal - DPhiMRDVal
        C1MRDHM = (PhiIntTempVal - PhiMRJoinTemp/eta - C2MRDHM*fMRDJoinPh).T
        Rholm, Taulm, DPhiMRDVal, PhiMRJoinTemp, C2MRDHM = Rholm.T, Taulm.T, DPhiMRDVal.T, PhiMRJoinTemp.T, C2MRDHM.T
        
        # Scale input frequencies according to PhenomHM model
        # Compute mapping coefficinets
        Map_flPhi = self.PHI_fJoin_INS
        Map_fiPhi = Map_flPhi / Rholm
        Map_flAmp = self.AMP_fJoin_INS
        Map_fiAmp = Map_flAmp / Rholm
        Map_fr = fringlm
        
        Map_ai, Map_bi = 2./mms, 0.

        Map_TrdAmp = Map_fr - fringlm + np.expand_dims(fring, len(fring.shape))
        Map_TiAmp  = 2. * Map_fiAmp / mms
        Map_amAmp  = (Map_TrdAmp - Map_TiAmp) / (Map_fr - Map_fiAmp)
        Map_bmAmp  = Map_TiAmp - Map_fiAmp * Map_amAmp

        Map_TrdPhi = Map_fr * Rholm
        Map_TiPhi  = 2. * Map_fiPhi / mms
        Map_amPhi  = (Map_TrdPhi - Map_TiPhi) / (Map_fr - Map_fiPhi)
        Map_bmPhi  = Map_TiPhi - Map_fiPhi * Map_amPhi

        Map_arAmp, Map_brAmp = 1., - Map_fr + np.expand_dims(fring, len(fring.shape))
        Map_arPhi, Map_brPhi = Rholm, 0.
        
        # Now scale as f -> f*a+b for each regime
        fgrid = np.expand_dims(fgrid, len(fgrid.shape))# Need a new axis to do all the 6 calculations together

        fgridScaled = np.where(fgrid < Map_fiAmp, fgrid*Map_ai + Map_bi, np.where(fgrid < Map_fr, fgrid*Map_amAmp + Map_bmAmp, fgrid*Map_arAmp + Map_brAmp))
        # Map the ampliude's range
        # We divide by the leading order l=m=2 behavior, and then scale in the expected PN behavior for the multipole of interest.
              
        beta_term1  = OnePointFiveSpinPN(fgrid, chi_s, chi_a)
        beta_term2  = OnePointFiveSpinPN(2.*fgrid/mms, chi_s, chi_a)
        HMamp_term1 = OnePointFiveSpinPN(fgridScaled, chi_s, chi_a)
        fgridScaled = np.moveaxis(fgridScaled, len(fgridScaled.shape)-1, len(fgridScaled.shape) - 2)
        HMamp_term2 = np.pi * np.sqrt(eta * 2. / 3.) * ((np.pi*fgridScaled)**(-7./6.))
        
        # The (3,3) and (4,3) modes vanish if eta=0.25 (equal mass case) and the (2,1) mode vanishes if both eta=0.25 and chi1z=chi2z
        # This results in NaNs having 0/0, correct for this using np.nan_to_num()
                
        AmplsAllModes = np.nan_to_num(completeAmpl(fgridScaled) * (beta_term1 / beta_term2) * HMamp_term1 / HMamp_term2)
        AmplsAllModes = np.moveaxis(AmplsAllModes, len(AmplsAllModes.shape)-1, len(AmplsAllModes.shape) - 2)
        C1MRDHM, C2MRDHM, Rholm, Taulm = C1MRDHM.T, C2MRDHM.T, Rholm.T, Taulm.T
        
        tmpMf = Map_amPhi * Map_fiPhi + Map_bmPhi
        PhDBconst = (completePhase(tmpMf.T, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_amPhi.T)
                    
        tmpMf = Map_arPhi * Map_fr + Map_brPhi
        PhDCconst = (completePhase(tmpMf.T, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_arPhi.T)
            
        tmpMf = Map_ai * Map_fiPhi + Map_bi
        PhDBAterm = (completePhase(tmpMf.T, C1MRDHM, C2MRDHM, Rholm, Taulm).T / Map_ai).T
        
        tmpMf = Map_amPhi * Map_fr + Map_bmPhi
        tmpphaseC = (- PhDBconst + PhDBAterm + completePhase(tmpMf.T, C1MRDHM, C2MRDHM, Rholm, Taulm) / Map_amPhi.T)
        
        tmpGridShape = len((fgrid*Map_ai + Map_bi).shape)
                                
        if len(AmplsAllModes.shape)==3:
            PhisAllModes = np.where(fgrid < Map_fiPhi, np.moveaxis(completePhase(np.moveaxis((fgrid*Map_ai + Map_bi), tmpGridShape-1, tmpGridShape-2), C1MRDHM, C2MRDHM, Rholm, Taulm), len(AmplsAllModes.shape)-1, len(AmplsAllModes.shape)-2)/Map_ai, np.where(fgrid < Map_fr, np.moveaxis(- PhDBconst + PhDBAterm + completePhase(np.moveaxis((fgrid*Map_amPhi + Map_bmPhi), tmpGridShape-1, tmpGridShape-2), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_amPhi.T, len(AmplsAllModes.shape)-1, len(AmplsAllModes.shape)-2), np.moveaxis(- PhDCconst + tmpphaseC + completePhase(np.moveaxis((fgrid*Map_arPhi + Map_brPhi), tmpGridShape-1, tmpGridShape-2), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_arPhi.T, len(AmplsAllModes.shape)-1, len(AmplsAllModes.shape)-2)))
        else:
            C1MRDHM, C2MRDHM, Rholm, Taulm = C1MRDHM.T, C2MRDHM.T, Rholm.T, Taulm.T
            PhDBconst, PhDCconst, PhDBAterm, tmpphaseC = PhDBconst.T, PhDCconst.T, PhDBAterm.T, tmpphaseC.T
            PhisAllModes = np.where(fgrid < Map_fiPhi, completePhase((fgrid*Map_ai + Map_bi), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_ai, np.where(fgrid < Map_fr, - PhDBconst + PhDBAterm + completePhase((fgrid*Map_amPhi + Map_bmPhi), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_amPhi, - PhDCconst + tmpphaseC + completePhase((fgrid*Map_arPhi + Map_brPhi), C1MRDHM, C2MRDHM, Rholm, Taulm)/Map_arPhi))
            
        PhisAllModes = PhisAllModes - np.expand_dims(t0, len(t0.shape))*(fgrid - np.expand_dims(fRef, len(fRef.shape))) - mms*np.expand_dims(phi0, len(phi0.shape)) + self.complShiftm[mms]
        modes = np.expand_dims(modes, len(modes.shape))
        Y, Ymstar = SpinWeighted_SphericalHarmonic(iota)
        Y, Ymstar = Y.T, np.conj(Ymstar).T
        
        hp = np.sum(AmplsAllModes*np.exp(-1j*PhisAllModes)*(0.5*(Y + ((-1)**ells)*Ymstar)), axis=-1)
        hc = np.sum(AmplsAllModes*np.exp(-1j*PhisAllModes)*(-1j* 0.5 * (Y - ((-1)**ells)* Ymstar)), axis=-1)
        
        return hp, hc

        
    def _finalspin(self, eta, chi1, chi2):
        # Compute the spin of the final object, as in LALSimIMRPhenomD_internals.c line 161 and 142,
        # which is taken from arXiv:1508.07250 eq. (3.6)
        Seta = np.sqrt(1.0 - 4.0*eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2)
        af1 = eta*(3.4641016151377544 - 4.399247300629289*eta + 9.397292189321194*eta*eta - 13.180949901606242*eta*eta*eta)
        af2 = eta*(s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) + (0.1014665242971878 - 2.0967746996832157*eta)*s))
        af3 = eta*(s*((-1.3546806617824356 + 4.108962025369336*eta)*s*s + (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
        return af1 + af2 + af3
        
    def _radiatednrg(self, eta, chi1, chi2):
        # Predict the total radiated energy, from arXiv:1508.07250 eq (3.7) and (3.8)
        Seta = np.sqrt(1.0 - 4.0*eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def _RDfreqCalc(self, finalmass, finalspin, l, m):
        # Compute the real and imag parts of the complex ringdown frequency for the (l,m) mode as in LALSimIMRPhenomHM.c line 189
        # These are all fits of the different modes
        
        # Domain mapping for dimnesionless BH spin
        alpha = np.log(2. - finalspin) / np.log(3.);
        beta = 1. / (2. + l - abs(m));
        kappa  = alpha**beta
        kappa2 = kappa*kappa
        kappa3 = kappa*kappa2
        kappa4 = kappa*kappa3
        
        if (2 == l) and (2 == m):
            res = 1.0 + kappa * (1.557847 * np.exp(2.903124 * 1j) + 1.95097051 * np.exp(5.920970 * 1j) * kappa + 2.09971716 * np.exp(2.760585 * 1j) * kappa2 + 1.41094660 * np.exp(5.914340 * 1j) * kappa3 + 0.41063923 * np.exp(2.795235 * 1j) * kappa4)
        
        elif (3 == l) and (2 == m):
            res = 1.022464 * np.exp(0.004870 * 1j) + 0.24731213 * np.exp(0.665292 * 1j) * kappa + 1.70468239 * np.exp(3.138283 * 1j) * kappa2 + 0.94604882 * np.exp(0.163247 * 1j) * kappa3 + 1.53189884 * np.exp(5.703573 * 1j) * kappa4 + 2.28052668 * np.exp(2.685231 * 1j) * kappa4*kappa + 0.92150314 * np.exp(5.841704 * 1j) * kappa4*kappa2
        
        elif (4 == l) and (4 == m):
            res = 2.0 + kappa * (2.658908 * np.exp(3.002787 * 1j) + 2.97825567 * np.exp(6.050955 * 1j) * kappa + 3.21842350 * np.exp(2.877514 * 1j) * kappa2 + 2.12764967 * np.exp(5.989669 * 1j) * kappa3 + 0.60338186 * np.exp(2.830031 * 1j) * kappa4)
        
        elif (2 == l) and (1 == m):
            res = 0.589113 * np.exp(0.043525 * 1j) + 0.18896353 * np.exp(2.289868 * 1j) * kappa + 1.15012965 * np.exp(5.810057 * 1j) * kappa2 + 6.04585476 * np.exp(2.741967 * 1j) * kappa3 + 11.12627777 * np.exp(5.844130 * 1j) * kappa4 + 9.34711461 * np.exp(2.669372 * 1j) * kappa4*kappa + 3.03838318 * np.exp(5.791518 * 1j) * kappa4*kappa2
        
        elif (3 == l) and (3 == m):
            res = 1.5 + kappa * (2.095657 * np.exp(2.964973 * 1j) + 2.46964352 * np.exp(5.996734 * 1j) * kappa + 2.66552551 * np.exp(2.817591 * 1j) * kappa2 + 1.75836443 * np.exp(5.932693 * 1j) * kappa3 + 0.49905688 * np.exp(2.781658 * 1j) * kappa4)
        
        elif (4 == l) and (3 == m):
            res = 1.5 + kappa * (0.205046 * np.exp(0.595328 * 1j) + 3.10333396 * np.exp(3.016200 * 1j) * kappa + 4.23612166 * np.exp(6.038842 * 1j) * kappa2 + 3.02890198 * np.exp(2.826239 * 1j) * kappa3 + 0.90843949 * np.exp(5.915164 * 1j) * kappa4)
        
        else:
            raise ValueError('Mode not present in IMRPhenomHM waveform model.')
        
        if m < 0:
            res = -np.conj(res)
        
        fring = np.real(res)/(2.*np.pi*finalmass)
        
        fdamp = np.imag(res)/(2.*np.pi*finalmass)
        
        return fring, fdamp
        
    def tau_star(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # For complex waveforms we use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta  = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        return self.fcutPar/(kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.)))

##############################################################################
# IMRPhenomNSBH WAVEFORM
##############################################################################

class IMRPhenomNSBH(WaveFormModel):
    '''
    IMRPhenomNSBH waveform model
    The inputs labelled as 1 refer to the BH (e.g. chi1z) and with 2 to the NS (e.g. Lambda2)
    
    NOTE: In LAL, to compute the parameter xi_tide in arXiv:1509.00512 eq. (8), the roots are extracted.
          In python this would break the possibility to vectorise so, to circumvent the issue, we compute
          a grid of xi_tide as a function of the compactness, mass ratio and BH spin, and then use a 3D
          interpolator. The first time the code runs, if this interpolator is not already present, it will be
          computed (the base resolution of the grid is 200 pts per parameter, that we find
          sufficient to reproduce LAL waveforms with good precision, given the smooth behaviour of the function,
          but this can be raised if needed. In this case, it is necessary to change tha name of the file assigned to self.path_xiTide_tab and the res parameter passed to _make_xiTide_interpolator())
    '''
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253, arXiv:1509.00512, arXiv:1905.06011
    def __init__(self, verbose=True, returnMergerType=False, **kwargs):
        # returnMergerType can be used to output the type of merger, determined in the amplitude calculation
        
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        self.PHI_fJoin_INS = 0.018
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.2
        self.verbose=verbose
        self.returnMergerType = returnMergerType
        super().__init__('NSBH', fcutPar, is_tidal=True, **kwargs)
        
        self.QNMgrid_a       = np.loadtxt(os.path.join(WFfilesPATH, 'QNMData_a.txt'))
        self.QNMgrid_fring   = np.loadtxt(os.path.join(WFfilesPATH, 'QNMData_fring.txt'))
        self.QNMgrid_fdamp   = np.loadtxt(os.path.join(WFfilesPATH, 'QNMData_fdamp.txt'))
        self.path_xiTide_tab = os.path.join(WFfilesPATH, 'xiTide_Table_200.h5')
        
        self._make_xiTide_interpolator(res=200)
        
    def Phi(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        
        if np.any(chi2 != 0.):
            print('WARNING: IMRPhenomNSBH is tuned only for chi_NS = 0')
        
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2 = chi1*chi2
        
        Lambda1, Lambda = kwargs['Lambda1'], kwargs['Lambda2']
        
        if np.any(Lambda1 != 0.):
            print('WARNING: BH tidal deformability cannot be different from 0, discarding it')

        del Lambda1
        # A non-zero tidal deformability induces a quadrupole moment (for BBH it is 1).
        # Taken from arXiv:1303.1528 eq. (54) and Tab. I
        QuadMon1, QuadMon2 = 1., np.where(Lambda<1e-5, 1., np.exp(0.194 + 0.0936*np.log(Lambda) + 0.0474*np.log(Lambda)*np.log(Lambda) - 0.00421*np.log(Lambda)*np.log(Lambda)*np.log(Lambda) + 0.000123*np.log(Lambda)*np.log(Lambda)*np.log(Lambda)*np.log(Lambda)))
        
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi1dotchi2 = chi1*chi2
        chi_sdotchi_a = chi_s*chi_a
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin and radiated energy for IMRPhenomNSBH, the rest is equivalent to IMRPhenomD_NRTidalv2
        # Get remnant spin for assumed aligned spin system, from arXiv:1903.11622 Table I and eq. (4), (5) and (6)
        
        p1_remSp = ((-5.44187381e-03*chi1 + 7.91165608e-03) + (2.33362046e-02*chi1 + 2.47764497e-02)*eta)*eta
        p2_remSp = ((-8.56844797e-07*chi1 - 2.81727682e-06) + (6.61290966e-06*chi1 + 4.28979016e-05)*eta)*eta
        p3_remSp = ((-3.04174272e-02*chi1 + 2.54889050e-01) + (1.47549350e-01*chi1 - 4.27905832e-01)*eta)*eta
        
        modelRemSp = (1. + Lambda * p1_remSp + Lambda*Lambda * p2_remSp) / ((1. + Lambda*p3_remSp*p3_remSp)*(1. + Lambda*p3_remSp*p3_remSp))

        modelRemSp = np.where((chi1 < 0.) & (eta < 0.188), 1., modelRemSp)
        modelRemSp = np.where(chi1 < -0.5, 1., modelRemSp)
        modelRemSp = np.where(modelRemSp > 1., 1., modelRemSp)
        
        del p1_remSp, p2_remSp, p3_remSp
        
        # Work with spin variables weighted on square of the BH mass over total mass
        S1BH = chi1 * m1ByM * m1ByM
        Shat = S1BH / (m1ByM*m1ByM + m2ByM*m2ByM) # this would be = (chi1*m1*m1 + chi2*m2*m2)/(m1*m1 + m2*m2), but chi2=0 by assumption
        
        # Compute fit to L_orb in arXiv:1611.00332 eq. (16)
        Lorb = (2.*np.sqrt(3.)*eta + 5.24*3.8326341618708577*eta2 + 1.3*(-9.487364155598392)*eta*eta2)/(1. + 2.88*2.5134875145648374*eta) + ((-0.194)*1.0009563702914628*Shat*(4.409160174224525*eta + 0.5118334706832706*eta2 + (64. - 16.*4.409160174224525 - 4.*0.5118334706832706)*eta2*eta) + 0.0851*0.7877509372255369*Shat*Shat*(8.77367320110712*eta + (-32.060648277652994)*eta2 + (64. - 16.*8.77367320110712 - 4.*(-32.060648277652994))*eta2*eta) + 0.00954*0.6540138407185817*Shat*Shat*Shat*(22.830033250479833*eta + (-153.83722669033995)*eta2 + (64. - 16.*22.830033250479833 - 4.*(-153.83722669033995))*eta2*eta))/(1. + (-0.579)*0.8396665722805308*Shat*(1.8804718791591157 + (-4.770246856212403)*eta + 0.*eta2 + (64. - 64.*1.8804718791591157 - 16.*(-4.770246856212403) - 4.*0.)*eta2*eta)) + 0.3223660562764661*Seta*eta2*(1. + 9.332575956437443*eta)*chi1 + 2.3170397514509933*Shat*Seta*eta2*eta*(1. + (-3.2624649875884852)*eta)*chi1 + (-0.059808322561702126)*eta2*eta*chi12;
        
        chif = (Lorb + S1BH)*modelRemSp

        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(chif, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(chif, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        
        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoin = self.PHI_fJoin_INS
        fMRDJoin = 0.5*fring
        
        # First the Inspiral - Intermediate: we compute C1Int and C2Int coeffs
        # Equations to solve for to get C(1) continuous join
        # PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
        # Joining at fInsJoin
        # PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
        # PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
        # This is the first derivative wrt f of the inspiral phase computed at fInsJoin, first add the PN contribution and then the higher order calibrated terms
        DPhiIns = (2.0*TF2coeffs['seven']*TF2OverallAmpl*((np.pi*fInsJoin)**(7./3.)) + (TF2coeffs['six']*TF2OverallAmpl + TF2coeffs['six_log']*TF2OverallAmpl * (1.0 + np.log(np.pi*fInsJoin)/3.))*((np.pi*fInsJoin)**(2.)) + TF2coeffs['five_log']*TF2OverallAmpl*((np.pi*fInsJoin)**(5./3.)) - TF2coeffs['four']*TF2OverallAmpl*((np.pi*fInsJoin)**(4./3.)) - 2.*TF2coeffs['three']*TF2OverallAmpl*(np.pi*fInsJoin) - 3.*TF2coeffs['two']*TF2OverallAmpl*((np.pi*fInsJoin)**(2./3.)) - 4.*TF2coeffs['one']*TF2OverallAmpl*((np.pi*fInsJoin)**(1./3.)) - 5.*TF2coeffs['zero']*TF2OverallAmpl)*np.pi/(3.*((np.pi*fInsJoin)**(8./3.)))
        DPhiIns = DPhiIns + (sigma1 + sigma2*(fInsJoin**(1./3.)) + sigma3*(fInsJoin**(2./3.)) + sigma4*fInsJoin)/eta
        # This is the first derivative of the Intermediate phase computed at fInsJoin
        DPhiInt = (beta1 + beta3/(fInsJoin**4) + beta2/fInsJoin)/eta
        
        C2Int = DPhiIns - DPhiInt
        
        # This is the inspiral phase computed at fInsJoin
        PhiInsJoin = PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fInsJoin**(2./3.)) + PhiInspcoeffs['third']*(fInsJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fInsJoin**(1./3.))*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['log']*np.log(np.pi*fInsJoin)/3. + PhiInspcoeffs['min_third']*(fInsJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fInsJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fInsJoin + PhiInspcoeffs['min_four_thirds']*(fInsJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fInsJoin**(-5./3.)) + (PhiInspcoeffs['one']*fInsJoin + PhiInspcoeffs['four_thirds']*(fInsJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fInsJoin**(5./3.)) + PhiInspcoeffs['two']*fInsJoin*fInsJoin)/eta
        # This is the Intermediate phase computed at fInsJoin
        PhiIntJoin = beta1*fInsJoin - beta3/(3.*fInsJoin*fInsJoin*fInsJoin) + beta2*np.log(fInsJoin)
        
        C1Int = PhiInsJoin - PhiIntJoin/eta - C2Int*fInsJoin
        
        # Now the same for Intermediate - Merger-Ringdown: we also need a temporary Intermediate Phase function
        PhiIntTempVal  = (beta1*fMRDJoin - beta3/(3.*fMRDJoin*fMRDJoin*fMRDJoin) + beta2*np.log(fMRDJoin))/eta + C1Int + C2Int*fMRDJoin
        DPhiIntTempVal = C2Int + (beta1 + beta3/(fMRDJoin**4) + beta2/fMRDJoin)/eta
        DPhiMRDVal     = (alpha1 + alpha2/(fMRDJoin*fMRDJoin) + alpha3/(fMRDJoin**(1./4.)) + alpha4/(fdamp*(1. + (fMRDJoin - alpha5*fring)*(fMRDJoin - alpha5*fring)/(fdamp*fdamp))))/eta
        PhiMRJoinTemp  = -(alpha2/fMRDJoin) + (4.0/3.0) * (alpha3 * (fMRDJoin**(3./4.))) + alpha1 * fMRDJoin + alpha4 * np.arctan((fMRDJoin - alpha5 * fring)/fdamp)
        
        C2MRD = DPhiIntTempVal - DPhiMRDVal
        C1MRD = PhiIntTempVal - PhiMRJoinTemp/eta - C2MRD*fMRDJoin
        
        fpeak = np.amax(fgrid, axis=0) # In LAL the maximum of the grid is used to rescale
        
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        # LAL sets fRef as the minimum frequency, do the same
        fRef   = np.amin(fgrid, axis=0)
        phiRef = np.where(fRef < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fRef**(2./3.)) + PhiInspcoeffs['third']*(fRef**(1./3.)) + PhiInspcoeffs['third_log']*(fRef**(1./3.))*np.log(np.pi*fRef)/3. + PhiInspcoeffs['log']*np.log(np.pi*fRef)/3. + PhiInspcoeffs['min_third']*(fRef**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fRef**(-2./3.)) + PhiInspcoeffs['min_one']/fRef + PhiInspcoeffs['min_four_thirds']*(fRef**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fRef**(-5./3.)) + (PhiInspcoeffs['one']*fRef + PhiInspcoeffs['four_thirds']*(fRef**(4./3.)) + PhiInspcoeffs['five_thirds']*(fRef**(5./3.)) + PhiInspcoeffs['two']*fRef*fRef)/eta, np.where(fRef<fMRDJoin, (beta1*fRef - beta3/(3.*fRef*fRef*fRef) + beta2*np.log(fRef))/eta + C1Int + C2Int*fRef, np.where(fRef < self.fcutPar, (-(alpha2/fRef) + (4.0/3.0) * (alpha3 * (fRef**(3./4.))) + alpha1 * fRef + alpha4 * np.arctan((fRef - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fRef,0.)))

        phis = np.where(fgrid < self.PHI_fJoin_INS, PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fgrid**(2./3.)) + PhiInspcoeffs['third']*(fgrid**(1./3.)) + PhiInspcoeffs['third_log']*(fgrid**(1./3.))*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['log']*np.log(np.pi*fgrid)/3. + PhiInspcoeffs['min_third']*(fgrid**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fgrid**(-2./3.)) + PhiInspcoeffs['min_one']/fgrid + PhiInspcoeffs['min_four_thirds']*(fgrid**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fgrid**(-5./3.)) + (PhiInspcoeffs['one']*fgrid + PhiInspcoeffs['four_thirds']*(fgrid**(4./3.)) + PhiInspcoeffs['five_thirds']*(fgrid**(5./3.)) + PhiInspcoeffs['two']*fgrid*fgrid)/eta, np.where(fgrid<fMRDJoin, (beta1*fgrid - beta3/(3.*fgrid*fgrid*fgrid) + beta2*np.log(fgrid))/eta + C1Int + C2Int*fgrid, np.where(fgrid <= self.fcutPar, (-(alpha2/fgrid) + (4.0/3.0) * (alpha3 * (fgrid**(3./4.))) + alpha1 * fgrid + alpha4 * np.arctan((fgrid - alpha5 * fring)/fdamp))/eta + C1MRD + C2MRD*fgrid,0.)))
        
        # Add the tidal contribution to the phase, as in arXiv:1905.06011
        # Compute the tidal coupling constant, arXiv:1905.06011 eq. (8) using Lambda = 2/3 k_2/C^5 (eq. (10))

        kappa2T = (3.0/13.0) * ((1.0 + 12.0*m1ByM/m2ByM)*(m2ByM**5)*Lambda)
        
        c_Newt   = 2.4375
        n_1      = -12.615214237993088
        n_3over2 =  19.0537346970349
        n_2      = -21.166863146081035
        n_5over2 =  90.55082156324926
        n_3      = -60.25357801943598
        d_1      = -15.11120782773667
        d_3over2 =  22.195327350624694
        d_2      =   8.064109635305156
    
        numTidal = 1.0 + (n_1 * ((np.pi*fgrid)**(2./3.))) + (n_3over2 * np.pi*fgrid) + (n_2 * ((np.pi*fgrid)**(4./3.))) + (n_5over2 * ((np.pi*fgrid)**(5./3.))) + (n_3 * np.pi*fgrid*np.pi*fgrid)
        denTidal = 1.0 + (d_1 * ((np.pi*fgrid)**(2./3.))) + (d_3over2 * np.pi*fgrid) + (d_2 * ((np.pi*fgrid)**(4./3.)))
        
        tidal_phase = - kappa2T * c_Newt / (m1ByM * m2ByM) * ((np.pi*fgrid)**(5./3.)) * numTidal / denTidal
        
        # This pi factor is needed to include LAL fRef rescaling, so to end up with the exact same waveform
        return phis + np.where(fgrid <= self.fcutPar, - t0*(fgrid - fRef) - phiRef + np.pi +  tidal_phase, 0.)
        
    def Ampl(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # This can speed up a bit, we call it multiple times
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        if np.any(chi2 != 0.):
            print('WARNING: IMRPhenomNSBH is tuned only for chi_NS = 0')
        chi12, chi22 = chi1*chi1, chi2*chi2
        
        Lambda1, Lambda = kwargs['Lambda1'], kwargs['Lambda2']
        
        if np.any(Lambda1 != 0.):
            print('WARNING: BH tidal deformability cannot be different from 0, discarding it')
        
        del Lambda1
        
        Seta = np.sqrt(1.0 - 4.0*eta)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        SetaPlus1 = 1.0 + Seta
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # As in arXiv:0909.2867
        chieff = m1ByM * chi1 + m2ByM * chi2
        chisum = 2.*chieff
        chiprod = chieff*chieff
        
        # compute needed IMRPhenomC attributes
        # First the SPA part, LALSimIMRPhenomC_internals.c line 38
        # Frequency-domain Amplitude coefficients
        xdotaN = 64.*eta/5.
        xdota2 = -7.43/3.36 - 11.*eta/4.
        xdota3 = 4.*np.pi - 11.3*chieff/1.2 + 19.*eta*chisum/6.
        xdota4 = 3.4103/1.8144 + 5*chiprod + eta*(13.661/2.016 - chiprod/8.) + 5.9*eta2/1.8
        xdota5 = -np.pi*(41.59/6.72 + 189.*eta/8.) - chieff*(31.571/1.008 - 116.5*eta/2.4) + chisum*(21.863*eta/1.008 - 79.*eta2/6.) - 3*chieff*chiprod/4. + 9.*eta*chieff*chiprod/4.
        xdota6 = 164.47322263/1.39708800 - 17.12*np.euler_gamma/1.05 + 16.*np.pi*np.pi/3 - 8.56*np.log(16.)/1.05 + eta*(45.1*np.pi*np.pi/4.8 - 561.98689/2.17728) + 5.41*eta2/8.96 - 5.605*eta*eta2/2.592 - 80.*np.pi*chieff/3. + eta*chisum*(20.*np.pi/3. - 113.5*chieff/3.6) + chiprod*(64.153/1.008 - 45.7*eta/3.6) - chiprod*(7.87*eta/1.44 - 30.37*eta2/1.44)
        xdota6log = -856./105.
        xdota7 = -np.pi*(4.415/4.032 - 358.675*eta/6.048 - 91.495*eta2/1.512) - chieff*(252.9407/2.7216 - 845.827*eta/6.048 + 415.51*eta2/8.64) + chisum*(158.0239*eta/5.4432 - 451.597*eta2/6.048 + 20.45*eta2*eta/4.32 + 107.*eta*chiprod/6. - 5.*eta2*chiprod/24.) + 12.*np.pi*chiprod - chiprod*chieff*(150.5/2.4 + eta/8.) + chieff*chiprod*(10.1*eta/2.4 + 3.*eta2/8.)
        # Time-domain amplitude coefficients, which also enters the fourier amplitude in this model
        AN = 8.*eta*np.sqrt(np.pi/5.)
        A2 = (-107. + 55.*eta)/42.
        A3 = 2.*np.pi - 4.*chieff/3. + 2.*eta*chisum/3.
        A4 = -2.173/1.512 - eta*(10.69/2.16 - 2.*chiprod) + 2.047*eta2/1.512
        A5 = -10.7*np.pi/2.1 + eta*(3.4*np.pi/2.1)
        A5imag = -24.*eta
        A6 = 270.27409/6.46800 - 8.56*np.euler_gamma/1.05 + 2.*np.pi*np.pi/3. + eta*(4.1*np.pi*np.pi/9.6 - 27.8185/3.3264) - 20.261*eta2/2.772 + 11.4635*eta*eta2/9.9792 - 4.28*np.log(16.)/1.05
        A6log = -428./105.
        A6imag = 4.28*np.pi/1.05
        
        z701, z702, z711, z710, z720 = 4.149e+00, -4.070e+00, -8.752e+01, -4.897e+01, 6.665e+02
        z801, z802, z811, z810, z820 = -5.472e-02, 2.094e-02, 3.554e-01, 1.151e-01, 9.640e-01
        z901, z902, z911, z910, z920 = -1.235e+00, 3.423e-01, 6.062e+00, 5.949e+00, -1.069e+01
        
        g1 = z701 * chieff + z702 * chiprod + z711 * eta * chieff + z710 * eta + z720 * eta2
        g1 = np.where(g1 < 0., 0., g1)
        
        del1 = z801 * chieff + z802 * chiprod + z811 * eta * chieff + z810 * eta + z820 * eta2
        del2 = z901 * chieff + z902 * chiprod + z911 * eta * chieff + z910 * eta + z920 * eta2
        del1 = np.where(del1 < 0., 0., del1)
        del2 = np.where(del2 < 1.0e-4, 1.0e-4, del2)
        
        d0 = 0.015
        
        # All the other coefficients from IMRPhenomC are not needed
        
        # Now compute NSBH coefficients
        # Get NS compactness and baryonic mass, see arXiv:1608.02582 eq. (78)
        a0Comp = 0.360
        a1Comp = -0.0355
        a2Comp = 0.000705
        
        Comp = np.where(Lambda > 1., a0Comp + a1Comp*np.log(Lambda) + a2Comp*np.log(Lambda)*np.log(Lambda), 0.5 + (3.*a0Comp-a1Comp-1.5)*Lambda*Lambda + (-2.*a0Comp+a1Comp+1.)*Lambda*Lambda*Lambda)
        
        # Get baryonic mass of the torus remnant of a BH-NS merger in units of the NS baryonic mass,
        # see arXiv:1509.00512 eq. (11)
        alphaTor = 0.296
        betaTor = 0.171
        # In LAL the relation is inverted each time, but this would break the vectorisation,
        # we use an interpolator on a grid of Comp, q, chi instead. Already with 200 pts per parameter the
        # agreement we find with LAL waveforms is at machine precision
        
        xiTide = self.xiTide_interp(np.asarray((np.asarray(Comp), np.asarray(q), np.asarray(chi1))).T)
        
        # Compute Kerr BH ISCO radius
        Z1_ISCO = 1.0 + ((1.0 - chi1*chi1)**(1./3.))*((1.0+chi1)**(1./3.) + (1.0-chi1)**(1./3.))
        Z2_ISCO = np.sqrt(3.0*chi1*chi1 + Z1_ISCO*Z1_ISCO)
        r_ISCO  = np.where(chi1>0., 3.0 + Z2_ISCO - np.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)), 3.0 + Z2_ISCO + np.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)))
        
        tmpMtorus = alphaTor * xiTide * (1.0-2.0*Comp) - betaTor * q*Comp * r_ISCO
        
        Mtorus = np.where(tmpMtorus>0., tmpMtorus, 0.)
        
        del tmpMtorus
        
        # Get remnant spin for assumed aligned spin system, from arXiv:1903.11622 Table I and eq. (4), (5) and (6)
        
        p1_remSp = ((-5.44187381e-03*chi1 + 7.91165608e-03) + (2.33362046e-02*chi1 + 2.47764497e-02)*eta)*eta
        p2_remSp = ((-8.56844797e-07*chi1 - 2.81727682e-06) + (6.61290966e-06*chi1 + 4.28979016e-05)*eta)*eta
        p3_remSp = ((-3.04174272e-02*chi1 + 2.54889050e-01) + (1.47549350e-01*chi1 - 4.27905832e-01)*eta)*eta
        
        modelRemSp = (1. + Lambda * p1_remSp + Lambda*Lambda * p2_remSp) / ((1. + Lambda*p3_remSp*p3_remSp)*(1. + Lambda*p3_remSp*p3_remSp))

        modelRemSp = np.where((chi1 < 0.) & (eta < 0.188), 1., modelRemSp)
        modelRemSp = np.where(chi1 < -0.5, 1., modelRemSp)
        modelRemSp = np.where(modelRemSp > 1., 1., modelRemSp)
        
        del p1_remSp, p2_remSp, p3_remSp
        
        # Work with spin variables weighted on square of the BH mass over total mass
        S1BH = chi1 * m1ByM * m1ByM
        Shat = S1BH / (m1ByM*m1ByM + m2ByM*m2ByM) # this would be = (chi1*m1*m1 + chi2*m2*m2)/(m1*m1 + m2*m2), but chi2=0 by assumption
        
        # Compute fit to L_orb in arXiv:1611.00332 eq. (16)
        Lorb = (2.*np.sqrt(3.)*eta + 5.24*3.8326341618708577*eta2 + 1.3*(-9.487364155598392)*eta*eta2)/(1. + 2.88*2.5134875145648374*eta) + ((-0.194)*1.0009563702914628*Shat*(4.409160174224525*eta + 0.5118334706832706*eta2 + (64. - 16.*4.409160174224525 - 4.*0.5118334706832706)*eta2*eta) + 0.0851*0.7877509372255369*Shat*Shat*(8.77367320110712*eta + (-32.060648277652994)*eta2 + (64. - 16.*8.77367320110712 - 4.*(-32.060648277652994))*eta2*eta) + 0.00954*0.6540138407185817*Shat*Shat*Shat*(22.830033250479833*eta + (-153.83722669033995)*eta2 + (64. - 16.*22.830033250479833 - 4.*(-153.83722669033995))*eta2*eta))/(1. + (-0.579)*0.8396665722805308*Shat*(1.8804718791591157 + (-4.770246856212403)*eta + 0.*eta2 + (64. - 64.*1.8804718791591157 - 16.*(-4.770246856212403) - 4.*0.)*eta2*eta)) + 0.3223660562764661*Seta*eta2*(1. + 9.332575956437443*eta)*chi1 + 2.3170397514509933*Shat*Seta*eta2*eta*(1. + (-3.2624649875884852)*eta)*chi1 + (-0.059808322561702126)*eta2*eta*chi12;
        
        chif = (Lorb + S1BH)*modelRemSp
        
        # Get remnant mass scaled to a total (initial) mass of 1
        
        p1_remM = ((-1.83417425e-03*chi1 + 2.39226041e-03) + (4.29407902e-03*chi1 + 9.79775571e-03)*eta)*eta
        p2_remM = ((2.33868869e-07*chi1 - 8.28090025e-07) + (-1.64315549e-06*chi1 + 8.08340931e-06)*eta)*eta
        p3_remM = ((-2.00726981e-02*chi1 + 1.31986011e-01) + (6.50754064e-02*chi1 - 1.42749961e-01)*eta)*eta

        modelRemM = (1. + Lambda * p1_remM + Lambda*Lambda * p2_remM) / ((1. + Lambda*p3_remM*p3_remM)*(1. + Lambda*p3_remM*p3_remM))
        modelRemM = np.where((chi1 < 0.) & (eta < 0.188), 1., modelRemM)
        modelRemM = np.where(chi1 < -0.5, 1., modelRemM)
        modelRemM = np.where(modelRemM > 1., 1., modelRemM)
        
        del p1_remM, p2_remM, p3_remM
        
        # Compute the radiated-energy fit from arXiv:1611.00332 eq. (27)
        EradNSBH = (((1. + -2.0/3.0*np.sqrt(2.))*eta + 0.5609904135313374*eta2 + (-0.84667563764404)*eta2*eta + 3.145145224278187*eta2*eta2)*(1. + 0.346*(-0.2091189048177395)*Shat*(1.8083565298668276 + 15.738082204419655*eta + (16. - 16.*1.8083565298668276 - 4.*15.738082204419655)*eta2) + 0.211*(-0.19709136361080587)*Shat*Shat*(4.271313308472851 + 0.*eta + (16. - 16.*4.271313308472851 - 4.*0.)*eta2) + 0.128*(-0.1588185739358418)*Shat*Shat*Shat*(31.08987570280556 + (-243.6299258830685)*eta + (16. - 16.*31.08987570280556 - 4.*(-243.6299258830685))*eta2)))/(1. + (-0.212)*2.9852925538232014*Shat*(1.5673498395263061 + (-0.5808669012986468)*eta + (16. - 16.*1.5673498395263061 - 4.*(-0.5808669012986468))*eta2)) + (-0.09803730445895877)*Seta*eta2*(1. + (-3.2283713377939134)*eta)*chi1 + (-0.01978238971523653)*Shat*Seta*eta*(1. + (-4.91667749015812)*eta)*chi1 + 0.01118530335431078*eta2*eta*chi12
        finalMass = (1.-EradNSBH)*modelRemM
        
        # Compute 22 quasi-normal mode dimensionless frequency
        kappaOm = np.sqrt(np.log(2.-chif)/np.log(3.))
        omega_tilde = (1.0 + kappaOm*(1.5578*np.exp(1j*2.9031) + 1.9510*np.exp(1j*5.9210)*kappaOm + 2.0997*np.exp(1j*2.7606)*kappaOm*kappaOm + 1.4109*np.exp(1j*5.9143)*kappaOm*kappaOm*kappaOm + 0.4106*np.exp(1j*2.7952)*(kappaOm**4)))
        
        fring = 0.5*np.real(omega_tilde)/np.pi/finalMass
        
        rtide = xiTide * (1.0 - 2.0 * Comp) / (q*Comp)
        
        q_factor = 0.5*np.real(omega_tilde)/np.imag(omega_tilde)
        
        ftide = abs(1.0/(np.pi*(chi1 + np.sqrt(rtide*rtide*rtide)))*(1.0 + 1.0 / q))
        
        # Now compute last amplitude quantities
        fring_tilde = 0.99 * 0.98 * fring
        
        gamma_correction = np.where(Lambda > 1.0, 1.25, 1.0 + 0.5*Lambda - 0.25*Lambda*Lambda)
        delta_2_prime = np.where(Lambda > 1.0, 1.62496*0.25*(1. + np.tanh(4.0*((ftide/fring_tilde - 1.)-0.0188092)/0.338737)), del2 - 2.*(del2 - 0.81248)*Lambda + (del2 - 0.81248)*Lambda*Lambda)
        
        sigma = delta_2_prime * fring / q_factor
        
        # Determine the type of merger we see and determine coefficients
        epsilon_tide = np.where(ftide < fring, 0., 2.*0.25*(1 + np.tanh(4.0*(((ftide/fring_tilde - 1.)*(ftide/fring_tilde - 1.) - 0.571505*Comp - 0.00508451*chi1)+0.0796251)/0.0801192)))
        
        epsilon_ins  = np.where(ftide < fring, np.where(1.29971 - 1.61724 * (Mtorus + 0.424912*Comp + 0.363604*np.sqrt(eta) - 0.0605591*chi1)>1., 1., 1.29971 - 1.61724 * (Mtorus + 0.424912*Comp + 0.363604*np.sqrt(eta) - 0.0605591*chi1)), np.where(Mtorus > 0., 1.29971 - 1.61724 * (Mtorus + 0.424912*Comp + 0.363604*np.sqrt(eta) - 0.0605591*chi1), 1.))
        
        sigma_tide   = np.where(ftide < fring, np.where(Mtorus>0., 0.137722 - 0.293237*(Mtorus - 0.132754*Comp + 0.576669*np.sqrt(eta) - 0.0603749*chi1 - 0.0601185*chi1*chi1 - 0.0729134*chi1*chi1*chi1), 0.5*(0.137722 - 0.293237*(Mtorus - 0.132754*Comp + 0.576669*np.sqrt(eta) - 0.0603749*chi1 - 0.0601185*chi1*chi1 - 0.0729134*chi1*chi1*chi1) + 0.5*(1. - np.tanh(4.0*(((ftide/fring_tilde - 1.)*(ftide/fring_tilde - 1.) - 0.657424*Comp - 0.0259977*chi1)+0.206465)/0.226844)))),0.5*(1. - np.tanh(4.0*(((ftide/fring_tilde - 1.)*(ftide/fring_tilde - 1.) - 0.657424*Comp - 0.0259977*chi1)+0.206465)/0.226844)))
        
        f0_tilde_PN  = np.where(ftide < fring, np.where(Mtorus>0., ftide / (M*utils.GMsun_over_c3), ((1.0 - 1.0 / q) * fring_tilde + epsilon_ins * ftide / q)/(M*utils.GMsun_over_c3)), np.where(Lambda>1., fring_tilde/(M*utils.GMsun_over_c3), ((1.0 - 0.02*Lambda + 0.01*Lambda*Lambda)*0.98*fring)/(M*utils.GMsun_over_c3)))
        
        f0_tilde_PM  = np.where(ftide < fring, np.where(Mtorus>0., ftide / (M*utils.GMsun_over_c3), ((1.0 - 1.0 / q) * fring_tilde + ftide/q)/(M*utils.GMsun_over_c3)), np.where(Lambda>1., fring_tilde/(M*utils.GMsun_over_c3), ((1.0 - 0.02*Lambda + 0.01*Lambda*Lambda)*0.98*fring)/(M*utils.GMsun_over_c3)))
        
        f0_tilde_RD  = np.where(ftide < fring, 0., np.where(Lambda>1., fring_tilde/(M*utils.GMsun_over_c3), ((1.0 - 0.02*Lambda + 0.01*Lambda*Lambda)*0.98*fring)/(M*utils.GMsun_over_c3)))
        
        # This can be used to output the merger type if needed
        merger_type = np.where(ftide < fring, np.where(Mtorus>0., 'DISRUPTIVE', 'MILDLY_DISRUPTIVE_NO_TORUS_REMNANT'), np.where(Mtorus>0.,'MILDLY_DISRUPTIVE_TORUS_REMNANT', 'NON_DISRUPTIVE'))
        
        v = (fgrid*np.pi)**(1./3.)

        xdot = xdotaN*(v**10)*(1. + xdota2*v*v + xdota3 * fgrid*np.pi + xdota4 * fgrid*np.pi*v + xdota5 * v*v*fgrid*np.pi + (xdota6 + xdota6log * 2.*np.log(v)) * fgrid*np.pi*fgrid*np.pi + xdota7 * v*fgrid*np.pi*fgrid*np.pi)
        ampfacTime = np.sqrt(abs(np.pi / (1.5 * v * xdot)))
        
        AmpPNre = ampfacTime * AN * v*v * (1. + A2*v*v + A3 * fgrid*np.pi + A4 * v*fgrid*np.pi + A5 * v*v*fgrid*np.pi + (A6 + A6log * 2.*np.log(v)) * fgrid*np.pi*fgrid*np.pi)
        AmpPNim = ampfacTime * AN * v*v * (A5imag * v*v*fgrid*np.pi + A6imag * fgrid*np.pi*fgrid*np.pi)
        
        aPN = np.sqrt(AmpPNre * AmpPNre + AmpPNim * AmpPNim)
        aPM = (gamma_correction * g1 * (fgrid**(5./6.)))
        
        LRD = sigma*sigma / ((fgrid - fring) * (fgrid - fring) + sigma*sigma*0.25)
        aRD = epsilon_tide * del1 * LRD * (fgrid**(-7./6.))
        
        wMinusf0_PN = 0.5 * (1. - np.tanh(4.*(fgrid - (epsilon_ins * f0_tilde_PN)*M*utils.GMsun_over_c3)/(d0 + sigma_tide)))
        wMinusf0_PM = 0.5 * (1. - np.tanh(4.*(fgrid - f0_tilde_PM*M*utils.GMsun_over_c3)/(d0 + sigma_tide)))
        wPlusf0     = 0.5 * (1. + np.tanh(4.*(fgrid - f0_tilde_RD*M*utils.GMsun_over_c3)/(d0 + sigma_tide)))
        
        amplitudeIMR = np.where(fgrid <= self.fcutPar, (aPN * wMinusf0_PN + aPM * wMinusf0_PM + aRD * wPlusf0), 0.)
        
        # Defined as in LALSimulation - LALSimIMRPhenomD.c line 332. Final units are correctly Hz^-1
        Overallamp = 2. * np.sqrt(5./(64.*np.pi)) * M * utils.GMsun_over_c2_Gpc * M * utils.GMsun_over_c3 / kwargs['dL']
        if not self.returnMergerType:
            return Overallamp*amplitudeIMR
        else:
            return Overallamp*amplitudeIMR, merger_type
    
    def _radiatednrg(self, eta, chi1, chi2):
        # Predict the total radiated energy, from arXiv:1508.07250 eq (3.7) and (3.8)
        Seta = np.sqrt(1.0 - 4.0*eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def tau_star(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # For complex waveforms we use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        return self.fcutPar/(kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.)))
    
    def _tabulate_xiTide(self, res=200, store=True, Compmin=.1, qmax=100.):
        '''
        The ranges are chosen to cover LAL's tuning range:
            - Compactness in [0.1, 0.5] (LAL is tuned up to Lambda=5000, corresponding to C = 0.109);
            - mass ratio, q, in [1, 100];
            - chi_BH in [-1, 1].
        They can easily be changed if needed
        '''
        Compgrid = np.linspace(Compmin, .5, res)
        qgrid = np.linspace(1., qmax, res)
        chigrid = np.linspace(-1.,1.,res)

        def sqrtxifun(Comp, q, chi):
            # Coefficients of eq. (8) of arXiv:1509.00512, using as variable sqrt(xi) (so order 10 polynomial)
            mu = q*Comp
            return np.array([1., 0., -3.*mu,  2.*chi*(mu**(3./2.)), 0., 0., -3.*q, 0., 6.*q*mu, 0., -3.*q*mu*chi*mu*chi])

        xires = np.zeros((res,res,res))
        in_time=time.time()
        for i,Comp in enumerate(Compgrid):
            for j,q in enumerate(qgrid):
                for k,chi in enumerate(chigrid):
                    tmpcoeffs = sqrtxifun(Comp, q, chi)
                    tmproots = np.roots(tmpcoeffs)
                    # We select only real and positive solutions and take the maximum of the squares
                    tmproots_rp = np.real(tmproots[(abs(np.imag(tmproots))<1e-5) & (np.real(tmproots)>0.)])
                    tmpres = max(tmproots_rp*tmproots_rp)
                    xires[i,j,k] = tmpres

        print('Done in %.2fs \n' %(time.time() - in_time))
        if store:
            print('Saving result...')
            if not os.path.isdir(os.path.join(WFfilesPATH)):
                os.mkdir(os.path.join(WFfilesPATH))

            with h5py.File(os.path.join(WFfilesPATH, 'xiTide_Table_'+str(res)+'.h5'), 'w') as out:
                out.create_dataset('Compactness', data=Compgrid, compression='gzip', shuffle=True)
                out.create_dataset('q', data=qgrid, compression='gzip', shuffle=True)
                out.create_dataset('chi', data=chigrid, compression='gzip', shuffle=True)
                out.create_dataset('xiTide', data=xires, compression='gzip', shuffle=True)
                out.attrs['npoints'] = res
                out.attrs['Compactness_min'] = Compmin
                out.attrs['q_max'] = qmax
            print('Done...')

        return xires, Compgrid, qgrid, chigrid

    def _make_xiTide_interpolator(self, res=200):

        from scipy.interpolate import RegularGridInterpolator
        
        if self.path_xiTide_tab is not None:
            if os.path.exists(self.path_xiTide_tab):
                if self.verbose:
                    print('Pre-computed xi_tide grid is present. Loading...')
                with h5py.File(self.path_xiTide_tab, 'r') as inp:
                    Comps = np.array(inp['Compactness'])
                    qs = np.array(inp['q'])
                    chis = np.array(inp['chi'])
                    xiTides = np.array(inp['xiTide'])
                    if self.verbose:
                        print('Attributes of pre-computed grid: ')
                        print([(k, inp.attrs[k]) for k in inp.attrs.keys()])
                        self.verbose=False
            else:
                print('Tabulating xi_tide...')
                xiTides, Comps, qs, chis = self._tabulate_xiTide(res=res)

        else:
            print('Tabulating xi_tide...')
            xiTides, Comps, qs, chis = self._tabulate_xiTide(res=res)

        self.xiTide_interp = RegularGridInterpolator((Comps, qs, chis), xiTides, bounds_error=False)

##############################################################################
# IMRPhenomXAS WAVEFORM
##############################################################################

class IMRPhenomXAS(WaveFormModel):
    '''
    IMRPhenomXAS waveform model
    '''
    # All is taken from LALSimulation and arXiv:2001.11412
    def __init__(self, InsPhaseVersion=104, IntPhaseVersion=105, IntAmpVersion=104, **kwargs):
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.3
        # Version of the inspiral phase to be used
        self.InsPhaseVersion = InsPhaseVersion
        # Version of the intermediate phase to be used
        self.IntPhaseVersion = IntPhaseVersion
        #  Version of the intermediate amplitude to be used
        self.IntAmpVersion   = IntAmpVersion
        
        super().__init__('BBH', fcutPar, **kwargs)
          
    def Phi(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2 = chi1*chi2
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        m1ByMSq = m1ByM*m1ByM
        m2ByMSq = m2ByM*m2ByM
        # PN symmetry coefficient
        delta = np.fabs((m1ByM - m2ByM) / (m1ByM + m2ByM))
        totchi = ((m1ByMSq * chi1 + m2ByMSq * chi2) / (m1ByMSq + m2ByMSq))
        totchi2 = totchi*totchi
        dchi = chi1-chi2
        # Normalised PN reduced spin parameter
        chi_eff = (m1ByM*chi1 + m2ByM*chi2)
        chiPN   = (chi_eff - (38./113.)*eta*(chi1 + chi2)) / (1. - (76.*eta/113.))
        chiPN2  = chiPN*chiPN
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
        
        dphase0      = 5. / (128. * (np.pi**(5./3.)))
        
        gpoints4     = [0., 1./4., 3./4., 1.]
        gpoints5     = [0., 1./2. - 1./(2.*np.sqrt(2.)), 1./2., 1./2 + 1./(2.*np.sqrt(2.)), 1.]
        
        # Compute final spin and radiated energy
        aeff   = (((3.4641016151377544*eta + 20.0830030082033*eta2 - 12.333573402277912*eta2*eta)/(1 + 7.2388440419467335*eta)) + ((m1ByMSq + m2ByMSq)*totchi + ((-0.8561951310209386*eta - 0.09939065676370885*eta2 + 1.668810429851045*eta2*eta)*totchi + (0.5881660363307388*eta - 2.149269067519131*eta2 + 3.4768263932898678*eta2*eta)*totchi2 + (0.142443244743048*eta - 0.9598353840147513*eta2 + 1.9595643107593743*eta2*eta)*totchi2*totchi) / (1 + (-0.9142232693081653 + 2.3191363426522633*eta - 9.710576749140989*eta2*eta)*totchi)) + (0.3223660562764661*dchi*Seta*(1 + 9.332575956437443*eta)*eta2 - 0.059808322561702126*dchi*dchi*eta2*eta + 2.3170397514509933*dchi*Seta*(1 - 3.2624649875884852*eta)*eta2*eta*totchi))
        Erad   = ((((0.057190958417936644*eta + 0.5609904135313374*eta2 - 0.84667563764404*eta2*eta + 3.145145224278187*eta2*eta2)*(1. + (-0.13084389181783257 - 1.1387311580238488*eta + 5.49074464410971*eta2)*totchi + (-0.17762802148331427 + 2.176667900182948*eta2)*totchi2 + (-0.6320191645391563 + 4.952698546796005*eta - 10.023747993978121*eta2)*totchi*totchi2)) / (1. + (-0.9919475346968611 + 0.367620218664352*eta + 4.274567337924067*eta2)*totchi)) + (- 0.09803730445895877*dchi*Seta*(1. - 3.2283713377939134*eta)*eta2 + 0.01118530335431078*dchi*dchi*eta2*eta - 0.01978238971523653*dchi*Seta*(1. - 4.91667749015812*eta)*eta*totchi))
        # Compute ringdown and damping frequencies from fits
        fring = ((0.05947169566573468 - 0.14989771215394762*aeff + 0.09535606290986028*aeff*aeff + 0.02260924869042963*aeff*aeff*aeff - 0.02501704155363241*aeff*aeff*aeff*aeff - 0.005852438240997211*(aeff**5) + 0.0027489038393367993*(aeff**6) + 0.0005821983163192694*(aeff**7))/(1 - 2.8570126619966296*aeff + 2.373335413978394*aeff*aeff - 0.6036964688511505*aeff*aeff*aeff*aeff + 0.0873798215084077*(aeff**6)))/(1. - Erad)
        fdamp = ((0.014158792290965177 - 0.036989395871554566*aeff + 0.026822526296575368*aeff*aeff + 0.0008490933750566702*aeff*aeff*aeff - 0.004843996907020524*aeff*aeff*aeff*aeff - 0.00014745235759327472*(aeff**5) + 0.0001504546201236794*(aeff**6))/(1 - 2.5900842798681376*aeff + 1.8952576220623967*aeff*aeff - 0.31416610693042507*aeff*aeff*aeff*aeff + 0.009002719412204133*(aeff**6)))/(1. - Erad)
        
        # Fitting function for hybrid minimum energy circular orbit (MECO) function and computation of ISCO frequency
        Z1tmp = 1. + np.cbrt((1. - aeff*aeff) ) * (np.cbrt(1. + aeff) + np.cbrt(1. - aeff))
        Z1tmp = np.where(Z1tmp>3., 3., Z1tmp)
        Z2tmp = np.sqrt(3.*aeff*aeff + Z1tmp*Z1tmp)
        fISCO  = (1. / ((3. + Z2tmp - np.sign(aeff)*np.sqrt((3. - Z1tmp) * (3. + Z1tmp + 2.*Z2tmp)))**(3./2.) + aeff))/np.pi
        
        fMECO    = (((0.018744340279608845 + 0.0077903147004616865*eta + 0.003940354686136861*eta2 - 0.00006693930988501673*eta2*eta)/(1. - 0.10423384680638834*eta)) + ((chiPN*(0.00027180386951683135 - 0.00002585252361022052*chiPN + eta2*eta2*(-0.0006807631931297156 + 0.022386313074011715*chiPN - 0.0230825153005985*chiPN2) + eta2*(0.00036556167661117023 - 0.000010021140796150737*chiPN - 0.00038216081981505285*chiPN2) + eta*(0.00024422562796266645 - 0.00001049013062611254*chiPN - 0.00035182990586857726*chiPN2) + eta2*eta*(-0.0005418851224505745 + 0.000030679548774047616*chiPN + 4.038390455349854e-6*chiPN2) - 0.00007547517256664526*chiPN2))/(0.026666543809890402 + (-0.014590539285641243 - 0.012429476486138982*eta + 1.4861197211952053*eta2*eta2 + 0.025066696514373803*eta2 + 0.005146809717492324*eta2*eta)*chiPN + (-0.0058684526275074025 - 0.02876774751921441*eta - 2.551566872093786*eta2*eta2 - 0.019641378027236502*eta2 - 0.001956646166089053*eta2*eta)*chiPN2 + (0.003507640638496499 + 0.014176504653145768*eta + 1.*eta2*eta2 + 0.012622225233586283*eta2 - 0.00767768214056772*eta2*eta)*chiPN2*chiPN)) + (dchi*dchi*(0.00034375176678815234 + 0.000016343732281057392*eta)*eta2 + dchi*Seta*eta*(0.08064665214195679*eta2 + eta*(-0.028476219509487793 - 0.005746537021035632*chiPN) - 0.0011713735642446144*chiPN)))
        
        del Z1tmp, Z2tmp
        
        fIMmatch = 0.6 * (0.5 * fring + fISCO)
        fINmatch = fMECO
        deltaf   = (fIMmatch - fINmatch) * 0.03
        fPhaseMatchIN  = fINmatch - 1.0*deltaf
        fPhaseMatchIM  = fIMmatch + 0.5*deltaf
        fPhaseInsMin  = 0.0026
        fPhaseInsMax  = 1.020 * fMECO
        fPhaseRDMin   = fIMmatch
        fPhaseRDMax   = fring + 1.25*fdamp
        
        #fRef = np.where(np.amin(fgrid, axis=0) > fPhaseInsMin, np.amin(fgrid, axis=0), fPhaseInsMin)
        fRef = np.amin(fgrid, axis=0)
        
        phiNorm = - (3. * (np.pi**(-5./3.)))/ 128.
        
        # Ringdown phase collocation points:
        # Default is to use 5 pseudo-PN coefficients and hence 5 collocation points.
        deltax = fPhaseRDMax - fPhaseRDMin
        xmin   = fPhaseRDMin
        
        CollocationPointsPhaseRD0 = gpoints5[0]* deltax + xmin
        CollocationPointsPhaseRD1 = gpoints5[1]* deltax + xmin
        CollocationPointsPhaseRD2 = gpoints5[2]* deltax + xmin
        # Collocation point 4 is set to the ringdown frequency ~ dip in Lorentzian
        CollocationPointsPhaseRD3 = fring
        CollocationPointsPhaseRD4 = gpoints5[4]* deltax + xmin
        
        CollocationValuesPhaseRD0 = (((eta*(0.7207992174994245 - 1.237332073800276*eta + 6.086871214811216*eta2))/(0.006851189888541745 + 0.06099184229137391*eta - 0.15500218299268662*eta2 + 1.*eta2*eta)) + (((0.06519048552628343 - 25.25397971063995*eta - 308.62513664956975*eta2*eta2 + 58.59408241189781*eta2 + 160.14971486043524*eta2*eta)*totchi + eta*(-5.215945111216946 + 153.95945758807616*eta - 693.0504179144295*eta2 + 835.1725103648205*eta2*eta)*totchi2 + (0.20035146870472367 - 0.28745205203100666*eta - 47.56042058800358*eta2*eta2)*totchi2*totchi + eta*(5.7756520242745735 - 43.97332874253772*eta + 338.7263666984089*eta2*eta)*totchi2*totchi2 + (-0.2697933899920511 + 4.917070939324979*eta - 22.384949087140086*eta2*eta2 - 11.61488280763592*eta2)*totchi2*totchi2*totchi)/(1. - 0.6628745847248266*totchi)) + (-23.504907495268824*dchi*delta*eta2))
        CollocationValuesPhaseRD1 = (((eta*(-9.460253118496386 + 9.429314399633007*eta + 64.69109972468395*eta2))/(-0.0670554310666559 - 0.09987544893382533*eta + 1.*eta2)) + ((17.36495157980372*eta*totchi + eta2*eta*totchi*(930.3458437154668 + 808.457330742532*totchi) + eta2*eta2*totchi*(-774.3633787391745 - 2177.554979351284*totchi - 1031.846477275069*totchi2) + eta2*totchi*(-191.00932194869588 - 62.997389062600035*totchi + 64.42947340363101*totchi2) + 0.04497628581617564*totchi2*totchi)/(1. - 0.7267610313751913*totchi)) + (dchi*delta*(-36.66374091965371 + 91.60477826830407*eta)*eta2))
        CollocationValuesPhaseRD2 = (((eta*(-8.506898502692536 + 13.936621412517798*eta))/(-0.40919671232073945 + 1.*eta)) + ((eta*(1.7280582989361533*totchi + 18.41570325463385*totchi2*totchi - 13.743271480938104*totchi2*totchi2) + eta2*(73.8367329022058*totchi - 95.57802408341716*totchi2*totchi + 215.78111099820157*totchi2*totchi2) + 0.046849371468156265*totchi2 + eta2*eta*totchi*(-27.976989112929353 + 6.404060932334562*totchi - 633.1966645925428*totchi2*totchi + 109.04824706217418*totchi2))/(1. - 0.6862449113932192*totchi)) + (641.8965762829259*dchi*delta*eta2*eta2*eta))
        CollocationValuesPhaseRD3 = (((-85.86062966719405 - 4616.740713893726*eta - 4925.756920247186*eta2 + 7732.064464348168*eta2*eta + 12828.269960300782*eta2*eta2 - 39783.51698102803*eta2*eta2*eta)/(1. + 50.206318806624004*eta)) + ((totchi*(33.335857451144356 - 36.49019206094966*totchi + eta2*eta*(1497.3545918387515 - 101.72731770500685*totchi)*totchi - 3.835967351280833*totchi2 + 2.302712009652155*totchi2*totchi + eta2*(93.64156367505917 - 18.184492163348665*totchi + 423.48863373726243*totchi2 - 104.36120236420928*totchi2*totchi - 719.8775484010988*totchi2*totchi2) + 1.6533417657003922*totchi2*totchi2 + eta*(-69.19412903018717 + 26.580344399838758*totchi - 15.399770764623746*totchi2 + 31.231253209893488*totchi2*totchi + 97.69027029734173*totchi2*totchi2) + eta2*eta2*(1075.8686153198323 - 3443.0233614187396*totchi - 4253.974688619423*totchi2 - 608.2901586790335*totchi2*totchi + 5064.173605639933*totchi2*totchi2)))/(-1.3705601055555852 + 1.*totchi)) + (dchi*delta*eta*(22.363215261437862 + 156.08206945239374*eta)))
        CollocationValuesPhaseRD4 = (((eta*(7.05731400277692 + 22.455288821807095*eta + 119.43820622871043*eta2))/(0.26026709603623255 + 1.*eta)) + ((eta2*(134.88158268621922 - 56.05992404859163*totchi)*totchi + eta*totchi*(-7.9407123129681425 + 9.486783128047414*totchi) + eta2*eta*totchi*(-316.26970506215554 + 90.31815139272628*totchi))/(1. - 0.7162058321905909*totchi)) + (43.82713604567481*dchi*delta*eta2*eta))

        CollocationValuesPhaseRD4 = CollocationValuesPhaseRD4 + CollocationValuesPhaseRD3
        CollocationValuesPhaseRD2 = CollocationValuesPhaseRD2 + CollocationValuesPhaseRD3
        CollocationValuesPhaseRD1 = CollocationValuesPhaseRD1 + CollocationValuesPhaseRD3
        CollocationValuesPhaseRD0 = CollocationValuesPhaseRD0 + CollocationValuesPhaseRD1
        
        A0i = np.array([np.ones(CollocationPointsPhaseRD0.shape), CollocationPointsPhaseRD0**(-1./3.), CollocationPointsPhaseRD0**(-2), CollocationPointsPhaseRD0**(-4), -(dphase0) / (fdamp*fdamp + (CollocationPointsPhaseRD0 - fring)*(CollocationPointsPhaseRD0 - fring))])
        A1i = np.array([np.ones(CollocationPointsPhaseRD1.shape), CollocationPointsPhaseRD1**(-1./3.), CollocationPointsPhaseRD1**(-2), CollocationPointsPhaseRD1**(-4), -(dphase0) / (fdamp*fdamp + (CollocationPointsPhaseRD1 - fring)*(CollocationPointsPhaseRD1 - fring))])
        A2i = np.array([np.ones(CollocationPointsPhaseRD2.shape), CollocationPointsPhaseRD2**(-1./3.), CollocationPointsPhaseRD2**(-2), CollocationPointsPhaseRD2**(-4), -(dphase0) / (fdamp*fdamp + (CollocationPointsPhaseRD2 - fring)*(CollocationPointsPhaseRD2 - fring))])
        A3i = np.array([np.ones(CollocationPointsPhaseRD3.shape), CollocationPointsPhaseRD3**(-1./3.), CollocationPointsPhaseRD3**(-2), CollocationPointsPhaseRD3**(-4), -(dphase0) / (fdamp*fdamp + (CollocationPointsPhaseRD3 - fring)*(CollocationPointsPhaseRD3 - fring))])
        A4i = np.array([np.ones(CollocationPointsPhaseRD4.shape), CollocationPointsPhaseRD4**(-1./3.), CollocationPointsPhaseRD4**(-2), CollocationPointsPhaseRD4**(-4), -(dphase0) / (fdamp*fdamp + (CollocationPointsPhaseRD4 - fring)*(CollocationPointsPhaseRD4 - fring))])
        
        Acoloc = np.array([A0i, A1i, A2i, A3i, A4i]).transpose(2,0,1)
        bcoloc = np.array([CollocationValuesPhaseRD0, CollocationValuesPhaseRD1, CollocationValuesPhaseRD2, CollocationValuesPhaseRD3, CollocationValuesPhaseRD4]).T
        
        coeffscoloc = np.linalg.solve(Acoloc, bcoloc)
        c0coloc  = coeffscoloc[:,0]
        c1coloc  = coeffscoloc[:,1]
        c2coloc  = coeffscoloc[:,2]
        c4coloc  = coeffscoloc[:,3]
        cRDcoloc = coeffscoloc[:,4]
        cLcoloc  = -(dphase0 * cRDcoloc)
        phaseRD  = CollocationValuesPhaseRD0
        
        # Inspiral phase collocation points
        # Default is to use 4 pseudo-PN coefficients and hence 4 collocation points.
        
        deltax      = fPhaseInsMax - fPhaseInsMin
        xmin        = fPhaseInsMin
        
        if self.InsPhaseVersion == 104:
            CollocationPointsPhaseIns0 = gpoints4[0]* deltax + xmin
            CollocationPointsPhaseIns1 = gpoints4[1]* deltax + xmin
            CollocationPointsPhaseIns2 = gpoints4[2]* deltax + xmin
            CollocationPointsPhaseIns3 = gpoints4[3]* deltax + xmin

            CollocationValuesPhaseIns0 = (((-17294.000000000007 - 19943.076428555978*eta + 483033.0998073767*eta2)/(1. + 4.460294035404433*eta)) + ((chiPN*(68384.62786426462 + 67663.42759836042*chiPN - 2179.3505885609297*chiPN2 + eta*(-58475.33302037833 + 62190.404951852535*chiPN + 18298.307770807573*chiPN2 - 303141.1945565486*chiPN2*chiPN) + 19703.894135534803*chiPN2*chiPN + eta2*(-148368.4954044637 - 758386.5685734496*chiPN - 137991.37032619823*chiPN2 + 1.0765877367729193e6*chiPN2*chiPN) + 32614.091002011017*chiPN2*chiPN2))/(2.0412979553629143 + 1.*chiPN)) + (12017.062595934838*dchi*delta*eta))
            CollocationValuesPhaseIns1 = (((-7579.300000000004 - 120297.86185566607*eta + 1.1694356931282217e6*eta2 - 557253.0066989232*eta2*eta)/(1. + 18.53018618227582*eta)) + ((chiPN*(-27089.36915061857 - 66228.9369155027*chiPN + eta2*(150022.21343386435 - 50166.382087278434*chiPN - 399712.22891153296*chiPN2) - 44331.41741405198*chiPN2 + eta*(50644.13475990821 + 157036.45676788126*chiPN + 126736.43159783827*chiPN2) + eta2*eta*(-593633.5370110178 - 325423.99477314285*chiPN + 847483.2999508682*chiPN2)))/(-1.5232497464826662 - 3.062957826830017*chiPN - 1.130185486082531*chiPN2 + 1.*chiPN2*chiPN)) + (3843.083992827935*dchi*delta*eta))
            CollocationValuesPhaseIns2 = (((15415.000000000007 + 873401.6255736464*eta + 376665.64637025696*eta2 - 3.9719980569125614e6*eta2*eta + 8.913612508054944e6*eta2*eta2)/(1. + 46.83697749859996*eta)) + ((chiPN*(397951.95299014193 - 207180.42746987*chiPN + eta2*eta*(4.662143741417853e6 - 584728.050612325*chiPN - 1.6894189124921719e6*chiPN2) + eta*(-1.0053073129700898e6 + 1.235279439281927e6*chiPN - 174952.69161683554*chiPN2) - 130668.37221912303*chiPN2 + eta2*(-1.9826323844247842e6 + 208349.45742548333*chiPN + 895372.155565861*chiPN2)))/(-9.675704197652225 + 3.5804521763363075*chiPN + 2.5298346636273306*chiPN2 + 1.*chiPN2*chiPN)) + (-1296.9289110696955*dchi*dchi*eta + dchi*delta*eta*(-24708.109411857182 + 24703.28267342699*eta + 47752.17032707405*chiPN)))
            CollocationValuesPhaseIns3 = (((2439.000000000001 - 31133.52170083207*eta + 28867.73328134167*eta2)/(1. + 0.41143032589262585*eta)) + ((chiPN*(16116.057657391262 + eta2*eta*(-375818.0132734753 - 386247.80765802023*chiPN) + eta*(-82355.86732027541 - 25843.06175439942*chiPN) + 9861.635308837876*chiPN + eta2*(229284.04542668918 + 117410.37432997991*chiPN)))/(-3.7385208695213668 + 0.25294420589064653*chiPN + 1.*chiPN)) + (194.5554531509207*dchi*delta*eta))

            CollocationValuesPhaseIns0 = CollocationValuesPhaseIns0 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns1 = CollocationValuesPhaseIns1 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns3 = CollocationValuesPhaseIns3 + CollocationValuesPhaseIns2

            A0i = np.array([np.ones(CollocationPointsPhaseIns0.shape), CollocationPointsPhaseIns0**(1./3.), CollocationPointsPhaseIns0**(2./3.), CollocationPointsPhaseIns0])
            A1i = np.array([np.ones(CollocationPointsPhaseIns1.shape), CollocationPointsPhaseIns1**(1./3.), CollocationPointsPhaseIns1**(2./3.), CollocationPointsPhaseIns1])
            A2i = np.array([np.ones(CollocationPointsPhaseIns2.shape), CollocationPointsPhaseIns2**(1./3.), CollocationPointsPhaseIns2**(2./3.), CollocationPointsPhaseIns2])
            A3i = np.array([np.ones(CollocationPointsPhaseIns3.shape), CollocationPointsPhaseIns3**(1./3.), CollocationPointsPhaseIns3**(2./3.), CollocationPointsPhaseIns3])

            Acoloc = np.array([A0i, A1i, A2i, A3i]).transpose(2,0,1)#.T
            bcoloc = np.array([CollocationValuesPhaseIns0, CollocationValuesPhaseIns1, CollocationValuesPhaseIns2, CollocationValuesPhaseIns3]).T

            coeffscoloc = np.linalg.solve(Acoloc, bcoloc)
            a0coloc  = coeffscoloc[:,0]
            a1coloc  = coeffscoloc[:,1]
            a2coloc  = coeffscoloc[:,2]
            a3coloc  = coeffscoloc[:,3]
            a4coloc  = np.zeros_like(a3coloc)
        
        elif self.InsPhaseVersion == 114:
            
            CollocationPointsPhaseIns0 = gpoints4[0]* deltax + xmin
            CollocationPointsPhaseIns1 = gpoints4[1]* deltax + xmin
            CollocationPointsPhaseIns2 = gpoints4[2]* deltax + xmin
            CollocationPointsPhaseIns3 = gpoints4[3]* deltax + xmin
            
            CollocationValuesPhaseIns0 = (((-36664.000000000015 + 277640.10051158903*eta - 581120.4916255298*eta2 + 1.415628418251648e6*eta2*eta - 7.640937162029471e6*eta2*eta2 + 1.1572710625359124e7*eta2*eta2*eta)/(1. - 4.011038704323779*eta)) + ((chiPN*(-38790.01253014577 - 50295.77273512981*chiPN + 15182.324439704937*chiPN2 + eta2*(57814.07222969789 + 344650.11918139807*chiPN + 17020.46497164955*chiPN2 - 574047.1384792664*chiPN2*chiPN) + 24626.598127509922*chiPN2*chiPN + eta*(23058.264859112394 - 16563.935447608965*chiPN - 36698.430436426395*chiPN2 + 105713.91549712936*chiPN2*chiPN)))/(-1.5445637219268247 - 0.24997068896075847*chiPN + 1.*chiPN2)) + (74115.77361380383*dchi*delta*eta2))
            CollocationValuesPhaseIns1 = (((-17762.000000000007 - 1.6929191194109183e6*eta + 8.420903644926643e6*eta2)/(1. + 98.061533474615*eta)) + ((chiPN*(-46901.6486082098 - 83648.57463631754*chiPN + eta2*(1.2502334322912344e6 + 1.4500798116821344e6*chiPN - 1.4822181506831646e6*chiPN2) - 41234.966418619966*chiPN2 + eta*(-24017.33452114588 - 15241.079745314566*chiPN + 136554.48806839858*chiPN2) + eta2*eta*(-3.584298922116994e6 - 3.9566921791790277e6*chiPN + 4.357599992831832e6*chiPN2)))/(-3.190220646817508 - 3.4308485421201387*chiPN - 0.6347932583034377*chiPN2 + 1.*chiPN2*chiPN)) + (24906.33337911219*dchi*delta*eta2))
            CollocationValuesPhaseIns2 = (((68014.00000000003 + 1.1513072539654972e6*eta - 2.725589921577228e6*eta2 + 312571.92531733884*eta2*eta)/(1. + 17.48539665509149*eta)) + ((chiPN*(-34467.00643820664 + 99693.81839115614*eta + 144345.24343461913*eta2*eta2 + (23618.044919850676 - 89494.69555164348*eta + 725554.5749749158*eta2*eta2 - 103449.15865381068*eta2)*chiPN + (10350.863429774612 - 73238.45609787296*eta + 3.559251543095961e6*eta2*eta2 + 888228.5439003729*eta2 - 3.4602940487291473e6*eta2*eta)*chiPN2))/(1. - 0.056846656084188936*chiPN - 0.32681474740130184*chiPN2 - 0.30562055811022015*chiPN2*chiPN)) + (-1182.4036752941936*dchi*dchi*eta + dchi*delta*eta*(-0.39185419821851025 - 99764.21095663306*eta + 41826.177356107364*chiPN)))
            CollocationValuesPhaseIns3 = (((5749.000000000003 - 37877.95816426952*eta)/(1. + 1.1883386102990128*eta)) + (((-4285.982163759047 + 24558.689969419473*eta - 49270.2296311733*eta2)*chiPN + eta*(-24205.71407420114 + 70777.38402634041*eta)*chiPN2 + (2250.661418551257 + 187.95136178643946*eta - 11976.624134935797*eta2)*chiPN2*chiPN)/(1. - 0.7220334077284601*chiPN)) + (dchi*delta*eta*(339.69292150803585 - 3459.894150148715*chiPN)))
            
            CollocationValuesPhaseIns0 = CollocationValuesPhaseIns0 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns1 = CollocationValuesPhaseIns1 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns3 = CollocationValuesPhaseIns3 + CollocationValuesPhaseIns2

            A0i = np.array([np.ones(CollocationPointsPhaseIns0.shape), CollocationPointsPhaseIns0**(1./3.), CollocationPointsPhaseIns0**(2./3.), CollocationPointsPhaseIns0])
            A1i = np.array([np.ones(CollocationPointsPhaseIns1.shape), CollocationPointsPhaseIns1**(1./3.), CollocationPointsPhaseIns1**(2./3.), CollocationPointsPhaseIns1])
            A2i = np.array([np.ones(CollocationPointsPhaseIns2.shape), CollocationPointsPhaseIns2**(1./3.), CollocationPointsPhaseIns2**(2./3.), CollocationPointsPhaseIns2])
            A3i = np.array([np.ones(CollocationPointsPhaseIns3.shape), CollocationPointsPhaseIns3**(1./3.), CollocationPointsPhaseIns3**(2./3.), CollocationPointsPhaseIns3])

            Acoloc = np.array([A0i, A1i, A2i, A3i]).transpose(2,0,1)
            bcoloc = np.array([CollocationValuesPhaseIns0, CollocationValuesPhaseIns1, CollocationValuesPhaseIns2, CollocationValuesPhaseIns3]).T

            coeffscoloc = np.linalg.solve(Acoloc, bcoloc)
            a0coloc  = coeffscoloc[:,0]
            a1coloc  = coeffscoloc[:,1]
            a2coloc  = coeffscoloc[:,2]
            a3coloc  = coeffscoloc[:,3]
            a4coloc  = np.zeros_like(a3coloc)
            
        elif self.InsPhaseVersion == 105:
            
            CollocationPointsPhaseIns0 = gpoints5[0]* deltax + xmin
            CollocationPointsPhaseIns1 = gpoints5[1]* deltax + xmin
            CollocationPointsPhaseIns2 = gpoints5[2]* deltax + xmin
            CollocationPointsPhaseIns3 = gpoints5[3]* deltax + xmin
            CollocationPointsPhaseIns4 = gpoints5[4]* deltax + xmin
            
            CollocationValuesPhaseIns0 = (((-14234.000000000007 + 16956.107542097994*eta + 176345.7518697656*eta2)/(1. + 1.294432443903631*eta)) + ((chiPN*(814.4249470391651 + 539.3944162216742*chiPN + 1985.8695471257474*chiPN2 + eta*(-918.541450687484 + 2531.457116826593*chiPN - 14325.55532310136*chiPN2 - 19213.48002675173*chiPN2*chiPN) + 1517.4561843692861*chiPN2*chiPN + eta2*(-517.7142591907573 - 14328.567448748548*chiPN + 21305.033147575057*chiPN2 + 50143.99945676916*chiPN2*chiPN)))/(0.03332712934306297 + 0.0025905919215826172*chiPN + (0.07388087063636305 - 0.6397891808905215*eta + 1.*eta2)*chiPN2)) + (dchi*delta*eta*(0.09704682517844336 + 69335.84692284222*eta)))
            CollocationValuesPhaseIns1 = (((-7520.900000000003 - 49463.18828584058*eta + 437634.8057596484*eta2)/(1. + 9.10538019868398*eta)) + ((chiPN*(25380.485895523005 + 30617.553968012628*chiPN + 5296.659585425608*chiPN2 + eta*(-49447.74841021405 - 94312.78229903466*chiPN - 5731.614612941746*chiPN2*chiPN) + 2609.50444822972*chiPN2*chiPN + 5206.717656940992*chiPN2*chiPN2 + eta2*(54627.758819129864 + 157460.98527210607*chiPN - 69726.85196686552*chiPN2 + 4674.992397927943*chiPN2*chiPN + 20704.368650323784*chiPN2*chiPN2)))/(1.5668927528319367 + 1.*chiPN)) + (-95.38600275845481*dchi*dchi*eta + dchi*delta*eta*(3271.8128884730654 + 12399.826237672185*eta + 9343.380589951552*chiPN)))
            CollocationValuesPhaseIns2 = (((11717.332402222377 + 4.972361134612872e6*eta + 2.137585030930089e7*eta2 - 8.882868155876668e7*eta2*eta + 2.4104945956043008e8*eta2*eta2 - 2.3143426271719798e8*eta2*eta2*eta)/(1. + 363.5524719849582*eta)) + ((chiPN*(52.42001436116159 - 50.547943589389966*chiPN + eta2*eta*chiPN*(-15355.56020802297 + 20159.588079899433*chiPN) + eta2*(-286.9576245212502 + 2795.982637986682*chiPN - 2633.1870842242447*chiPN2) - 1.0824224105690476*chiPN2 + eta*(-123.78531181532225 + 136.1961976556154*chiPN - 7.534890781638552*chiPN2*chiPN) + 5.973206330664007*chiPN2*chiPN + eta2*eta2*(1777.2176433016125 + 24069.288079063674*chiPN - 44794.9522164669*chiPN2 + 1584.1515277998406*chiPN2*chiPN)))/(-0.0015307616935628491 + (0.0010676159178395538 - 0.25*eta2*eta + 1.*eta2*eta2)*chiPN)) + (-1357.9794908614106*dchi*dchi*eta + dchi*delta*eta*(-23093.829989687543 + 21908.057881789653*eta + 49493.91485992256*chiPN)))
            CollocationValuesPhaseIns3 = (((4085.300000000002 + 62935.7755506329*eta - 1.3712743918777364e6*eta2 + 5.024685134555112e6*eta2*eta - 3.242443755025284e6*eta2*eta2)/(1. + 20.889132970603523*eta - 99.39498823723363*eta2)) + ((chiPN*(-299.6987332025542 - 106.2596940493108*chiPN + eta2*eta*(2383.4907865977148 - 13637.11364447208*chiPN - 14808.138346145908*chiPN2) + eta*(1205.2993091547498 - 508.05838536573464*chiPN - 1453.1997617403304*chiPN2) + 132.22338129554674*chiPN2 + eta2*(-2438.4917103042208 + 5032.341879949591*chiPN + 7026.9206794027405*chiPN2)))/(0.03089183275944264 + 1.*eta2*eta*chiPN - 0.010670764224621489*chiPN2)) + (-1392.6847421907178*dchi*delta*eta))
            CollocationValuesPhaseIns4 = (((5474.400000000003 + 131008.0112992443*eta - 1.9692364337640922e6*eta2 + 1.8732325307375633e6*eta2*eta)/(1. + 32.90929274981482*eta)) + ((chiPN*(18609.016486281424 - 1337.4947536109685*chiPN + eta2*(219014.98908698096 - 307162.33823247004*chiPN - 124111.02067626518*chiPN2) - 7394.9595046977365*chiPN2 + eta*(-87820.37490863055 + 53564.4178831741*chiPN + 34070.909093771494*chiPN2) + eta2*eta*(-248096.84257893753 + 536024.5354098587*chiPN + 243877.65824670633*chiPN2)))/(-1.5282904337787517 + 1.*chiPN)) + (-429.1148607925461*dchi*delta*eta))
            
            CollocationValuesPhaseIns0 = CollocationValuesPhaseIns0 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns1 = CollocationValuesPhaseIns1 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns3 = CollocationValuesPhaseIns3 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns4 = CollocationValuesPhaseIns4 + CollocationValuesPhaseIns2
            
            A0i = np.array([np.ones(CollocationPointsPhaseIns0.shape), CollocationPointsPhaseIns0**(1./3.), CollocationPointsPhaseIns0**(2./3.), CollocationPointsPhaseIns0, CollocationPointsPhaseIns0**(4./3.)])
            A1i = np.array([np.ones(CollocationPointsPhaseIns1.shape), CollocationPointsPhaseIns1**(1./3.), CollocationPointsPhaseIns1**(2./3.), CollocationPointsPhaseIns1, CollocationPointsPhaseIns1**(4./3.)])
            A2i = np.array([np.ones(CollocationPointsPhaseIns2.shape), CollocationPointsPhaseIns2**(1./3.), CollocationPointsPhaseIns2**(2./3.), CollocationPointsPhaseIns2, CollocationPointsPhaseIns2**(4./3.)])
            A3i = np.array([np.ones(CollocationPointsPhaseIns3.shape), CollocationPointsPhaseIns3**(1./3.), CollocationPointsPhaseIns3**(2./3.), CollocationPointsPhaseIns3, CollocationPointsPhaseIns3**(4./3.)])
            A4i = np.array([np.ones(CollocationPointsPhaseIns4.shape), CollocationPointsPhaseIns4**(1./3.), CollocationPointsPhaseIns4**(2./3.), CollocationPointsPhaseIns4, CollocationPointsPhaseIns4**(4./3.)])
            
            Acoloc = np.array([A0i, A1i, A2i, A3i, A4i]).transpose(2,0,1)
            bcoloc = np.array([CollocationValuesPhaseIns0, CollocationValuesPhaseIns1, CollocationValuesPhaseIns2, CollocationValuesPhaseIns3, CollocationValuesPhaseIns4]).T
        
            coeffscoloc = np.linalg.solve(Acoloc, bcoloc)
            a0coloc  = coeffscoloc[:,0]
            a1coloc  = coeffscoloc[:,1]
            a2coloc  = coeffscoloc[:,2]
            a3coloc  = coeffscoloc[:,3]
            a4coloc  = coeffscoloc[:,4]
            
        elif self.InsPhaseVersion == 115:
            
            CollocationPointsPhaseIns0 = gpoints5[0]* deltax + xmin
            CollocationPointsPhaseIns1 = gpoints5[1]* deltax + xmin
            CollocationPointsPhaseIns2 = gpoints5[2]* deltax + xmin
            CollocationPointsPhaseIns3 = gpoints5[3]* deltax + xmin
            CollocationPointsPhaseIns4 = gpoints5[4]* deltax + xmin
            
            CollocationValuesPhaseIns0 = (((-29240.00000000001 - 12488.41035199958*eta + 1.3911845288427814e6*eta2 - 3.492477584609041e6*eta2*eta)/(1. + 2.6711462529779824*eta - 26.80876660227278*eta2)) + ((chiPN*(-29536.155624432842 - 40024.5988680615*chiPN + 11596.401177843705*chiPN2 + eta2*(122185.06346551726 + 351091.59147835104*chiPN - 37366.6143666202*chiPN2 - 505834.54206320125*chiPN2*chiPN) + 20386.815769841945*chiPN2*chiPN + eta*(-9638.828456576934 - 30683.453790630676*chiPN - 15694.962417099561*chiPN2 + 91690.51338194775*chiPN2*chiPN)))/(-1.5343852108869265 - 0.2215651087041266*chiPN + 1.*chiPN2)) + (68951.75344813892*dchi*delta*eta2))
            CollocationValuesPhaseIns1 = (((-18482.000000000007 - 1.2846128476247871e6*eta + 4.894853535651343e6*eta2 + 3.1555931338015324e6*eta2*eta)/(1. + 82.79386070797756*eta)) + ((chiPN*(-19977.10130179636 - 24729.114403562427*chiPN + 10325.295899053815*chiPN2 + eta*(30934.123894659646 + 58636.235226102894*chiPN - 32465.70372990005*chiPN2 - 38032.16219587224*chiPN2*chiPN) + 15485.725898689267*chiPN2*chiPN + eta2*(-38410.1127729419 - 87088.84559983511*chiPN + 61286.73536122325*chiPN2 + 42503.913487705235*chiPN2*chiPN)))/(-1.5148031011828222 - 0.24267195338525768*chiPN + 1.*chiPN2)) + (5661.027157084334*dchi*delta*eta))
            CollocationValuesPhaseIns2 = (((60484.00000000003 + 4.370611564781374e6*eta - 5.785128542827255e6*eta2 - 8.82141241633613e6*eta2*eta + 1.3406996319926713e7*eta2*eta2)/(1. + 70.89393713617065*eta)) + ((chiPN*(21.91241092620993 - 32.57779678272056*chiPN + eta2*(-102.4982890239095 + 2570.7566494633033*chiPN - 2915.1250015652076*chiPN2) + 8.130585173873232*chiPN2 + eta*(-28.158436727309365 + 47.42665852824481*chiPN2) + eta2*eta*(-1635.6270690726785 - 13745.336370568011*chiPN + 19539.310912464192*chiPN2) + 1.2792283911312285*chiPN2*chiPN + eta2*eta2*(5558.382039622131 + 21398.7730201213*chiPN - 37773.40511355719*chiPN2 + 768.6183344184254*chiPN2*chiPN)))/(-0.0007758753818017038 + (0.0005304023864415552 - 0.25000000000000006*eta2*eta + 1.*eta2*eta2)*chiPN)) + (-1223.769262298142*dchi*dchi*eta + dchi*delta*eta*(-16.705471562129436 - 93771.93750060834*eta + 43675.70151058481*chiPN)))
            CollocationValuesPhaseIns3 = (((9760.400000000005 + 9839.852773121198*eta - 398521.0434645335*eta2 + 267575.4709475981*eta2*eta)/(1. + 6.1355249449135005*eta)) + ((chiPN*(-1271.406488219572 + eta2*(-9641.611385554736 - 9620.333878140807*chiPN) - 1144.516395171019*chiPN + eta*(5155.337817255137 + 4450.755534615418*chiPN)))/(0.1491519640750958 + (-0.0008208549820159909 - 0.15468508831447628*eta + 0.7266887643762937*eta2)*chiPN + (0.02282542856845755 - 0.445924460572114*eta + 1.*eta2)*chiPN2)) + (-1366.7949288045616*dchi*delta*eta))
            CollocationValuesPhaseIns4 = (((12971.000000000005 - 93606.05144508784*eta + 102472.4473167639*eta2)/(1. - 0.8909484992212859*eta)) + ((chiPN*(16182.268123259992 + 3513.8535400032874*chiPN + eta2*(343388.99445324624 - 240407.0282222587*chiPN - 312202.59917289804*chiPN2) - 10814.056847109632*chiPN2 + eta*(-94090.9232151429 + 35305.66247590705*chiPN + 65450.36389642103*chiPN2) + eta2*eta*(-484443.15601144277 + 449511.3965208116*chiPN + 552355.592066788*chiPN2)))/(-1.4720837917195788 + 1.*chiPN)) + (-494.2754225110706*dchi*delta*eta))
            
            CollocationValuesPhaseIns0 = CollocationValuesPhaseIns0 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns1 = CollocationValuesPhaseIns1 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns3 = CollocationValuesPhaseIns3 + CollocationValuesPhaseIns2
            CollocationValuesPhaseIns4 = CollocationValuesPhaseIns4 + CollocationValuesPhaseIns2
            
            A0i = np.array([np.ones(CollocationPointsPhaseIns0.shape), CollocationPointsPhaseIns0**(1./3.), CollocationPointsPhaseIns0**(2./3.), CollocationPointsPhaseIns0, CollocationPointsPhaseIns0**(4./3.)])
            A1i = np.array([np.ones(CollocationPointsPhaseIns1.shape), CollocationPointsPhaseIns1**(1./3.), CollocationPointsPhaseIns1**(2./3.), CollocationPointsPhaseIns1, CollocationPointsPhaseIns1**(4./3.)])
            A2i = np.array([np.ones(CollocationPointsPhaseIns2.shape), CollocationPointsPhaseIns2**(1./3.), CollocationPointsPhaseIns2**(2./3.), CollocationPointsPhaseIns2, CollocationPointsPhaseIns2**(4./3.)])
            A3i = np.array([np.ones(CollocationPointsPhaseIns3.shape), CollocationPointsPhaseIns3**(1./3.), CollocationPointsPhaseIns3**(2./3.), CollocationPointsPhaseIns3, CollocationPointsPhaseIns3**(4./3.)])
            A4i = np.array([np.ones(CollocationPointsPhaseIns4.shape), CollocationPointsPhaseIns4**(1./3.), CollocationPointsPhaseIns4**(2./3.), CollocationPointsPhaseIns4, CollocationPointsPhaseIns4**(4./3.)])
            
            Acoloc = np.array([A0i, A1i, A2i, A3i, A4i]).transpose(2,0,1)
            bcoloc = np.array([CollocationValuesPhaseIns0, CollocationValuesPhaseIns1, CollocationValuesPhaseIns2, CollocationValuesPhaseIns3, CollocationValuesPhaseIns4]).T
        
            coeffscoloc = np.linalg.solve(Acoloc, bcoloc)
            a0coloc  = coeffscoloc[:,0]
            a1coloc  = coeffscoloc[:,1]
            a2coloc  = coeffscoloc[:,2]
            a3coloc  = coeffscoloc[:,3]
            a4coloc  = coeffscoloc[:,4]
        
        else:
            raise ValueError('Inspiral phase version not implemented.')
            
            
        sigma1 = (-5./3.) * a0coloc
        sigma2 = (-5./4.) * a1coloc
        sigma3 = (-5./5.) * a2coloc
        sigma4 = (-5./6.) * a3coloc
        sigma5 = (-5./7.) * a4coloc
        
        # TaylorF2 PN Coefficients
        # Newtonian
        phi0   = 1.
        # .5 PN
        phi1   = 0.
        # 1 PN
        phi2   = (3715./756. + (55.*eta)/9.) * (np.pi**(2./3.))
        # 1.5 PN, Non-Spinning and Spin-Orbit
        phi3   = -16.0*np.pi*np.pi + ((113.*(chi1 + chi2 + chi1*delta - chi2*delta) - 76.*(chi1 + chi2)*eta)/6.) * np.pi
        # 2 PN, Non-Spinning and Spin-Spin
        phi4   = (15293365./508032. + (27145.*eta)/504. + (3085*eta2)/72.)*(np.pi**(4./3.)) + ((-5.*(81.*chi12*(1. + delta - 2.*eta) + 316.*chi1dotchi2*eta - 81.*chi22*(-1. + delta + 2.*eta)))/16.)*(np.pi**(4./3.))
        # 2.5 PN, Non-Spinning and Spin-Orbit
        phi5L  = (((5.*(46374. - 6552.*eta)*np.pi)/4536.) + ((-732985.*(chi1 + chi2 + chi1*delta - chi2*delta) - 560.*(-1213.*(chi1 + chi2) + 63.*(chi1 - chi2)*delta)*eta + 85680.*(chi1 + chi2)*eta2)/4536.)) * (np.pi**(5./3.))
        phi5   = 0.
        # 3 PN, Non-Spinning, Spin-Orbit, Spin-Spin
        phi6   = (11583231236531./4.69421568e9 - (5.*eta*(3147553127. + 588.*eta*(-45633. + 102260.*eta)))/3.048192e6 - (6848.*np.euler_gamma)/21. -(640.*np.pi*np.pi)/3. + (2255.*eta*np.pi*np.pi)/12. - (13696*np.log(2.))/21. - (6848.*np.log(np.pi))/63.)*np.pi*np.pi + ((5.*(227.*(chi1 + chi2 + chi1*delta - chi2*delta) - 156.*(chi1 + chi2)*eta)*np.pi)/3.)*np.pi*np.pi + ((5.*(20.*chi1dotchi2*eta*(11763. + 12488.*eta) + 7.*chi22*(-15103.*(-1. + delta) + 2.*(-21683. + 6580.*delta)*eta - 9808.*eta2) - 7.*chi12*(-15103.*(1. + delta) + 2.*(21683. + 6580.*delta)*eta + 9808.*eta2)))/4032.)*np.pi*np.pi
        # 3 PN log, Non-Spinning, Spin-Orbit and Spin-Spin
        phi6L  = (-6848/63.) * np.pi*np.pi
        # 3.5 PN, Non-Spinning, Spin-Orbit, Spin-Spin and Cubic-in-Spin
        phi7   = ((5.*(15419335. + 168.*(75703. - 29618.*eta)*eta)*np.pi)/254016.) * (np.pi**(7./3.)) + ((5.*(-5030016755.*(chi1 + chi2 + chi1*delta - chi2*delta) + 4.*(2113331119.*(chi1 + chi2) + 675484362.*(chi1 - chi2)*delta)*eta - 1008.*(208433.*(chi1 + chi2) + 25011.*(chi1 - chi2)*delta)*eta2 + 90514368.*(chi1 + chi2)*eta2*eta))/6.096384e6) * (np.pi**(7./3.)) + (-5.*(57.*chi12*(1. + delta - 2.*eta) + 220.*chi1dotchi2*eta - 57.*chi22*(-1. + delta + 2.*eta))*np.pi)*(np.pi**(7./3.)) + ((14585.*(-(chi22*chi2*(-1. + delta)) + chi12*chi1*(1. + delta)) - 5.*(chi22*chi2*(8819. - 2985.*delta) + 8439.*chi1*chi22*(-1. + delta) - 8439.*chi12*chi2*(1. + delta) + chi12*chi1*(8819. + 2985.*delta))*eta + 40.*(chi1 + chi2)*(17.*chi12 - 14.*chi1dotchi2 + 17.*chi22)*eta2)/48.) * (np.pi**(7./3.))
        # 4 PN, Non-Spinning and Spin-Orbit
        phi8   = ((-5.*(1263141.*(chi1 + chi2 + chi1*delta - chi2*delta) - 2.*(794075.*(chi1 + chi2) + 178533.*(chi1 - chi2)*delta)*eta + 94344.*(chi1 + chi2)*eta2)*np.pi*(-1. + np.log(np.pi)))/9072. ) * (np.pi**(8./3.))
        # 4 PN log, Non-Spinning and Spin-Orbit
        phi8L  = ((-5.*(1263141.*(chi1 + chi2 + chi1*delta - chi2*delta) - 2.*(794075.*(chi1 + chi2) + 178533.*(chi1 - chi2)*delta)*eta + 94344.*(chi1 + chi2)*eta2)*np.pi)/9072.) * (np.pi**(8./3.))
        
        if self.InsPhaseVersion in [114, 115]:
            # 3.5PN, Leading Order Spin-Spin Tail Term
            phi7   = phi7 + ((5.*(65.*chi12*(1. + delta - 2.*eta) + 252.*chi1dotchi2*eta - 65.*chi22*(-1. + delta + 2.*eta))*np.pi)/4.) * (np.pi**(7./3.))
            # 4.5PN, Tail Term
            phi9   = ((5.*(-256. + 451.*eta)*np.pi*np.pi*np.pi)/6. + (np.pi*(105344279473163. + 700.*eta*(-298583452147. + 96.*eta*(99645337. + 14453257.*eta)) - 12246091038720.*np.euler_gamma - 24492182077440*np.log(2.)))/1.877686272e10 - (13696*np.pi*np.log(np.pi))/63. ) * np.pi*np.pi*np.pi
            # 4.5PN, Log Term
            phi9L  = ((-13696*np.pi)/63.) * np.pi*np.pi*np.pi
        else:
            phi9   = 0.
            phi9L  = 0.
        
        # Normalized Phase Derivatives
        dphi0  = 1.0
        dphi1  = 0.0
        dphi2  = (743./252. + (11.*eta)/3. )*(np.pi**(2./3.))
        dphi3  = ((chi2*(113. - 113.*delta - 76.*eta) + chi1*(113.*(1. + delta) - 76.*eta) - 96.*np.pi)/15.) * np.pi
        dphi4  = (3058673/508032. - (81*chi12*(1 + delta))/16. - (79*chi1dotchi2*eta)/4. + (81*chi22*(-1 + delta + 2*eta))/16. + (eta*(5429 + 5103*chi12 + 4319*eta))/504. ) * (np.pi**(4./3.))
        dphi5  = ( (-146597*chi2*delta + 146597*(chi1 + chi2 + chi1*delta) + 112*(chi1*(-1213 + 63*delta) - chi2*(1213 + 63*delta))*eta - 17136*(chi1 + chi2)*eta2 + 6*(-7729 + 1092*eta)*np.pi)/1512. ) * (np.pi**(5./3.))
        dphi6  = ( (-10052469856691 + 24236159077900*eta)/2.34710784e10 + (6848*np.euler_gamma)/105. + (-951489*chi12*(1 + delta) - 180*chi1dotchi2*eta*(11763 + 12488*eta) + 63*chi22*(15103*(-1 + delta) + 2*(21683 - 6580*delta)*eta + 9808*eta2) + 7*eta*(18*chi12*(21683 + 6580*delta + 4904*eta) + eta*(-45633 + 102260*eta)) - 12096*(chi2*(227 - 227*delta - 156*eta) + chi1*(227*(1 + delta) - 156*eta))*np.pi - 3024*(-512 + 451*eta)*np.pi*np.pi )/36288. + (13696*np.log(2.))/105. + (6848*np.log(np.pi))/315.0   ) * np.pi*np.pi
        dphi6L = (  6848 / 315. ) * np.pi*np.pi
        dphi7  = (-(chi12*chi2*eta*(2813*(1 + delta) + 8*eta))/8. + (chi12*chi1*(-2917*(1 + delta) + (8819 + 2985*delta)*eta - 136*eta2))/24. + (chi2*(-5030016755*(-1 + delta) + 4*(-2113331119 + 675484362*delta)*eta + 1008*(208433 - 25011*delta)*eta2 - 90514368*eta2*eta))/3.048192e6 - (chi22*chi2*(2917 + eta*(-8819 + 136*eta) + delta*(-2917 + 2985*eta)))/24. + (chi1*(5030016755 - 8453324476*eta + 1008*eta*((208433 - 89796*eta)*eta - 378*chi22*(2813 + 8*eta)) + delta*(5030016755 + 504*eta*(-5360987 + 2126628*chi22 + 50022*eta))))/3.048192e6 + (-15419335/127008. + 114*chi12*(1 + delta - 2*eta) - (75703*eta)/756. + 440*chi1dotchi2*eta + (14809*eta2)/378. - 114*chi22*(-1 + delta + 2*eta))*np.pi ) * (np.pi**(7./3.))
        dphi8  = ( ((chi1*(1263141 - 1588150*eta + 94344*eta2 - 9*delta*(-140349 + 39674*eta)) + chi2*(1263141 - 1588150*eta + 94344*eta2 + 9*delta*(-140349 + 39674*eta)))*np.pi)/3024. ) * (np.pi**(8./3.))* np.log(np.pi)
        dphi8L  = ( ((chi1*(1263141 - 1588150*eta + 94344*eta2 - 9*delta*(-140349 + 39674*eta)) + chi2*(1263141 - 1588150*eta + 94344*eta2 + 9*delta*(-140349 + 39674*eta)))*np.pi)/3024. ) * (np.pi**(8./3.))


        
        if self.InsPhaseVersion in [114, 115]:
            # 3.5PN, Leading Order Spin-Spin Tail Term
            dphi7   = dphi7 + (((-65*chi12*(1 + delta - 2*eta) - 252*chi1dotchi2*eta + 65*chi22*(-1 + delta + 2*eta))* np.pi)/2. ) * (np.pi**(7./3.))
            # 4.5PN, Tail Term
            dphi9 = ( (512/3. - (902*eta)/3.)*np.pi*np.pi*np.pi + np.pi * (-102282756713483/2.34710784e10 + (298583452147*eta)/3.3530112e7 - (9058667*eta2)/31752. - (2064751*eta2*eta)/49896. + (54784*np.euler_gamma)/105. + (109568*np.log(2.))/105. + (54784*np.log(np.pi))/315.) ) * np.pi*np.pi*np.pi
            # 4.5PN, Log Term
            dphi9L = ( (54784*np.pi)/315. ) * np.pi*np.pi*np.pi
        else:
            dphi9   = 0.
            dphi9L  = 0.
        
        # Calculate phase at fmatchIN
        # First the standard TaylorF2 part
        phaseIN = dphi0 + dphi1*(fPhaseMatchIN**(1./3.)) + dphi2*(fPhaseMatchIN**(2./3.)) + dphi3*fPhaseMatchIN + dphi4*(fPhaseMatchIN**(4./3.)) + dphi5*(fPhaseMatchIN**(5./3.)) + (dphi6 + dphi6L*np.log(fPhaseMatchIN))*fPhaseMatchIN*fPhaseMatchIN + dphi7*(fPhaseMatchIN**(7./3.)) + (dphi8 + dphi8L*np.log(fPhaseMatchIN))*(fPhaseMatchIN**(8./3.)) + (dphi9 + dphi9L*np.log(fPhaseMatchIN))*fPhaseMatchIN*fPhaseMatchIN*fPhaseMatchIN
        # Then the pseudo-PN
        phaseIN = phaseIN + a0coloc*(fPhaseMatchIN**(8./3.)) + a1coloc*fPhaseMatchIN*fPhaseMatchIN*fPhaseMatchIN + a2coloc*(fPhaseMatchIN**(10./3.)) + a3coloc*(fPhaseMatchIN**(11./3.)) + a4coloc*(fPhaseMatchIN**4)
        # Finally the overall phase
        phaseIN = phaseIN*(fPhaseMatchIN**(-8./3.))*dphase0
        
        # Intermediate phase collocation points:
        # Default is to use 5 collocation points
        
        deltax      = fPhaseMatchIM - fPhaseMatchIN
        xmin        = fPhaseMatchIN
        
        if self.IntPhaseVersion == 104:
            CollocationPointsPhaseInt0 = gpoints4[0]* deltax + xmin
            CollocationPointsPhaseInt1 = gpoints4[1]* deltax + xmin
            CollocationPointsPhaseInt2 = gpoints4[2]* deltax + xmin
            CollocationPointsPhaseInt3 = gpoints4[3]* deltax + xmin
            
            CollocationValuesPhaseInt0 = phaseIN
            CollocationValuesPhaseInt3 = phaseRD
            
            v2IMmRDv4 = (((eta*(-8.244230124407343 - 182.80239160435949*eta + 638.2046409916306*eta2 - 578.878727101827*eta2*eta))/(-0.004863669418916522 - 0.5095088831841608*eta + 1.*eta2)) + ((totchi*(0.1344136125169328 + 0.0751872427147183*totchi + eta2*(7.158823192173721 + 25.489598292111104*totchi - 7.982299108059922*totchi2) + eta*(-5.792368563647471 + 1.0190164430971749*totchi + 0.29150399620268874*totchi2) + 0.033627267594199865*totchi2 + eta2*eta*(17.426503081351534 - 90.69790378048448*totchi + 20.080325133405847*totchi2)))/(0.03449545664201546 - 0.027941977370442107*totchi + (0.005274757661661763 + 0.0455523144123269*eta - 0.3880379911692037*eta2 + 1.*eta2*eta)*totchi2)) + (160.2975913661124*dchi*delta*eta2))
            v3IMmRDv4 = (((0.3145740304678042 + 299.79825045000655*eta - 605.8886581267144*eta2 - 1302.0954503758007*eta2*eta)/(1. + 2.3903703875353255*eta - 19.03836730923657*eta2)) + ((totchi*(1.150925084789774 - 0.3072381261655531*totchi + eta2*eta2*(12160.22339193134 - 1459.725263347619*totchi - 9581.694749116636*totchi2) + eta2*(1240.3900459406875 - 289.48392062629966*totchi - 1218.1320231846412*totchi2) - 1.6191217310064605*totchi2 + eta*(-41.38762957457647 + 60.47477582076457*totchi2) + eta2*eta*(-7540.259952257055 + 1379.3429194626635*totchi + 6372.99271204178*totchi2)))/(-1.4325421225106187 + 1.*totchi)) + (dchi*delta*eta2*eta*(-444.797248674011 + 1448.47758082697*eta + 152.49290092435044*totchi)))
            v2IM      = (((-84.09400000000004 - 1782.8025405571802*eta + 5384.38721936653*eta2)/(1. + 28.515617312596103*eta + 12.404097877099353*eta2)) + ((totchi*(22.5665046165141 - 39.94907120140026*totchi + 4.668251961072*totchi2 + 12.648584361431245*totchi2*totchi + eta2*(-298.7528127869681 + 14.228745354543983*totchi + 398.1736953382448*totchi2 + 506.94924905801673*totchi2*totchi - 626.3693674479357*totchi2*totchi2) - 5.360554789680035*totchi2*totchi2 + eta*(152.07900889608595 - 121.70617732909962*totchi2 - 169.36311036967322*totchi2*totchi + 182.40064586992762*totchi2*totchi2)))/(-1.1571201220629341 + 1.*totchi)) + (dchi*delta*eta2*eta*(5357.695021063607 - 15642.019339339662*eta + 674.8698102366333*totchi)))
            
            CollocationValuesPhaseInt1 = 0.75*(v2IMmRDv4 + CollocationValuesPhaseRD3) + 0.25*v2IM
            CollocationValuesPhaseInt2 = v3IMmRDv4 + CollocationValuesPhaseRD3
            
            A0i = np.array([np.ones(CollocationPointsPhaseInt0.shape), fring/CollocationPointsPhaseInt0, (fring/CollocationPointsPhaseInt0)*(fring/CollocationPointsPhaseInt0), (fring/CollocationPointsPhaseInt0)**3])
            A1i = np.array([np.ones(CollocationPointsPhaseInt1.shape), fring/CollocationPointsPhaseInt1, (fring/CollocationPointsPhaseInt1)*(fring/CollocationPointsPhaseInt1), (fring/CollocationPointsPhaseInt1)**3])
            A2i = np.array([np.ones(CollocationPointsPhaseInt2.shape), fring/CollocationPointsPhaseInt2, (fring/CollocationPointsPhaseInt2)*(fring/CollocationPointsPhaseInt2), (fring/CollocationPointsPhaseInt2)**3])
            A3i = np.array([np.ones(CollocationPointsPhaseInt3.shape), fring/CollocationPointsPhaseInt3, (fring/CollocationPointsPhaseInt3)*(fring/CollocationPointsPhaseInt3), (fring/CollocationPointsPhaseInt3)**3])
            
            Acoloc = np.array([A0i, A1i, A2i, A3i]).transpose(2,0,1)
            bcoloc = np.array([CollocationValuesPhaseInt0 - ((4. * cLcoloc) / ((2.*fdamp)*(2.*fdamp) + (CollocationPointsPhaseInt0 - fring)*(CollocationPointsPhaseInt0 - fring))), CollocationValuesPhaseInt1 - ((4. * cLcoloc) / ((2.*fdamp)*(2.0*fdamp) + (CollocationPointsPhaseInt1 - fring)*(CollocationPointsPhaseInt1 - fring))), CollocationValuesPhaseInt2 - ((4. * cLcoloc) / ((2.*fdamp)*(2.0*fdamp) + (CollocationPointsPhaseInt2 - fring)*(CollocationPointsPhaseInt2 - fring))), CollocationValuesPhaseInt3 - ((4. * cLcoloc) / ((2.*fdamp)*(2.0*fdamp) + (CollocationPointsPhaseInt3 - fring)*(CollocationPointsPhaseInt3 - fring)))]).T
            
            coeffscoloc = np.linalg.solve(Acoloc, bcoloc)

            b0coloc = coeffscoloc[:,0]
            b1coloc = coeffscoloc[:,1] * fring
            b2coloc = coeffscoloc[:,2] * fring * fring
            b3coloc = np.zeros_like(b2coloc)
            b4coloc = coeffscoloc[:,3] * fring * fring * fring * fring
            
            
        elif self.IntPhaseVersion == 105:
            CollocationPointsPhaseInt0 = gpoints5[0]* deltax + xmin
            CollocationPointsPhaseInt1 = gpoints5[1]* deltax + xmin
            CollocationPointsPhaseInt2 = gpoints5[2]* deltax + xmin
            CollocationPointsPhaseInt3 = gpoints5[3]* deltax + xmin
            CollocationPointsPhaseInt4 = gpoints5[4]* deltax + xmin

            CollocationValuesPhaseInt0 = phaseIN
            CollocationValuesPhaseInt4 = phaseRD

            v2IMmRDv4 = (((eta*(0.9951733419499662 + 101.21991715215253*eta + 632.4731389009143*eta2))/(0.00016803066316882238 + 0.11412314719189287*eta + 1.8413983770369362*eta2 + 1.*eta2*eta)) + ((totchi*(18.694178521101332 + 16.89845522539974*totchi + 4941.31613710257*eta2*totchi + eta*(-697.6773920613674 - 147.53381808989846*totchi2) + 0.3612417066833153*totchi2 + eta2*eta*(3531.552143264721 - 14302.70838220423*totchi + 178.85850322465944*totchi2)))/(2.965640445745779 - 2.7706595614504725*totchi + 1.*totchi2)) + (dchi*delta*eta2*(356.74395864902294 + 1693.326644293169*eta2*totchi)))
            v3IMmRDv4 = (((eta*(-5.126358906504587 - 227.46830225846668*eta + 688.3609087244353*eta2 - 751.4184178636324*eta2*eta))/(-0.004551938711031158 - 0.7811680872741462*eta + 1.*eta2)) + ((totchi*(0.1549280856660919 - 0.9539250460041732*totchi - 539.4071941841604*eta2*totchi + eta*(73.79645135116367 - 8.13494176717772*totchi2) - 2.84311102369862*totchi2 + eta2*eta*(-936.3740515136005 + 1862.9097047992134*totchi + 224.77581754671272*totchi2)))/(-1.5308507364054487 + 1.*totchi)) + (2993.3598520496153*dchi*delta*eta2*eta2*eta2))
            v2IM      = (((-82.54500000000004 - 5.58197349185435e6*eta - 3.5225742421184325e8*eta2 + 1.4667258334378073e9*eta2*eta)/(1. + 66757.12830903867*eta + 5.385164380400193e6*eta2 + 2.5176585751772933e6*eta2*eta)) + ((totchi*(19.416719811164853 - 36.066611959079935*totchi - 0.8612656616290079*totchi2 + eta2*(170.97203068800542 - 107.41099349364234*totchi - 647.8103976942541*totchi2*totchi) + 5.95010003393006*totchi2*totchi + eta2*eta*(-1365.1499998427248 + 1152.425940764218*totchi + 415.7134909564443*totchi2 + 1897.5444343138167*totchi2*totchi - 866.283566780576*totchi2*totchi2) + 4.984750041013893*totchi2*totchi2 + eta*(207.69898051583655 - 132.88417400679026*totchi - 17.671713040498304*totchi2 + 29.071788188638315*totchi2*totchi + 37.462217031512786*totchi2*totchi2)))/(-1.1492259468169692 + 1.*totchi)) + (dchi*delta*eta2*eta*(7343.130973149263 - 20486.813161100774*eta + 515.9898508588834*totchi)))

            CollocationValuesPhaseInt1 = 0.75*(v2IMmRDv4 + CollocationValuesPhaseRD3) + 0.25*v2IM
            CollocationValuesPhaseInt2 = v3IMmRDv4 + CollocationValuesPhaseRD3
            CollocationValuesPhaseInt3 = (((0.4248820426833804 - 906.746595921514*eta - 282820.39946006844*eta2 - 967049.2793750163*eta2*eta + 670077.5414916876*eta2*eta2)/(1. + 1670.9440812294847*eta + 19783.077247023448*eta2)) + ((totchi*(0.22814271667259703 + 1.1366593671801855*totchi + eta2*eta*(3499.432393555856 - 877.8811492839261*totchi - 4974.189172654984*totchi2) + eta*(12.840649528989287 - 61.17248283184154*totchi2) + 0.4818323187946999*totchi2 + eta2*(-711.8532052499075 + 269.9234918621958*totchi + 941.6974723887743*totchi2) + eta2*eta2*(-4939.642457025497 - 227.7672020783411*totchi + 8745.201037897836*totchi2)))/(-1.2442293719740283 + 1.*totchi)) + (dchi*delta*(-514.8494071830514 + 1493.3851099678195*eta)*eta2*eta))
            CollocationValuesPhaseInt3 = CollocationValuesPhaseInt3 + CollocationValuesPhaseInt2

            A0i = np.array([np.ones(CollocationPointsPhaseInt0.shape), fring/CollocationPointsPhaseInt0, (fring/CollocationPointsPhaseInt0)*(fring/CollocationPointsPhaseInt0), (fring/CollocationPointsPhaseInt0)**3, (fring/CollocationPointsPhaseInt0)**4])
            A1i = np.array([np.ones(CollocationPointsPhaseInt1.shape), fring/CollocationPointsPhaseInt1, (fring/CollocationPointsPhaseInt1)*(fring/CollocationPointsPhaseInt1), (fring/CollocationPointsPhaseInt1)**3, (fring/CollocationPointsPhaseInt1)**4])
            A2i = np.array([np.ones(CollocationPointsPhaseInt2.shape), fring/CollocationPointsPhaseInt2, (fring/CollocationPointsPhaseInt2)*(fring/CollocationPointsPhaseInt2), (fring/CollocationPointsPhaseInt2)**3, (fring/CollocationPointsPhaseInt2)**4])
            A3i = np.array([np.ones(CollocationPointsPhaseInt3.shape), fring/CollocationPointsPhaseInt3, (fring/CollocationPointsPhaseInt3)*(fring/CollocationPointsPhaseInt3), (fring/CollocationPointsPhaseInt3)**3, (fring/CollocationPointsPhaseInt3)**4])
            A4i = np.array([np.ones(CollocationPointsPhaseInt4.shape), fring/CollocationPointsPhaseInt4, (fring/CollocationPointsPhaseInt4)*(fring/CollocationPointsPhaseInt4), (fring/CollocationPointsPhaseInt4)**3, (fring/CollocationPointsPhaseInt4)**4])

            Acoloc = np.array([A0i, A1i, A2i, A3i, A4i]).transpose(2,0,1)#.T
            bcoloc = np.array([CollocationValuesPhaseInt0 - ((4. * cLcoloc) / ((2.*fdamp)*(2.*fdamp) + (CollocationPointsPhaseInt0 - fring)*(CollocationPointsPhaseInt0 - fring))), CollocationValuesPhaseInt1 - ((4. * cLcoloc) / ((2.*fdamp)*(2.0*fdamp) + (CollocationPointsPhaseInt1 - fring)*(CollocationPointsPhaseInt1 - fring))), CollocationValuesPhaseInt2 - ((4. * cLcoloc) / ((2.*fdamp)*(2.0*fdamp) + (CollocationPointsPhaseInt2 - fring)*(CollocationPointsPhaseInt2 - fring))), CollocationValuesPhaseInt3 - ((4. * cLcoloc) / ((2.*fdamp)*(2.0*fdamp) + (CollocationPointsPhaseInt3 - fring)*(CollocationPointsPhaseInt3 - fring))), CollocationValuesPhaseInt4 - ((4. * cLcoloc) / ((2.*fdamp)*(2.0*fdamp) + (CollocationPointsPhaseInt4 - fring)*(CollocationPointsPhaseInt4 - fring)))]).T

            coeffscoloc = np.linalg.solve(Acoloc, bcoloc)

            b0coloc = coeffscoloc[:,0]
            b1coloc = coeffscoloc[:,1] * fring
            b2coloc = coeffscoloc[:,2] * fring * fring
            b3coloc = coeffscoloc[:,3] * fring * fring * fring
            b4coloc = coeffscoloc[:,4] * fring * fring * fring * fring
        
        else:
            raise ValueError('Intermediate phase version not implemented.')

        c4ov3   = c4coloc / 3.
        cLovfda = cLcoloc / fdamp
            
        # Intermediate phase derivative at fPhaseMatchIN
        DPhiInt = b0coloc + b1coloc/fPhaseMatchIN + b2coloc/(fPhaseMatchIN*fPhaseMatchIN) + b3coloc/(fPhaseMatchIN*fPhaseMatchIN*fPhaseMatchIN) + b4coloc/(fPhaseMatchIN*fPhaseMatchIN*fPhaseMatchIN*fPhaseMatchIN) + (4.*cLcoloc) / ((4.*fdamp*fdamp) + (fPhaseMatchIN - fring)*(fPhaseMatchIN - fring))
        
        C2Int = phaseIN - DPhiInt
        # Inspiral phase at fPhaseMatchIN
        phiIN = phi0 + phi1*(fPhaseMatchIN**(1./3.)) + phi2*(fPhaseMatchIN**(2./3.)) + phi3*fPhaseMatchIN + phi4*(fPhaseMatchIN**(4./3.)) + (phi5 + phi5L*np.log(fPhaseMatchIN))*(fPhaseMatchIN**(5./3.)) + (phi6 + phi6L*np.log(fPhaseMatchIN))*fPhaseMatchIN*fPhaseMatchIN + phi7*(fPhaseMatchIN**(7./3.)) + (phi8 + phi8L*np.log(fPhaseMatchIN))*(fPhaseMatchIN**(8./3.)) + (phi9  + phi9L*np.log(fPhaseMatchIN))*fPhaseMatchIN*fPhaseMatchIN*fPhaseMatchIN
        # add the pseudo-PN Coefficients
        phiIN = phiIN + sigma1*(fPhaseMatchIN**(8./3.)) + sigma2*(fPhaseMatchIN*fPhaseMatchIN*fPhaseMatchIN) + sigma3*(fPhaseMatchIN**(10./3.)) + sigma4*(fPhaseMatchIN**(11./3.)) + sigma5*(fPhaseMatchIN**4)
        # finally the normalisation
        phiIN = phiIN * phiNorm*(fPhaseMatchIN**(-5./3.))
        
        # Intermediate phase at fPhaseMatchIN
        
        phiIM = b0coloc*fPhaseMatchIN + b1coloc*np.log(fPhaseMatchIN) - b2coloc/fPhaseMatchIN - b3coloc/(fPhaseMatchIN*fPhaseMatchIN)/2. - (b4coloc/(fPhaseMatchIN*fPhaseMatchIN*fPhaseMatchIN)/3.) + (2. * cLcoloc * np.arctan((fPhaseMatchIN - fring) / (2. * fdamp)))/fdamp
         
        C1Int = phiIN - phiIM - (C2Int * fPhaseMatchIN)
        
        # Intermediate phase at fPhaseMatchIM + connections
        phiIMC   = b0coloc*fPhaseMatchIM + b1coloc*np.log(fPhaseMatchIM) - b2coloc/fPhaseMatchIM - b3coloc/(fPhaseMatchIM*fPhaseMatchIM)/2. - (b4coloc/(fPhaseMatchIM*fPhaseMatchIM*fPhaseMatchIM)/3.) + (2. * cLcoloc * np.arctan((fPhaseMatchIM - fring) / (2. * fdamp)))/fdamp + C1Int + C2Int*fPhaseMatchIM
        # Ringdown phase at fPhaseMatchIM
        phiRD    = c0coloc*fPhaseMatchIM + 1.5*c1coloc*(fPhaseMatchIM**(2./3.)) - c2coloc/fPhaseMatchIM - c4ov3/(fPhaseMatchIM*fPhaseMatchIM*fPhaseMatchIM) + (cLovfda * np.arctan((fPhaseMatchIM - fring)/fdamp))
        # Intermediate phase derivative at fPhaseMatchIM + connection
        DPhiIntC = b0coloc + b1coloc/fPhaseMatchIM + b2coloc/(fPhaseMatchIM*fPhaseMatchIM) + b3coloc/(fPhaseMatchIM*fPhaseMatchIM*fPhaseMatchIM) + b4coloc/(fPhaseMatchIM*fPhaseMatchIM*fPhaseMatchIM*fPhaseMatchIM) + (4.*cLcoloc) / ((4.*fdamp*fdamp) + (fPhaseMatchIM - fring)*(fPhaseMatchIM - fring)) + C2Int
        # Ringdown phase derivative at fPhaseMatchIM
        DPhiRD   = (c0coloc + c1coloc*(fPhaseMatchIM**(-1./3.)) + c2coloc/(fPhaseMatchIM*fPhaseMatchIM) + c4coloc/(fPhaseMatchIM*fPhaseMatchIM*fPhaseMatchIM*fPhaseMatchIM) + (cLcoloc / (fdamp*fdamp + (fPhaseMatchIM - fring)*(fPhaseMatchIM - fring))))
        
        C2MRD = DPhiIntC - DPhiRD
        C1MRD = phiIMC - phiRD - C2MRD*fPhaseMatchIM
        
        def completePhaseDer(infreqs):
            return np.where(infreqs <= fPhaseMatchIN, (infreqs**(-8./3.))*dphase0*(dphi0 + dphi1*(infreqs**(1./3.)) + dphi2*(infreqs**(2./3.)) + dphi3*infreqs + dphi4*(infreqs**(4./3.)) + dphi5*(infreqs**(5./3.)) + (dphi6 + dphi6L*np.log(infreqs))*infreqs*infreqs + dphi7*(infreqs**(7./3.)) + (dphi8 + dphi8L*np.log(infreqs))*(infreqs**(8./3.)) + (dphi9  + dphi9L*np.log(infreqs))*infreqs*infreqs*infreqs + a0coloc*(infreqs**(8./3.)) + a1coloc*infreqs*infreqs*infreqs + a2coloc*(infreqs**(10./3.)) + a3coloc*(infreqs**(11./3.)) + a4coloc*(infreqs**4)), np.where(infreqs <= fPhaseMatchIM, b0coloc + b1coloc/infreqs + b2coloc/(infreqs*infreqs) + b3coloc/(infreqs*infreqs*infreqs) + b4coloc/(infreqs*infreqs*infreqs*infreqs) + (4.*cLcoloc) / ((4.*fdamp*fdamp) + (infreqs - fring)*(infreqs - fring)) + C2Int, (c0coloc + c1coloc*(infreqs**(-1./3.)) + c2coloc/(infreqs*infreqs) + c4coloc/(infreqs*infreqs*infreqs*infreqs) + (cLcoloc / (fdamp*fdamp + (infreqs - fring)*(infreqs - fring)))) + C2MRD))
        
        def completePhase(infreqs):
            return np.where(infreqs <= fPhaseMatchIN, phiNorm*(infreqs**(-5./3.))*(phi0 + phi1*(infreqs**(1./3.)) + phi2*(infreqs**(2./3.)) + phi3*infreqs + phi4*(infreqs**(4./3.)) + (phi5 + phi5L*np.log(infreqs))*(infreqs**(5./3.)) + (phi6 + phi6L*np.log(infreqs))*infreqs*infreqs + phi7*(infreqs**(7./3.)) + (phi8 + phi8L*np.log(infreqs))*(infreqs**(8./3.)) + (phi9  + phi9L*np.log(infreqs))*infreqs*infreqs*infreqs + sigma1*(infreqs**(8./3.)) + sigma2*(infreqs*infreqs*infreqs) + sigma3*(infreqs**(10./3.)) + sigma4*(infreqs**(11./3.)) + sigma5*(infreqs**4)), np.where(infreqs <= fPhaseMatchIM, b0coloc*infreqs + b1coloc*np.log(infreqs) - b2coloc/infreqs - b3coloc/(infreqs*infreqs)/2. - (b4coloc/(infreqs*infreqs*infreqs)/3.) + (2. * cLcoloc * np.arctan((infreqs - fring) / (2. * fdamp)))/fdamp + C1Int + C2Int*infreqs, np.where(infreqs < self.fcutPar, (c0coloc*infreqs + 1.5*c1coloc*(infreqs**(2./3.)) - c2coloc/infreqs - c4ov3/(infreqs*infreqs*infreqs) + (cLovfda * np.arctan((infreqs - fring)/fdamp))) + C1MRD + C2MRD*infreqs, 0.)))
        
        # Linear time and phase shifts so that model peaks near t ~ 0
        lina = 0.
        linb         = ((3155.1635543201924 + 1257.9949740608242*eta - 32243.28428870599*eta2 + 347213.65466875216*eta2*eta - 1.9223851649491738e6*eta2*eta2 + 5.3035911346921865e6*eta2*eta2*eta - 5.789128656876938e6*eta2*eta2*eta2) + ((-24.181508118588667 + 115.49264174560281*eta - 380.19778216022763*eta2)*totchi + (24.72585609641552 - 328.3762360751952*eta + 725.6024119989094*eta2)*totchi2 + (23.404604124552 - 646.3410199799737*eta + 1941.8836639529036*eta2)*totchi2*totchi + (-12.814828278938885 - 325.92980012408367*eta + 1320.102640190539*eta2)*totchi2*totchi2) + (-148.17317525117338*dchi*delta*eta2))
        dphi22Ref    = etaInv * completePhaseDer(fring-fdamp)
        psi4tostrain = ((13.39320482758057 - 175.42481512989315*eta + 2097.425116152503*eta2 - 9862.84178637907*eta2*eta + 16026.897939722587*eta2*eta2) + ((4.7895602776763 - 163.04871764530466*eta + 609.5575850476959*eta2)*totchi + (1.3934428041390161 - 97.51812681228478*eta + 376.9200932531847*eta2)*totchi2 + (15.649521097877374 + 137.33317057388916*eta - 755.9566456906406*eta2)*totchi2*totchi + (13.097315867845788 + 149.30405703643288*eta - 764.5242164872267*eta2)*totchi2*totchi2) + (105.37711654943146*dchi*Seta*eta2))
        linb         = linb - dphi22Ref - 2.*np.pi*(500.+psi4tostrain)
        
        phifRef = -(etaInv*completePhase(fRef) + linb*fRef + lina) + np.pi/4. + np.pi
        phis    = etaInv*completePhase(fgrid) + np.where(fgrid <= self.fcutPar, linb*fgrid + lina + phifRef, 0.)
        
        return phis
    
    def Ampl(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        etaInv = 1./eta
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2 = chi1*chi2
        Seta = np.sqrt(1.0 - 4.0*eta)
        SetaPlus1 = 1.0 + Seta
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        q = 0.5*(1.0 + Seta - 2.0*eta)/eta
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        m1ByMSq = m1ByM*m1ByM
        m2ByMSq = m2ByM*m2ByM
        # PN symmetry coefficient
        delta = np.fabs((m1ByM - m2ByM) / (m1ByM + m2ByM))
        totchi = ((m1ByMSq * chi1 + m2ByMSq * chi2) / (m1ByMSq + m2ByMSq))
        totchi2 = totchi*totchi
        dchi = chi1-chi2
        # Normalised PN reduced spin parameter
        chi_eff = (m1ByM*chi1 + m2ByM*chi2)
        chiPN   = (chi_eff - (38./113.)*eta*(chi1 + chi2)) / (1. - (76.*eta/113.))
        chiPN2  = chiPN*chiPN
        # We work in dimensionless frequency M*f, not f
        fgrid = M*utils.GMsun_over_c3*f
            
        amp0    = 2. * np.sqrt(5./(64.*np.pi)) * M * utils.GMsun_over_c2_Gpc * M * utils.GMsun_over_c3 / kwargs['dL']
        ampNorm = np.sqrt(2.*eta/3.) * (np.pi**(-1./6.))
        
        dphase0      = 5. / (128. * (np.pi**(5./3.)))
        
        gpoints4     = [0., 1./4., 3./4., 1.]
        gpoints5     = [0., 1./2. - 1./(2.*np.sqrt(2.)), 1./2., 1./2 + 1./(2.*np.sqrt(2.)), 1.]
        
        # Compute final spin and radiated energy
        aeff   = (((3.4641016151377544*eta + 20.0830030082033*eta2 - 12.333573402277912*eta2*eta)/(1 + 7.2388440419467335*eta)) + ((m1ByMSq + m2ByMSq)*totchi + ((-0.8561951310209386*eta - 0.09939065676370885*eta2 + 1.668810429851045*eta2*eta)*totchi + (0.5881660363307388*eta - 2.149269067519131*eta2 + 3.4768263932898678*eta2*eta)*totchi2 + (0.142443244743048*eta - 0.9598353840147513*eta2 + 1.9595643107593743*eta2*eta)*totchi2*totchi) / (1 + (-0.9142232693081653 + 2.3191363426522633*eta - 9.710576749140989*eta2*eta)*totchi)) + (0.3223660562764661*dchi*Seta*(1 + 9.332575956437443*eta)*eta2 - 0.059808322561702126*dchi*dchi*eta2*eta + 2.3170397514509933*dchi*Seta*(1 - 3.2624649875884852*eta)*eta2*eta*totchi))
        Erad   = ((((0.057190958417936644*eta + 0.5609904135313374*eta2 - 0.84667563764404*eta2*eta + 3.145145224278187*eta2*eta2)*(1. + (-0.13084389181783257 - 1.1387311580238488*eta + 5.49074464410971*eta2)*totchi + (-0.17762802148331427 + 2.176667900182948*eta2)*totchi2 + (-0.6320191645391563 + 4.952698546796005*eta - 10.023747993978121*eta2)*totchi*totchi2)) / (1. + (-0.9919475346968611 + 0.367620218664352*eta + 4.274567337924067*eta2)*totchi)) + (- 0.09803730445895877*dchi*Seta*(1. - 3.2283713377939134*eta)*eta2 + 0.01118530335431078*dchi*dchi*eta2*eta - 0.01978238971523653*dchi*Seta*(1. - 4.91667749015812*eta)*eta*totchi))
        # Compute ringdown and damping frequencies from fits
        fring = ((0.05947169566573468 - 0.14989771215394762*aeff + 0.09535606290986028*aeff*aeff + 0.02260924869042963*aeff*aeff*aeff - 0.02501704155363241*aeff*aeff*aeff*aeff - 0.005852438240997211*(aeff**5) + 0.0027489038393367993*(aeff**6) + 0.0005821983163192694*(aeff**7))/(1 - 2.8570126619966296*aeff + 2.373335413978394*aeff*aeff - 0.6036964688511505*aeff*aeff*aeff*aeff + 0.0873798215084077*(aeff**6)))/(1. - Erad)
        fdamp = ((0.014158792290965177 - 0.036989395871554566*aeff + 0.026822526296575368*aeff*aeff + 0.0008490933750566702*aeff*aeff*aeff - 0.004843996907020524*aeff*aeff*aeff*aeff - 0.00014745235759327472*(aeff**5) + 0.0001504546201236794*(aeff**6))/(1 - 2.5900842798681376*aeff + 1.8952576220623967*aeff*aeff - 0.31416610693042507*aeff*aeff*aeff*aeff + 0.009002719412204133*(aeff**6)))/(1. - Erad)
        #fring = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fring)/(1. - Erad)
        #fdamp = np.interp(aeff, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        
        # Fitting function for hybrid minimum energy circular orbit (MECO) function and computation of ISCO frequency
        Z1tmp = 1. + np.cbrt((1. - aeff*aeff) ) * (np.cbrt(1. + aeff) + np.cbrt(1. - aeff))
        Z1tmp = np.where(Z1tmp>3., 3., Z1tmp)
        Z2tmp = np.sqrt(3.*aeff*aeff + Z1tmp*Z1tmp)
        fISCO  = (1. / ((3. + Z2tmp - np.sign(aeff)*np.sqrt((3. - Z1tmp) * (3. + Z1tmp + 2.*Z2tmp)))**(3./2.) + aeff))/np.pi
        
        fMECO    = (((0.018744340279608845 + 0.0077903147004616865*eta + 0.003940354686136861*eta2 - 0.00006693930988501673*eta2*eta)/(1. - 0.10423384680638834*eta)) + ((chiPN*(0.00027180386951683135 - 0.00002585252361022052*chiPN + eta2*eta2*(-0.0006807631931297156 + 0.022386313074011715*chiPN - 0.0230825153005985*chiPN2) + eta2*(0.00036556167661117023 - 0.000010021140796150737*chiPN - 0.00038216081981505285*chiPN2) + eta*(0.00024422562796266645 - 0.00001049013062611254*chiPN - 0.00035182990586857726*chiPN2) + eta2*eta*(-0.0005418851224505745 + 0.000030679548774047616*chiPN + 4.038390455349854e-6*chiPN2) - 0.00007547517256664526*chiPN2))/(0.026666543809890402 + (-0.014590539285641243 - 0.012429476486138982*eta + 1.4861197211952053*eta2*eta2 + 0.025066696514373803*eta2 + 0.005146809717492324*eta2*eta)*chiPN + (-0.0058684526275074025 - 0.02876774751921441*eta - 2.551566872093786*eta2*eta2 - 0.019641378027236502*eta2 - 0.001956646166089053*eta2*eta)*chiPN2 + (0.003507640638496499 + 0.014176504653145768*eta + 1.*eta2*eta2 + 0.012622225233586283*eta2 - 0.00767768214056772*eta2*eta)*chiPN2*chiPN)) + (dchi*dchi*(0.00034375176678815234 + 0.000016343732281057392*eta)*eta2 + dchi*Seta*eta*(0.08064665214195679*eta2 + eta*(-0.028476219509487793 - 0.005746537021035632*chiPN) - 0.0011713735642446144*chiPN)))
        
        del Z1tmp, Z2tmp
    
        gamma2        = (((0.8312293675316895 + 7.480371544268765*eta - 18.256121237800397*eta2)/(1. + 10.915453595496611*eta - 30.578409433912874*eta2)) + ((totchi*(0.5869408584532747 + eta*(-0.1467158405070222 - 2.8489481072076472*totchi) + 0.031852563636196894*totchi + eta2*(0.25295441250444334 + 4.6849496672664594*totchi)))/(3.8775263105069953 - 3.41755361841226*totchi + 1.*totchi2)) + (-0.00548054788508203*dchi*delta*eta))
        gamma3        = (((1.3666000000000007 - 4.091333144596439*eta + 2.109081209912545*eta2 - 4.222259944408823*eta2*eta)/(1. - 2.7440263888207594*eta)) + ((0.07179105336478316 + eta2*(2.331724812782498 - 0.6330998412809531*totchi) + eta*(-0.8752427297525086 + 0.4168560229353532*totchi) - 0.05633734476062242*totchi)*totchi) + (0. * delta * dchi))
        fAmpRDMin     = np.where(gamma2 <= 1., np.fabs(fring + fdamp * gamma3 * (np.sqrt(1. - gamma2 * gamma2) - 1.0) / gamma2), np.fabs(fring + fdamp*(-1.)*gamma3/gamma2))
        v1RD          = (((0.03689164742964719 + 25.417967754401182*eta + 162.52904393600332*eta2)/(1. + 61.19874463331437*eta - 29.628854485544874*eta2)) + ((totchi*(-0.14352506969368556 + 0.026356911108320547*totchi + 0.19967405175523437*totchi2 - 0.05292913111731128*totchi2*totchi + eta2*eta*(-48.31945248941757 - 3.751501972663298*totchi + 81.9290740950083*totchi2 + 30.491948143930266*totchi2*totchi - 132.77982622925845*totchi2*totchi2) + eta*(-4.805034453745424 + 1.11147906765112*totchi + 6.176053843938542*totchi2 - 0.2874540719094058*totchi2*totchi - 8.990840289951514*totchi2*totchi2) - 0.18147275151697131*totchi2*totchi2 + eta2*(27.675454081988036 - 2.398327419614959*totchi - 47.99096500250743*totchi2 - 5.104257870393138*totchi2*totchi + 72.08174136362386*totchi2*totchi2)))/(-1.4160870461211452 + 1.*totchi)) + (-0.04426571511345366*dchi*delta*eta2))
        F1            = fAmpRDMin
        gamma1        = (v1RD / (fdamp * gamma3) ) * (F1*F1 - 2.*F1*fring + fring*fring + fdamp*fdamp*gamma3*gamma3)* np.exp(((F1 - fring) * gamma2 ) / (fdamp * gamma3))
        gammaR        = gamma2 / (fdamp * gamma3)
        gammaD2       = (gamma3 * fdamp) * (gamma3 * fdamp)
        gammaD13      = fdamp * gamma1 * gamma3
        fAmpInsMin    = 0.0026
        fAmpInsMax    = fMECO + 0.25*(fISCO - fMECO)
        fAmpMatchIN   = fAmpInsMax
        fAmpIntMax    = fAmpRDMin
        
        # TaylorF2 PN Amplitude Coefficients
        pnInitial     = 1.
        pnOneThird    = 0.
        pnTwoThirds   = ((-969 + 1804*eta)/672. ) * (np.pi**(2./3.))
        pnThreeThirds = ((81*(chi1 + chi2) + 81*chi1*delta - 81*chi2*delta - 44*(chi1 + chi2)*eta)/48. ) * np.pi
        pnFourThirds  = ((-27312085 - 10287648*chi12*(1 + delta) + 24*(428652*chi22*(-1 + delta) + (-1975055 + 10584*(81*chi12 - 94*chi1*chi2 + 81*chi22))*eta + 1473794*eta2))/8.128512e6 ) * (np.pi**(4./3.))
        pnFiveThirds  = ((-6048*chi12*chi1*(-1 - delta + (3 + delta)*eta) + chi2*(-((287213 + 6048*chi22)*(-1 + delta)) + 4*(-93414 + 1512*chi22*(-3 + delta) + 2083*delta)*eta - 35632*eta2) + chi1*(287213*(1 + delta) - 4*eta*(93414 + 2083*delta + 8908*eta)) + 42840*(-1 + 4*eta)*np.pi)/32256. ) * (np.pi**(5./3.))
        pnSixThirds   = ((-1242641879927 + 12.0*(28.0*(-3248849057.0 + 11088*(163199*chi12 - 266498*chi1*chi2 + 163199*chi22))*eta2 + 27026893936*eta2*eta - 116424*(147117*(-(chi22*(-1.0 + delta)) + chi12*(1.0 + delta)) + 60928*(chi1 + chi2 + chi1*delta - chi2*delta)*np.pi)  + eta*(545384828789.0 - 77616*(638642*chi1*chi2 + chi12*(-158633 + 282718*delta) - chi22*(158633.0 + 282718.0*delta) - 107520.0*(chi1 + chi2)*np.pi + 275520*np.pi*np.pi))))/6.0085960704e10 )*np.pi*np.pi
        
        # Now the collocation points for the inspiral
        CollocationValuesAmpIns0 = 0.
        CollocationValuesAmpIns1 = (((-0.015178276424448592 - 0.06098548699809163*eta + 0.4845148547154606*eta2)/(1. + 0.09799277215675059*eta)) + (((0.02300153747158323 + 0.10495263104245876*eta2)*chiPN + (0.04834642258922544 - 0.14189350657140673*eta)*eta*chiPN2*chiPN + (0.01761591799745109 - 0.14404522791467844*eta2)*chiPN2)/(1. - 0.7340448493183307*chiPN)) + (dchi*delta*eta2*eta2*(0.0018724905795891192 + 34.90874132485147*eta)))
        CollocationValuesAmpIns2 = (((-0.058572000924124644 - 1.1970535595488723*eta + 8.4630293045015*eta2)/(1. + 15.430818840453686*eta)) + (((-0.08746408292050666 + eta*(-0.20646621646484237 - 0.21291764491897636*chiPN) + eta2*(0.788717372588848 + 0.8282888482429105*chiPN) - 0.018924013869130434*chiPN)*chiPN)/(-1.332123330797879 + 1.*chiPN)) + (dchi*delta*eta2*eta2*(0.004389995099201855 + 105.84553997647659*eta)))
        CollocationValuesAmpIns3 = (((-0.16212854591357853 + 1.617404703616985*eta - 3.186012733446088*eta2 + 5.629598195000046*eta2*eta)/(1. + 0.04507019231274476*eta)) + ((chiPN*(1.0055835408962206 + eta2*(18.353433894421833 - 18.80590889704093*chiPN) - 0.31443470118113853*chiPN + eta*(-4.127597118865669 + 5.215501942120774*chiPN) + eta2*eta*(-41.0378120175805 + 19.099315016873643*chiPN)))/(5.852706459485663 - 5.717874483424523*chiPN + 1.*chiPN2)) + ( dchi*delta*eta2*eta2*(0.05575955418803233 + 208.92352600701068*eta)))
        
        CollocationPointsAmpIns0 = 0.25 * fAmpMatchIN
        CollocationPointsAmpIns1 = 0.50 * fAmpMatchIN
        CollocationPointsAmpIns2 = 0.75 * fAmpMatchIN
        CollocationPointsAmpIns3 = 1.00 * fAmpMatchIN
        
        V1 = CollocationValuesAmpIns1
        V2 = CollocationValuesAmpIns2
        V3 = CollocationValuesAmpIns3
        
        F1 = CollocationPointsAmpIns1
        F2 = CollocationPointsAmpIns2
        F3 = CollocationPointsAmpIns3
        
        rho1 = (-((F2**(8./3.))*(F3*F3*F3)*V1) + F2*F2*F2*(F3**(8./3.))*V1 + (F1**(8./3.))*(F3*F3*F3)*V2 - F1*F1*F1*(F3**(8./3.))*V2 - (F1**(8./3.))*(F2*F2*F2)*V3 + F1*F1*F1*(F2**(8./3.))*V3) / ((F1**(7./3.))*(np.cbrt(F1) - np.cbrt(F2))*(F2**(7./3.))*(np.cbrt(F1) - np.cbrt(F3))*(np.cbrt(F2) - np.cbrt(F3))*(F3**(7./3.)))
        rho2 = ((F2**(7./3.))*(F3*F3*F3)*V1 - F2*F2*F2*(F3**(7./3.))*V1 - (F1**(7./3.))*(F3*F3*F3)*V2 + F1*F1*F1*(F3**(7./3.))*V2 + (F1**(7./3.))*(F2*F2*F2)*V3 - F1*F1*F1*(F2**(7./3.))*V3 ) / ((F1**(7./3.))*(np.cbrt(F1) - np.cbrt(F2))*(F2**(7./3.))*(np.cbrt(F1) - np.cbrt(F3))*(np.cbrt(F2) - np.cbrt(F3))*(F3**(7./3.)))
        rho3 = ((F2**(8./3.))*(F3**(7./3.))*V1 - (F2**(7./3.))*(F3**(8./3.))*V1 - (F1**(8./3.))*(F3**(7./3.))*V2 + (F1**(7./3.))*(F3**(8./3.))*V2 + (F1**(8./3.))*(F2**(7./3.))*V3 - (F1**(7./3.))*(F2**(8./3.))*V3 ) / ( (F1**(7./3.))*(np.cbrt(F1) - np.cbrt(F2))*(F2**(7./3.))*(np.cbrt(F1) - np.cbrt(F3))*(np.cbrt(F2) - np.cbrt(F3))*(F3**(7./3.)))
        pnSevenThirds = rho1
        pnEightThirds = rho2
        pnNineThirds  = rho3
        
        # Now the intermediate region
        F1     = fAmpMatchIN
        F4     = fAmpRDMin
        
        d1     = (((chi2*(81 - 81*delta - 44*eta) + chi1*(81*(1 + delta) - 44*eta))*np.pi)/48. + ((-969 + 1804*eta)*(np.pi**(2./3.)))/(1008.*(F1**(1./3.))) + ((-27312085 - 10287648*chi22 + 10287648*chi22*delta - 10287648*chi12*(1 + delta) + 24*(-1975055 + 857304*chi12 - 994896*chi1*chi2 + 857304*chi22)*eta + 35371056*eta2)*(np.pi**(4./3.))*(F1**(1./3.)))/6.096384e6 + (5*(np.pi**(5./3.))*(-6048*chi12*chi1*(-1 - delta + (3 + delta)*eta) + chi1*(287213*(1 + delta) - 4*(93414 + 2083*delta)*eta - 35632*eta2) + chi2*(-((287213 + 6048*chi22)*(-1 + delta)) + 4*(-93414 + 1512*chi22*(-3 + delta) + 2083*delta)*eta - 35632*eta2) + 42840*(-1 + 4*eta)*np.pi)*(F1**(2./3.)))/96768. - ((np.pi**(2.0))*(-336*(-3248849057 + 1809550512*chi12 - 2954929824*chi1*chi2 + 1809550512*chi22)*eta2 - 324322727232*eta2*eta + 7*(177520268561 + 29362199328*chi22 - 29362199328*chi22*delta + 29362199328*chi12*(1 + delta) + 12160253952*(chi1 + chi2 + chi1*delta - chi2*delta)*np.pi) + 12*eta*(-545384828789 + 49568837472*chi1*chi2 - 12312458928*chi22 - 21943440288*chi22*delta + 77616*chi12*(-158633 + 282718*delta) - 8345272320*(chi1 + chi2)*np.pi + 21384760320*(np.pi**(2.0))))*F1)/3.0042980352e10 + (7.0/3.0)*(F1**(4./3.))*rho1 + (8./3.)*(F1**(5.0/3.0))*rho2 + 3*F1*F1*rho3)
        d4     = - np.exp(- gamma2 * (F4 - fring) / (fdamp*gamma3)) * gamma1 * ((F4 - fring)*(F4 - fring)*gamma2 + 2.0*fdamp*(F4 - fring)*gamma3 + fdamp*fdamp*gamma2*gamma3*gamma3) / (((F4 - fring)*(F4 - fring) + (fdamp*gamma3)*(fdamp*gamma3)) * ((F4 - fring)*(F4 - fring) + (fdamp*gamma3)*(fdamp*gamma3)))
        inspF1 = (pnInitial + (F1**(1./3.))*pnOneThird + (F1**(2./3.))*pnTwoThirds + F1*pnThreeThirds + F1*((F1**(1./3.))*pnFourThirds + (F1**(2./3.))*pnFiveThirds + F1*pnSixThirds + F1*((F1**(1./3.))*rho1 + (F1**(2./3.))*rho2 + F1*rho3)))
        rdF4   = np.exp(- (F4 - fring) * gammaR ) * (gammaD13) / ((F4 - fring)*(F4 - fring) + gammaD2)
        
        # Use d1 and d4 calculated above to get the derivative of the amplitude on the boundaries
        d1     = ((7.0/6.0) * (F1**(1./6.)) / inspF1) - ( (F1**(7./6.)) * d1 / (inspF1*inspF1))
        d4     = ((7.0/6.0) * (F4**(1./6.)) / rdF4)   - ( (F4**(7./6.)) * d4 / (rdF4*rdF4))
        
        if self.IntAmpVersion == 104:
            # Use a 4th order polynomial in intermediate - good extrapolation, recommended default fit
            F2     = F1 + (1./2.) * (F4 - F1)
            F3     = 0.
            
            V1     = (F1**(-7./6)) * inspF1
            V2     = (((1.4873184918202145 + 1974.6112656679577*eta + 27563.641024162127*eta2 - 19837.908020966777*eta2*eta)/(1. + 143.29004876335128*eta + 458.4097306093354*eta2)) + ((totchi*(27.952730865904343 + eta*(-365.55631765202895 - 260.3494489873286*totchi) + 3.2646808851249016*totchi + 3011.446602208493*eta2*totchi - 19.38970173389662*totchi2 + eta2*eta*(1612.2681322644232 - 6962.675551371755*totchi + 1486.4658089990298*totchi2)))/(12.647425554323242 - 10.540154508599963*totchi + 1.*totchi2)) + (dchi*delta*(-0.016404056649860943 - 296.473359655246*eta)*eta2))
            V3     = 0.
            V4     = (F4**(-7./6)) * rdF4
            
            V1     = 1.0 / V1
            V2     = 1.0 / V2
            V4     = 1.0 / V4
            # Reconstruct the phenomenological coefficients for the intermediate ansatz
            F12 = F1*F1
            F13 = F12*F1
            F14 = F13*F1
            F15 = F14*F1
            F16 = F15*F1
            F17 = F16*F1

            F22 = F2*F2
            F23 = F22*F2
            F24 = F23*F2
            F25 = F24*F2
            F26 = F25*F2
            F27 = F26*F2

            F32 = F3*F3
            F33 = F32*F3
            F34 = F33*F3
            F35 = F34*F3
            F36 = F35*F3
            F37 = F36*F3

            F42 = F4*F4
            F43 = F42*F4
            F44 = F43*F4
            F45 = F44*F4
            F46 = F45*F4
            F47 = F46*F4

            F1mF2 = F1-F2
            F1mF3 = F1-F3
            F1mF4 = F1-F4
            F2mF3 = F2-F3
            F2mF4 = F2-F4
            F3mF4 = F3-F4

            F1mF22 = F1mF2*F1mF2
            F1mF32 = F1mF3*F1mF3
            F1mF42 = F1mF4*F1mF4
            F2mF32 = F2mF3*F2mF3
            F2mF42 = F2mF4*F2mF4
            F3mF42 = F3mF4*F3mF4
            F1mF43 = F1mF4*F1mF4*F1mF4


            
            delta0 = ((-(d4*F12*F1mF22*F1mF4*F2*F2mF4*F4) + d1*F1*F1mF2*F1mF4*F2*F2mF42*F42 + F42*(F2*F2mF42*(-4*F12 + 3*F1*F2 + 2*F1*F4 - F2*F4)*V1 + F12*F1mF43*V2) + F12*F1mF22*F2*(F1*F2 - 2*F1*F4 - 3*F2*F4 + 4*F42)*V4)/(F1mF22*F1mF43*F2mF42))
            delta1 = ((d4*F1*F1mF22*F1mF4*F2mF4*(2*F2*F4 + F1*(F2 + F4)) + F4*(-(d1*F1mF2*F1mF4*F2mF42*(2*F1*F2 + (F1 + F2)*F4)) - 2*F1*(F44*(V1 - V2) + 3*F24*(V1 - V4) + F14*(V2 - V4) + 4*F23*F4*(-V1 + V4) + 2*F13*F4*(-V2 + V4) + F1*(2*F43*(-V1 + V2) + 6*F22*F4*(V1 - V4) + 4*F23*(-V1 + V4)))))/(F1mF22*F1mF43*F2mF42))
            delta2 = ((-(d4*F1mF22*F1mF4*F2mF4*(F12 + F2*F4 + 2*F1*(F2 + F4))) + d1*F1mF2*F1mF4*F2mF42*(F1*F2 + 2*(F1 + F2)*F4 + F42) - 4*F12*F23*V1 + 3*F1*F24*V1 - 4*F1*F23*F4*V1 + 3*F24*F4*V1 + 12*F12*F2*F42*V1 - 4*F23*F42*V1 - 8*F12*F43*V1 + F1*F44*V1 + F45*V1 + F15*V2 + F14*F4*V2 - 8*F13*F42*V2 + 8*F12*F43*V2 - F1*F44*V2 - F45*V2 - F1mF22*(F13 + F2*(3*F2 - 4*F4)*F4 + F12*(2*F2 + F4) + F1*(3*F2 - 4*F4)*(F2 + 2*F4))*V4)/(F1mF22*F1mF43*F2mF42))
            delta3 = ((d4*F1mF22*F1mF4*F2mF4*(2*F1 + F2 + F4) - d1*F1mF2*F1mF4*F2mF42*(F1 + F2 + 2*F4) + 2*(F44*(-V1 + V2) + 2*F12*F2mF42*(V1 - V4) + 2*F22*F42*(V1 - V4) + 2*F13*F4*(V2 - V4) + F24*(-V1 + V4) + F14*(-V2 + V4) + 2*F1*F4*(F42*(V1 - V2) + F22*(V1 - V4) + 2*F2*F4*(-V1 + V4)))) / (F1mF22*F1mF43*F2mF42))
            delta4 = ((-(d4*F1mF22*F1mF4*F2mF4) + d1*F1mF2*F1mF4*F2mF42 - 3*F1*F22*V1 + 2*F23*V1 + 6*F1*F2*F4*V1 - 3*F22*F4*V1 - 3*F1*F42*V1 + F43*V1 + F13*V2 - 3*F12*F4*V2 + 3*F1*F42*V2 - F43*V2 - F1mF22*(F1 + 2*F2 - 3*F4)*V4)/(F1mF22*F1mF43*F2mF42))
            delta5 = 0.
            
        elif self.IntAmpVersion in [1043, 105]:
            # Use a 5th order polynomial in intermediate - great agreement to calibration points but poor extrapolation
            F2     = F1 + (1./3.) * (F4 - F1)
            F3     = F1 + (2./3.) * (F4 - F1)
            
            V1     = (F1**(-7./6)) * inspF1
            V2     = (((2.2436523786378983 + 2162.4749081764216*eta + 24460.158604784723*eta2 - 12112.140570900956*eta2*eta)/(1. + 120.78623282522702*eta + 416.4179522274108*eta2)) + ((totchi*(6.727511603827924 + eta2*(414.1400701039126 - 234.3754066885935*totchi) - 5.399284768639545*totchi + eta*(-186.87972530996245 + 128.7402290554767*totchi)))/(3.24359204029217 - 3.975650468231452*totchi + 1.*totchi2)) + (dchi*delta*(-59.52510939953099 + 13.12679437100751*eta)*eta2))
            V3     = (((1.195392410912163 + 1677.2558976605421*eta + 24838.37133975971*eta2 - 17277.938868280915*eta2*eta)/(1. + 144.78606839716073*eta + 428.8155916011666*eta2)) + ((totchi*(-2.1413952025647793 + 0.5719137940424858*totchi + eta*(46.61350006858767 + 0.40917927503842105*totchi - 11.526500209146906*totchi2) + 1.1833965566688387*totchi2 + eta2*(-84.82318288272965 - 34.90591158988979*totchi + 19.494962340530186*totchi2)))/(-1.4786392693666195 + 1.*totchi)) + (dchi*delta*(-333.7662575986524 + 532.2475589084717*eta)*eta2*eta))
            V4     = (F4**(-7./6)) * rdF4
            
            V1     = 1.0 / V1
            V2     = 1.0 / V2
            V3     = 1.0 / V3
            V4     = 1.0 / V4
            # Reconstruct the phenomenological coefficients for the intermediate ansatz
            F12 = F1*F1
            F13 = F12*F1
            F14 = F13*F1
            F15 = F14*F1
            F16 = F15*F1
            F17 = F16*F1

            F22 = F2*F2
            F23 = F22*F2
            F24 = F23*F2
            F25 = F24*F2
            F26 = F25*F2
            F27 = F26*F2

            F32 = F3*F3
            F33 = F32*F3
            F34 = F33*F3
            F35 = F34*F3
            F36 = F35*F3
            F37 = F36*F3

            F42 = F4*F4
            F43 = F42*F4
            F44 = F43*F4
            F45 = F44*F4
            F46 = F45*F4
            F47 = F46*F4

            F1mF2 = F1-F2
            F1mF3 = F1-F3
            F1mF4 = F1-F4
            F2mF3 = F2-F3
            F2mF4 = F2-F4
            F3mF4 = F3-F4

            F1mF22 = F1mF2*F1mF2
            F1mF32 = F1mF3*F1mF3
            F1mF42 = F1mF4*F1mF4
            F2mF32 = F2mF3*F2mF3
            F2mF42 = F2mF4*F2mF4
            F3mF42 = F3mF4*F3mF4
            F1mF43 = F1mF4*F1mF4*F1mF4
            
            if self.IntAmpVersion == 105:
                delta0 = ((-(d4*F12*F1mF22*F1mF32*F1mF4*F2*F2mF3*F2mF4*F3*F3mF4*F4) - d1*F1*F1mF2*F1mF3*F1mF4*F2*F2mF3*F2mF42*F3*F3mF42*F42 + 5*F13*F24*F33*F42*V1 - 4*F12*F25*F33*F42*V1 - 5*F13*F23*F34*F42*V1 + 3*F1*F25*F34*F42*V1 + 4*F12*F23*F35*F42*V1 - 3*F1*F24*F35*F42*V1 - 10*F13*F24*F32*F43*V1 + 8*F12*F25*F32*F43*V1 + 5*F12*F24*F33*F43*V1 - 4*F1*F25*F33*F43*V1 + 10*F13*F22*F34*F43*V1 - 5*F12*F23*F34*F43*V1 - F25*F34*F43*V1 - 8*F12*F22*F35*F43*V1 + 4*F1*F23*F35*F43*V1 + F24*F35*F43*V1 + 5*F13*F24*F3*F44*V1 - 4*F12*F25*F3*F44*V1 + 15*F13*F23*F32*F44*V1 - 10*F12*F24*F32*F44*V1 - F1*F25*F32*F44*V1 - 15*F13*F22*F33*F44*V1 + 5*F1*F24*F33*F44*V1 + 2*F25*F33*F44*V1 - 5*F13*F2*F34*F44*V1 + 10*F12*F22*F34*F44*V1 - 5*F1*F23*F34*F44*V1 + 4*F12*F2*F35*F44*V1 + F1*F22*F35*F44*V1 - 2*F23*F35*F44*V1 - 10*F13*F23*F3*F45*V1 + 5*F12*F24*F3*F45*V1 + 2*F1*F25*F3*F45*V1 - F12*F23*F32*F45*V1 + 2*F1*F24*F32*F45*V1 - F25*F32*F45*V1 + 10*F13*F2*F33*F45*V1 + F12*F22*F33*F45*V1 - 3*F24*F33*F45*V1 - 5*F12*F2*F34*F45*V1 - 2*F1*F22*F34*F45*V1 + 3*F23*F34*F45*V1 - 2*F1*F2*F35*F45*V1 + F22*F35*F45*V1 + 5*F13*F22*F3*F46*V1 + 2*F12*F23*F3*F46*V1 - 4*F1*F24*F3*F46*V1 - 5*F13*F2*F32*F46*V1 - F1*F23*F32*F46*V1 + 2*F24*F32*F46*V1 - 2*F12*F2*F33*F46*V1 + F1*F22*F33*F46*V1 + 4*F1*F2*F34*F46*V1 - 2*F22*F34*F46*V1 - 3*F12*F22*F3*F47*V1 + 2*F1*F23*F3*F47*V1 + 3*F12*F2*F32*F47*V1 - F23*F32*F47*V1 - 2*F1*F2*F33*F47*V1 + F22*F33*F47*V1 - F17*F33*F42*V2 + 2*F16*F34*F42*V2 - F15*F35*F42*V2 + 2*F17*F32*F43*V2 - F16*F33*F43*V2 - 4*F15*F34*F43*V2 + 3*F14*F35*F43*V2 - F17*F3*F44*V2 - 4*F16*F32*F44*V2 + 8*F15*F33*F44*V2 - 3*F13*F35*F44*V2 + 3*F16*F3*F45*V2 - 8*F14*F33*F45*V2 + 4*F13*F34*F45*V2 + F12*F35*F45*V2 - 3*F15*F3*F46*V2 + 4*F14*F32*F46*V2 + F13*F33*F46*V2 - 2*F12*F34*F46*V2 + F14*F3*F47*V2 - 2*F13*F32*F47*V2 + F12*F33*F47*V2 + F17*F23*F42*V3 - 2*F16*F24*F42*V3 + F15*F25*F42*V3 - 2*F17*F22*F43*V3 + F16*F23*F43*V3 + 4*F15*F24*F43*V3 - 3*F14*F25*F43*V3 + F17*F2*F44*V3 + 4*F16*F22*F44*V3 - 8*F15*F23*F44*V3 + 3*F13*F25*F44*V3 - 3*F16*F2*F45*V3 + 8*F14*F23*F45*V3 - 4*F13*F24*F45*V3 - F12*F25*F45*V3 + 3*F15*F2*F46*V3 - 4*F14*F22*F46*V3 - F13*F23*F46*V3 + 2*F12*F24*F46*V3 - F14*F2*F47*V3 + 2*F13*F22*F47*V3 - F12*F23*F47*V3 + F12*F1mF22*F1mF32*F2*F2mF3*F3*(F4*(-3*F2*F3 + 4*(F2 + F3)*F4 - 5*F42) + F1*(F2*F3 - 2*(F2 + F3)*F4 + 3*F42))*V4)/(F1mF22*F1mF32*F1mF43*F2mF3*F2mF42*F3mF42))
                delta1 = ((d4*F1*F1mF22*F1mF32*F1mF4*F2mF3*F2mF4*F3mF4*(F1*F2*F3 + 2*F2*F3*F4 + F1*(F2 + F3)*F4) + F4*(d1*F1mF2*F1mF3*F1mF4*F2mF3*F2mF42*F3mF42*(2*F1*F2*F3 + F2*F3*F4 + F1*(F2 + F3)*F4) + F1*(F16*(F43*(V2 - V3) + 2*F33*(V2 - V4) + 3*F22*F4*(V3 - V4) + 3*F32*F4*(-V2 + V4) + 2*F23*(-V3 + V4)) + F13*F4*(F45*(-V2 + V3) + 5*F34*F4*(V2 - V4) + 4*F25*(V3 - V4) + 4*F35*(-V2 + V4) + 5*F24*F4*(-V3 + V4)) + F14*(3*F45*(V2 - V3) + 2*F35*(V2 - V4) + 5*F34*F4*(V2 - V4) + 10*F23*F42*(V3 - V4) + 10*F33*F42*(-V2 + V4) + 2*F25*(-V3 + V4) + 5*F24*F4*(-V3 + V4)) + F15*(3*F44*(-V2 + V3) + 2*F33*F4*(V2 - V4) + 5*F32*F42*(V2 - V4) + 4*F24*(V3 - V4) + 4*F34*(-V2 + V4) + 2*F23*F4*(-V3 + V4) + 5*F22*F42*(-V3 + V4)) - 5*F12*(-(F32*F3mF42*F43*(V1 - V2)) + 2*F23*(F44*(-V1 + V3) + 2*F32*F42*(V1 - V4) + F34*(-V1 + V4)) + F24*(F43*(V1 - V3) + 2*F33*(V1 - V4) + 3*F32*F4*(-V1 + V4)) + F22*F4*(F44*(V1 - V3) + 3*F34*(V1 - V4) + 4*F33*F4*(-V1 + V4))) + F1*(-(F32*F3mF42*(4*F3 + 3*F4)*F43*(V1 - V2)) + 2*F23*(F45*(-V1 + V3) + 5*F34*F4*(V1 - V4) + 4*F35*(-V1 + V4)) + 4*F25*(F43*(V1 - V3) + 2*F33*(V1 - V4) + 3*F32*F4*(-V1 + V4)) - 5*F24*F4*(F43*(V1 - V3) + 2*F33*(V1 - V4) + 3*F32*F4*(-V1 + V4)) + 3*F22*F4*(F45*(V1 - V3) + 4*F35*(V1 - V4) + 5*F34*F4*(-V1 + V4))) - 2*(-(F33*F3mF42*F44*(V1 - V2)) + F24*(2*F45*(-V1 + V3) + 5*F33*F42*(V1 - V4) + 3*F35*(-V1 + V4)) + F25*(F44*(V1 - V3) + 3*F34*(V1 - V4) + 4*F33*F4*(-V1 + V4)) + F23*F4*(F45*(V1 - V3) + 4*F35*(V1 - V4) + 5*F34*F4*(-V1 + V4))))))/(F1mF22*F1mF32*F1mF43*F2mF3*F2mF42*F3mF42) )
                delta2 = ((-(d4*F1mF22*F1mF32*F1mF4*F2mF3*F2mF4*F3mF4*(F2*F3*F4 + F12*(F2 + F3 + F4) + 2*F1*(F2*F3 + (F2 + F3)*F4))) - d1*F1mF2*F1mF3*F1mF4*F2mF3*F2mF42*F3mF42*(F1*F2*F3 + 2*(F2*F3 + F1*(F2 + F3))*F4 + (F1 + F2 + F3)*F42) + 5*F13*F24*F33*V1 - 4*F12*F25*F33*V1 - 5*F13*F23*F34*V1 + 3*F1*F25*F34*V1 + 4*F12*F23*F35*V1 - 3*F1*F24*F35*V1 + 5*F12*F24*F33*F4*V1 - 4*F1*F25*F33*F4*V1 - 5*F12*F23*F34*F4*V1 + 3*F25*F34*F4*V1 + 4*F1*F23*F35*F4*V1 - 3*F24*F35*F4*V1 - 15*F13*F24*F3*F42*V1 + 12*F12*F25*F3*F42*V1 + 5*F1*F24*F33*F42*V1 - 4*F25*F33*F42*V1 + 15*F13*F2*F34*F42*V1 - 5*F1*F23*F34*F42*V1 - 12*F12*F2*F35*F42*V1 + 4*F23*F35*F42*V1 + 10*F13*F24*F43*V1 - 8*F12*F25*F43*V1 + 20*F13*F23*F3*F43*V1 - 15*F12*F24*F3*F43*V1 - 20*F13*F2*F33*F43*V1 + 5*F24*F33*F43*V1 - 10*F13*F34*F43*V1 + 15*F12*F2*F34*F43*V1 - 5*F23*F34*F43*V1 + 8*F12*F35*F43*V1 - 15*F13*F23*F44*V1 + 10*F12*F24*F44*V1 + F1*F25*F44*V1 + 15*F13*F33*F44*V1 - 10*F12*F34*F44*V1 - F1*F35*F44*V1 + F12*F23*F45*V1 - 2*F1*F24*F45*V1 + F25*F45*V1 - F12*F33*F45*V1 + 2*F1*F34*F45*V1 - F35*F45*V1 + 5*F13*F2*F46*V1 + F1*F23*F46*V1 - 2*F24*F46*V1 - 5*F13*F3*F46*V1 - F1*F33*F46*V1 + 2*F34*F46*V1 - 3*F12*F2*F47*V1 + F23*F47*V1 + 3*F12*F3*F47*V1 - F33*F47*V1 - F17*F33*V2 + 2*F16*F34*V2 - F15*F35*V2 - F16*F33*F4*V2 + 2*F15*F34*F4*V2 - F14*F35*F4*V2 + 3*F17*F3*F42*V2 - F15*F33*F42*V2 - 10*F14*F34*F42*V2 + 8*F13*F35*F42*V2 - 2*F17*F43*V2 - 5*F16*F3*F43*V2 + 15*F14*F33*F43*V2 - 8*F12*F35*F43*V2 + 4*F16*F44*V2 - 15*F13*F33*F44*V2 + 10*F12*F34*F44*V2 + F1*F35*F44*V2 + F12*F33*F45*V2 - 2*F1*F34*F45*V2 + F35*F45*V2 - 4*F14*F46*V2 + 5*F13*F3*F46*V2 + F1*F33*F46*V2 - 2*F34*F46*V2 + 2*F13*F47*V2 - 3*F12*F3*F47*V2 + F33*F47*V2 + F17*F23*V3 - 2*F16*F24*V3 + F15*F25*V3 + F16*F23*F4*V3 - 2*F15*F24*F4*V3 + F14*F25*F4*V3 - 3*F17*F2*F42*V3 + F15*F23*F42*V3 + 10*F14*F24*F42*V3 - 8*F13*F25*F42*V3 + 2*F17*F43*V3 + 5*F16*F2*F43*V3 - 15*F14*F23*F43*V3 + 8*F12*F25*F43*V3 - 4*F16*F44*V3 + 15*F13*F23*F44*V3 - 10*F12*F24*F44*V3 - F1*F25*F44*V3 - F12*F23*F45*V3 + 2*F1*F24*F45*V3 - F25*F45*V3 + 4*F14*F46*V3 - 5*F13*F2*F46*V3 - F1*F23*F46*V3 + 2*F24*F46*V3 - 2*F13*F47*V3 + 3*F12*F2*F47*V3 - F23*F47*V3 - F1mF22*F1mF32*F2mF3*(F13*(F22 + F2*F3 + F32 - 3*F42) + F2*F3*F4*(3*F2*F3 - 4*(F2 + F3)*F4 + 5*F42) + F1*(F2*F3 + 2*(F2 + F3)*F4)*(3*F2*F3 - 4*(F2 + F3)*F4 + 5*F42) + F12*(2*F2*F3*(F2 + F3) + (F22 + F2*F3 + F32)*F4 - 6*(F2 + F3)*F42 + 5*F43))*V4)/(F1mF22*F1mF32*F1mF43*F2mF3*F2mF42*F3mF42) )
                delta3 = ((d4*F1mF22*F1mF32*F1mF4*F2mF3*F2mF4*F3mF4*(F12 + F2*F3 + (F2 + F3)*F4 + 2*F1*(F2 + F3 + F4)) + d1*F1mF2*F1mF3*F1mF4*F2mF3*F2mF42*F3mF42*(F1*F2 + F1*F3 + F2*F3 + 2*(F1 + F2 + F3)*F4 + F42) -5*F13*F24*F32*V1 + 4*F12*F25*F32*V1 + 5*F13*F22*F34*V1 - 2*F25*F34*V1 - 4*F12*F22*F35*V1 + 2*F24*F35*V1 + 10*F13*F24*F3*F4*V1 - 8*F12*F25*F3*F4*V1 - 5*F12*F24*F32*F4*V1 + 4*F1*F25*F32*F4*V1 -10*F13*F2*F34*F4*V1 + 5*F12*F22*F34*F4*V1 + 8*F12*F2*F35*F4*V1 - 4*F1*F22*F35*F4*V1 - 5*F13*F24*F42*V1 + 4*F12*F25*F42*V1 + 10*F12*F24*F3*F42*V1 - 8*F1*F25*F3*F42*V1 - 5*F1*F24*F32*F42*V1 +4*F25*F32*F42*V1 + 5*F13*F34*F42*V1 - 10*F12*F2*F34*F42*V1 + 5*F1*F22*F34*F42*V1 - 4*F12*F35*F42*V1 + 8*F1*F2*F35*F42*V1 - 4*F22*F35*F42*V1 - 5*F12*F24*F43*V1 + 4*F1*F25*F43*V1 -20*F13*F22*F3*F43*V1 + 10*F1*F24*F3*F43*V1 + 20*F13*F2*F32*F43*V1 - 5*F24*F32*F43*V1 + 5*F12*F34*F43*V1 - 10*F1*F2*F34*F43*V1 + 5*F22*F34*F43*V1 - 4*F1*F35*F43*V1 + 15*F13*F22*F44*V1 -5*F1*F24*F44*V1 - 2*F25*F44*V1 - 15*F13*F32*F44*V1 + 5*F1*F34*F44*V1 + 2*F35*F44*V1 - 10*F13*F2*F45*V1 - F12*F22*F45*V1 + 3*F24*F45*V1 + 10*F13*F3*F45*V1 + F12*F32*F45*V1 - 3*F34*F45*V1 +2*F12*F2*F46*V1 - F1*F22*F46*V1 - 2*F12*F3*F46*V1 + F1*F32*F46*V1 + 2*F1*F2*F47*V1 - F22*F47*V1 - 2*F1*F3*F47*V1 + F32*F47*V1 + F17*F32*V2 - 3*F15*F34*V2 + 2*F14*F35*V2 - 2*F17*F3*F4*V2 +F16*F32*F4*V2 + 5*F14*F34*F4*V2 - 4*F13*F35*F4*V2 + F17*F42*V2 - 2*F16*F3*F42*V2 + F15*F32*F42*V2 + F16*F43*V2 + 10*F15*F3*F43*V2 - 15*F14*F32*F43*V2 + 4*F1*F35*F43*V2 - 8*F15*F44*V2 +15*F13*F32*F44*V2 - 5*F1*F34*F44*V2 - 2*F35*F44*V2 + 8*F14*F45*V2 - 10*F13*F3*F45*V2 - F12*F32*F45*V2 + 3*F34*F45*V2 - F13*F46*V2 + 2*F12*F3*F46*V2 - F1*F32*F46*V2 - F12*F47*V2 +2*F1*F3*F47*V2 - F32*F47*V2 - F17*F22*V3 + 3*F15*F24*V3 - 2*F14*F25*V3 + 2*F17*F2*F4*V3 - F16*F22*F4*V3 - 5*F14*F24*F4*V3 + 4*F13*F25*F4*V3 - F17*F42*V3 + 2*F16*F2*F42*V3 - F15*F22*F42*V3 -F16*F43*V3 - 10*F15*F2*F43*V3 + 15*F14*F22*F43*V3 - 4*F1*F25*F43*V3 + 8*F15*F44*V3 - 15*F13*F22*F44*V3 + 5*F1*F24*F44*V3 + 2*F25*F44*V3 - 8*F14*F45*V3 + 10*F13*F2*F45*V3 + F12*F22*F45*V3 -3*F24*F45*V3 + F13*F46*V3 - 2*F12*F2*F46*V3 + F1*F22*F46*V3 + F12*F47*V3 - 2*F1*F2*F47*V3 + F22*F47*V3 + F1mF22*F1mF32*F2mF3*(2*F22*F32 + F13*(F2 + F3 - 2*F4) + F12*(F2 + F3 - 2*F4)*(2*(F2 + F3) + F4) - 4*(F22 + F2*F3 + F32)*F42 + 5*(F2 + F3)*F43 + F1*(4*F2*F3*(F2 + F3) - 4*(F22 + F2*F3 + F32)*F4 - 3*(F2 + F3)*F42 + 10*F43))*V4)/(F1mF22*F1mF32*F1mF43*F2mF3*F2mF42*F3mF42) )
                delta4 = ((-(d4*F1mF22*F1mF32*F1mF4*F2mF3*F2mF4*F3mF4*(2*F1 + F2 + F3 + F4)) - d1*F1mF2*F1mF3*F1mF4*F2mF3*F2mF42*F3mF42*(F1 + F2 + F3 + 2*F4) + 5*F13*F23*F32*V1 - 3*F1*F25*F32*V1 - 5*F13*F22*F33*V1 +2*F25*F33*V1 + 3*F1*F22*F35*V1 - 2*F23*F35*V1 - 10*F13*F23*F3*F4*V1 + 6*F1*F25*F3*F4*V1 + 5*F12*F23*F32*F4*V1 - 3*F25*F32*F4*V1 + 10*F13*F2*F33*F4*V1 - 5*F12*F22*F33*F4*V1 -6*F1*F2*F35*F4*V1 + 3*F22*F35*F4*V1 + 5*F13*F23*F42*V1 - 3*F1*F25*F42*V1 + 15*F13*F22*F3*F42*V1 - 10*F12*F23*F3*F42*V1 - 15*F13*F2*F32*F42*V1 + 5*F1*F23*F32*F42*V1 - 5*F13*F33*F42*V1 +10*F12*F2*F33*F42*V1 - 5*F1*F22*F33*F42*V1 + 3*F1*F35*F42*V1 - 10*F13*F22*F43*V1 + 5*F12*F23*F43*V1 + F25*F43*V1 + 15*F12*F22*F3*F43*V1 - 10*F1*F23*F3*F43*V1 + 10*F13*F32*F43*V1 -15*F12*F2*F32*F43*V1 + 5*F23*F32*F43*V1 - 5*F12*F33*F43*V1 + 10*F1*F2*F33*F43*V1 - 5*F22*F33*F43*V1 - F35*F43*V1 + 5*F13*F2*F44*V1 - 10*F12*F22*F44*V1 + 5*F1*F23*F44*V1 - 5*F13*F3*F44*V1 +10*F12*F32*F44*V1 - 5*F1*F33*F44*V1 + 5*F12*F2*F45*V1 + 2*F1*F22*F45*V1 - 3*F23*F45*V1 - 5*F12*F3*F45*V1 - 2*F1*F32*F45*V1 + 3*F33*F45*V1 - 4*F1*F2*F46*V1 + 2*F22*F46*V1 + 4*F1*F3*F46*V1 -2*F32*F46*V1 - 2*F16*F32*V2 + 3*F15*F33*V2 - F13*F35*V2 + 4*F16*F3*F4*V2 - 2*F15*F32*F4*V2 - 5*F14*F33*F4*V2 + 3*F12*F35*F4*V2 - 2*F16*F42*V2 - 5*F15*F3*F42*V2 + 10*F14*F32*F42*V2 -3*F1*F35*F42*V2 + 4*F15*F43*V2 - 5*F14*F3*F43*V2 + F35*F43*V2 + 5*F13*F3*F44*V2 - 10*F12*F32*F44*V2 + 5*F1*F33*F44*V2 - 4*F13*F45*V2 + 5*F12*F3*F45*V2 + 2*F1*F32*F45*V2 - 3*F33*F45*V2 +2*F12*F46*V2 - 4*F1*F3*F46*V2 + 2*F32*F46*V2 + 2*F16*F22*V3 - 3*F15*F23*V3 + F13*F25*V3 - 4*F16*F2*F4*V3 + 2*F15*F22*F4*V3 + 5*F14*F23*F4*V3 - 3*F12*F25*F4*V3 + 2*F16*F42*V3 +5*F15*F2*F42*V3 - 10*F14*F22*F42*V3 + 3*F1*F25*F42*V3 - 4*F15*F43*V3 + 5*F14*F2*F43*V3 - F25*F43*V3 - 5*F13*F2*F44*V3 + 10*F12*F22*F44*V3 - 5*F1*F23*F44*V3 + 4*F13*F45*V3 - 5*F12*F2*F45*V3 -2*F1*F22*F45*V3 + 3*F23*F45*V3 - 2*F12*F46*V3 + 4*F1*F2*F46*V3 - 2*F22*F46*V3 -F1mF22*F1mF32*F2mF3*(2*F2*F3*(F2 + F3) + 2*F12*(F2 + F3 - 2*F4) - 3*(F22 + F2*F3 + F32)*F4 + F1*(F22 + 5*F2*F3 + F32 - 6*(F2 + F3)*F4 + 5*F42) + 5*F43)*V4)/(F1mF22*F1mF32*F1mF43*F2mF3*F2mF42*F3mF42))
                delta5 = ((d4*F1mF22*F1mF32*F1mF4*F2mF3*F2mF4*F3mF4 + d1*F1mF2*F1mF3*F1mF4*F2mF3*F2mF42*F3mF42 - 4*F12*F23*F32*V1 + 3*F1*F24*F32*V1 + 4*F12*F22*F33*V1 - 2*F24*F33*V1 - 3*F1*F22*F34*V1 + 2*F23*F34*V1 +8*F12*F23*F3*F4*V1 - 6*F1*F24*F3*F4*V1 - 4*F1*F23*F32*F4*V1 + 3*F24*F32*F4*V1 - 8*F12*F2*F33*F4*V1 + 4*F1*F22*F33*F4*V1 + 6*F1*F2*F34*F4*V1 - 3*F22*F34*F4*V1 - 4*F12*F23*F42*V1 +3*F1*F24*F42*V1 - 12*F12*F22*F3*F42*V1 + 8*F1*F23*F3*F42*V1 + 12*F12*F2*F32*F42*V1 - 4*F23*F32*F42*V1 + 4*F12*F33*F42*V1 - 8*F1*F2*F33*F42*V1 + 4*F22*F33*F42*V1 - 3*F1*F34*F42*V1 +8*F12*F22*F43*V1 - 4*F1*F23*F43*V1 - F24*F43*V1 - 8*F12*F32*F43*V1 + 4*F1*F33*F43*V1 + F34*F43*V1 - 4*F12*F2*F44*V1 - F1*F22*F44*V1 + 2*F23*F44*V1 + 4*F12*F3*F44*V1 + F1*F32*F44*V1 -2*F33*F44*V1 + 2*F1*F2*F45*V1 - F22*F45*V1 - 2*F1*F3*F45*V1 + F32*F45*V1 + F15*F32*V2 - 2*F14*F33*V2 + F13*F34*V2 - 2*F15*F3*F4*V2 + F14*F32*F4*V2 + 4*F13*F33*F4*V2 - 3*F12*F34*F4*V2 +F15*F42*V2 + 4*F14*F3*F42*V2 - 8*F13*F32*F42*V2 + 3*F1*F34*F42*V2 - 3*F14*F43*V2 + 8*F12*F32*F43*V2 - 4*F1*F33*F43*V2 - F34*F43*V2 + 3*F13*F44*V2 - 4*F12*F3*F44*V2 - F1*F32*F44*V2 +2*F33*F44*V2 - F12*F45*V2 + 2*F1*F3*F45*V2 - F32*F45*V2 - F15*F22*V3 + 2*F14*F23*V3 - F13*F24*V3 + 2*F15*F2*F4*V3 - F14*F22*F4*V3 - 4*F13*F23*F4*V3 + 3*F12*F24*F4*V3 - F15*F42*V3 -4*F14*F2*F42*V3 + 8*F13*F22*F42*V3 - 3*F1*F24*F42*V3 + 3*F14*F43*V3 - 8*F12*F22*F43*V3 + 4*F1*F23*F43*V3 + F24*F43*V3 - 3*F13*F44*V3 + 4*F12*F2*F44*V3 + F1*F22*F44*V3 - 2*F23*F44*V3 +F12*F45*V3 - 2*F1*F2*F45*V3 + F22*F45*V3 + F1mF22*F1mF32*F2mF3*(2*F2*F3 + F1*(F2 + F3 - 2*F4) - 3*(F2 + F3)*F4 + 4*F42)*V4)/(F1mF22*F1mF32*F1mF43*F2mF3*F2mF42*F3mF42))
            else:
                V1mV2 = V1-V2
                V2mV3 = V2-V3
                V2mV4 = V2-V4
                V1mV3 = V1-V3
                V1mV4 = V1-V4
                V3mV4 = V3-V4
                
                delta0 = (-(d4*F1*F1mF2*F1mF3*F1mF4*F2*F2mF3*F2mF4*F3*F3mF4*F4) - F1*F1mF3*F1mF42*F3*F3mF42*F42*V2 + F24*(-(F1*F1mF42*F42*V3) + F33*(F42*V1 + F12*V4 - 2*F1*F4*V4) + F3*F4*(F43*V1 + 2*F13*V4 - 3*F12*F4*V4) - F32*(2*F43*V1 + F13*V4 - 3*F1*F42*V4)) + F2*F4*(F12*F1mF42*F43*V3 - F34*(F43*V1 + 2*F13*V4 - 3*F12*F4*V4) - F32*F4*(F44*V1 + 3*F14*V4 - 4*F13*F4*V4) + 2*F33*(F44*V1 + F14*V4 - 2*F12*F42*V4)) + F22*(-(F1*F1mF42*(2*F1 + F4)*F43*V3) + F3*F42*(F44*V1 + 3*F14*V4 - 4*F13*F4*V4) + F34*(2*F43*V1 + F13*V4 - 3*F1*F42*V4) - F33*(3*F44*V1 + F14*V4 - 4*F1*F43*V4)) + F23*(F1*F1mF42*(F1 + 2*F4)*F42*V3 - F34*(F42*V1 + F12*V4 - 2*F1*F4*V4) + F32*(3*F44*V1 + F14*V4 - 4*F1*F43*V4) - 2*F3*(F45*V1 + F14*F4*V4 - 2*F12*F43*V4)))/(F1mF2*F1mF3*F1mF42*F2mF3*F2mF42*F3mF42)
                delta1 = (d4*F1mF4*F2mF4*F3mF4*(F1*F2*F3 + F2*F3*F4 + F1*(F2 + F3)*F4) + (F4*(F12*F1mF42*F43*V2mV3 + F34*(F43*V1mV2 + 3*F12*F4*V2mV4 + 2*F13*(-V2 + V4)) + F32*F4*(F44*V1mV2 + 4*F13*F4*V2mV4 + 3*F14*(-V2 + V4)) + 2*F33*(F44*(-V1 + V2) + F14*V2mV4 + 2*F12*F42*(-V2 + V4)) + 2*F23*(F44*V1mV3 + F34*V1mV4 + 2*F12*F42*V3mV4 + 2*F32*F42*(-V1 + V4) + F14*(-V3 + V4)) + F24*(3*F32*F4*V1mV4 + F43*(-V1 + V3) + 2*F13*V3mV4 + 2*F33*(-V1 + V4) + 3*F12*F4*(-V3 + V4)) + F22*F4*(4*F33*F4*V1mV4 + F44*(-V1 + V3) + 3*F14*V3mV4 + 3*F34*(-V1 + V4) + 4*F13*F4*(-V3 + V4))))/(F1mF2*F1mF3*F2mF3))/(F1mF42*F2mF42*F3mF42)
                delta2 = (-(d4*F1mF2*F1mF3*F1mF4*F2mF3*F2mF4*F3mF4*(F3*F4 + F2*(F3 + F4) + F1*(F2 + F3 + F4))) - 2*F34*F43*V1 + 3*F33*F44*V1 - F3*F46*V1 - F14*F33*V2 + F13*F34*V2 + 3*F14*F3*F42*V2 - 3*F1*F34*F42*V2 - 2*F14*F43*V2 - 4*F13*F3*F43*V2 + 4*F1*F33*F43*V2 + 2*F34*F43*V2 + 3*F13*F44*V2 - 3*F33*F44*V2 - F1*F46*V2 + F3*F46*V2 + 2*F14*F43*V3 - 3*F13*F44*V3 + F1*F46*V3 + F2*F42*(F44*V1mV3 + 3*F34*V1mV4 - 4*F33*F4*V1mV4 - 3*F14*V3mV4 + 4*F13*F4*V3mV4) + F24*(2*F43*V1mV3 + F33*V1mV4 - 3*F3*F42*V1mV4 - F13*V3mV4 + 3*F1*F42*V3mV4) + F23*(-3*F44*V1mV3 - F34*V1mV4 + 4*F3*F43*V1mV4 + F14*V3mV4 - 4*F1*F43*V3mV4) + F14*F33*V4 - F13*F34*V4 - 3*F14*F3*F42*V4 + 3*F1*F34*F42*V4 + 4*F13*F3*F43*V4 - 4*F1*F33*F43*V4)/ (F1mF2*F1mF3*F1mF42*F2mF3*F2mF42*F3mF42)
                delta3 = (d4*F1mF2*F1mF3*F1mF4*F2mF3*F2mF4*F3mF4*(F1 + F2 + F3 + F4) + F34*F42*V1 - 3*F32*F44*V1 + 2*F3*F45*V1 + F14*F32*V2 - F12*F34*V2 - 2*F14*F3*F4*V2 + 2*F1*F34*F4*V2 + F14*F42*V2 - F34*F42*V2 + 4*F12*F3*F43*V2 - 4*F1*F32*F43*V2 - 3*F12*F44*V2 + 3*F32*F44*V2 + 2*F1*F45*V2 - 2*F3*F45*V2 - F14*F42*V3 + 3*F12*F44*V3 - 2*F1*F45*V3 + F24*(-(F42*V1mV3) - F32*V1mV4 + 2*F3*F4*V1mV4 + F12*V3mV4 - 2*F1*F4*V3mV4) - 2*F2*F4*(F44*V1mV3 + F34*V1mV4 - 2*F32*F42*V1mV4 - F14*V3mV4 + 2*F12*F42*V3mV4) + F22*(3*F44*V1mV3 + F34*V1mV4 - 4*F3*F43*V1mV4 - F14*V3mV4 + 4*F1*F43*V3mV4) - F14*F32*V4 + F12*F34*V4 + 2*F14*F3*F4*V4 - 2*F1*F34*F4*V4 - 4*F12*F3*F43*V4 + 4*F1*F32*F43*V4)/ (F1mF2*F1mF3*F1mF42*F2mF3*F2mF42*F3mF42)
                delta4 = (-(d4*F1mF2*F1mF3*F1mF4*F2mF3*F2mF4*F3mF4) - F33*F42*V1 + 2*F32*F43*V1 - F3*F44*V1 - F13*F32*V2 + F12*F33*V2 + 2*F13*F3*F4*V2 - 2*F1*F33*F4*V2 - F13*F42*V2 - 3*F12*F3*F42*V2 + 3*F1*F32*F42*V2 + F33*F42*V2 + 2*F12*F43*V2 - 2*F32*F43*V2 - F1*F44*V2 + F3*F44*V2 + F13*F42*V3 - 2*F12*F43*V3 + F1*F44*V3 + F23*(F42*V1mV3 + F32*V1mV4 - 2*F3*F4*V1mV4 - F12*V3mV4 + 2*F1*F4*V3mV4) + F2*F4*(F43*V1mV3 + 2*F33*V1mV4 - 3*F32*F4*V1mV4 - 2*F13*V3mV4 + 3*F12*F4*V3mV4) + F22*(-2*F43*V1mV3 - F33*V1mV4 + 3*F3*F42*V1mV4 + F13*V3mV4 - 3*F1*F42*V3mV4) + F13*F32*V4 - F12*F33*V4 - 2*F13*F3*F4*V4 + 2*F1*F33*F4*V4 + 3*F12*F3*F42*V4 - 3*F1*F32*F42*V4)/(F1mF2*F1mF3*F1mF42*F2mF3*F2mF42*F3mF42)
                delta5 = 0.
        else:
            raise ValueError('Intermediate amplitude version not implemented.')
        
        Overallamp = amp0 * ampNorm
        
        amplitudeIMR = np.where(fgrid <= fAmpMatchIN, (pnInitial + (fgrid**(1./3.))*pnOneThird + (fgrid**(2./3.))*pnTwoThirds + fgrid*pnThreeThirds + fgrid*((fgrid**(1./3.))*pnFourThirds + (fgrid**(2./3.))*pnFiveThirds + fgrid*pnSixThirds + fgrid*((fgrid**(1./3.))*rho1 + (fgrid**(2./3.))*rho2 + fgrid*rho3))), np.where(fgrid <= fAmpRDMin, (fgrid**(7./6.))/(delta0 + fgrid*(delta1 + fgrid*(delta2 + fgrid*(delta3 + fgrid*(delta4 + fgrid*delta5))))), np.where(fgrid <= self.fcutPar, np.exp(- (fgrid - fring) * gammaR ) * (gammaD13) / ((fgrid - fring)*(fgrid - fring) + gammaD2), 0.)))
        
        return Overallamp * amplitudeIMR * (fgrid**(-7./6.))
        
    def tau_star(self, f, **kwargs):
        utils.check_evparams(kwargs, checktidal=self.is_tidal)
        # For complex waveforms we use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        
        return self.fcutPar/(kwargs['Mc']*utils.GMsun_over_c3/(kwargs['eta']**(3./5.)))
