#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import numpy as np
from WF4Py import WFutils as utils

from .WFclass_definition import WaveFormModel

##############################################################################
# TAYLORF2 3.5 RESTRICTED PN WAVEFORM
##############################################################################

class TaylorF2_RestrictedPN(WaveFormModel):
    '''
    TaylorF2 restricted PN waveform model
    '''
    
    # This waveform model is restricted PN (the amplitude stays as in Newtonian approximation) up to 3.5 PN
    def __init__(self, fHigh=None, is_tidal=False, use_3p5PN_SpinHO=False, phiref_vlso=False, is_eccentric=False, fRef_ecc=None, **kwargs):
        
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
        self.fRef_ecc=fRef_ecc
        super().__init__(objectT, fHigh, is_tidal=is_tidal, is_eccentric=is_eccentric, **kwargs)
    
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

        if self.is_eccentric:
            # These are the eccentricity dependent coefficients up to 3 PN order, in the low-eccentricity limit, from arXiv:1605.00304
            ecc = kwargs['ecc']
            if self.fRef_ecc is None:
                v0ecc = np.amin(v, axis=0)
            else:
                v0ecc = (np.pi*Mtot_sec*self.fRef_ecc)**(1./3.)
                
            TF2EccCoeffs = {}
            
            TF2EccOverallAmpl = -2.355/1.462*ecc*ecc*((v0ecc/v)**(19./3.))
            
            TF2EccCoeffs['zero']      = 1.
            TF2EccCoeffs['one']       = 0.
            TF2EccCoeffs['twoV']      = 29.9076223/8.1976608 + 18.766963/2.927736*eta
            TF2EccCoeffs['twoV0']     = 2.833/1.008 - 19.7/3.6*eta
            TF2EccCoeffs['threeV']    = -28.19123/2.82600*np.pi
            TF2EccCoeffs['threeV0']   = 37.7/7.2*np.pi
            TF2EccCoeffs['fourV4']    = 16.237683263/3.330429696 + 241.33060753/9.71375328*eta+156.2608261/6.9383952*eta2
            TF2EccCoeffs['fourV2V02'] = 84.7282939759/8.2632420864-7.18901219/3.68894736*eta-36.97091711/1.05398496*eta2
            TF2EccCoeffs['fourV04']   = -1.193251/3.048192 - 66.317/9.072*eta +18.155/1.296*eta2
            TF2EccCoeffs['fiveV5']    = -28.31492681/1.18395270*np.pi - 115.52066831/2.70617760*np.pi*eta
            TF2EccCoeffs['fiveV3V02'] = -79.86575459/2.84860800*np.pi + 55.5367231/1.0173600*np.pi*eta
            TF2EccCoeffs['fiveV2V03'] = 112.751736071/5.902315776*np.pi + 70.75145051/2.10796992*np.pi*eta
            TF2EccCoeffs['fiveV05']   = 76.4881/9.0720*np.pi - 94.9457/2.2680*np.pi*eta
            TF2EccCoeffs['sixV6']     = -436.03153867072577087/1.32658535116800000 + 53.6803271/1.9782000*np.euler_gamma + 157.22503703/3.25555200*np.pi*np.pi +(2991.72861614477/6.89135247360 - 15.075413/1.446912*np.pi*np.pi)*eta +345.5209264991/4.1019955200*eta2 + 506.12671711/8.78999040*eta2*eta + 384.3505163/5.9346000*np.log(2.) - 112.1397129/1.7584000*np.log(3.)
            TF2EccCoeffs['sixV4V02']  = 46.001356684079/3.357073133568 + 253.471410141755/5.874877983744*eta - 169.3852244423/2.3313007872*eta2 - 307.833827417/2.497822272*eta2*eta
            TF2EccCoeffs['sixV3V03']  = -106.2809371/2.0347200*np.pi*np.pi
            TF2EccCoeffs['sixV2V04']  = -3.56873002170973/2.49880440692736 - 260.399751935005/8.924301453312*eta + 15.0484695827/3.5413894656*eta2 + 340.714213265/3.794345856*eta2*eta
            TF2EccCoeffs['sixV06']    = 265.31900578691/1.68991764480 - 33.17/1.26*np.euler_gamma + 12.2833/1.0368*np.pi*np.pi + (91.55185261/5.48674560 - 3.977/1.152*np.pi*np.pi)*eta - 5.732473/1.306368*eta2 - 30.90307/1.39968*eta2*eta + 87.419/1.890*np.log(2.) - 260.01/5.60*np.log(3.)
            
            phi_Ecc = TF2EccOverallAmpl*(TF2EccCoeffs['zero'] + TF2EccCoeffs['one']*v + (TF2EccCoeffs['twoV']*v*v + TF2EccCoeffs['twoV0']*v0ecc*v0ecc) + (TF2EccCoeffs['threeV']*v*v*v + TF2EccCoeffs['threeV0']*v0ecc*v0ecc*v0ecc) + (TF2EccCoeffs['fourV4']*v*v*v*v + TF2EccCoeffs['fourV2V02']*v*v*v0ecc*v0ecc + TF2EccCoeffs['fourV04']*v0ecc*v0ecc*v0ecc*v0ecc) + (TF2EccCoeffs['fiveV5']*v*v*v*v*v + TF2EccCoeffs['fiveV3V02']*v*v*v*v0ecc*v0ecc + TF2EccCoeffs['fiveV2V03']*v*v*v0ecc*v0ecc*v0ecc + TF2EccCoeffs['fiveV05']*v0ecc*v0ecc*v0ecc*v0ecc*v0ecc) + ((TF2EccCoeffs['sixV6'] + 53.6803271/3.9564000*np.log(16.*v*v))*(v**6) + TF2EccCoeffs['sixV4V02']*v*v*v*v*v0ecc*v0ecc + TF2EccCoeffs['sixV3V03']*v*v*v*v0ecc*v0ecc*v0ecc + TF2EccCoeffs['sixV2V04']*v*v*v0ecc*v0ecc*v0ecc*v0ecc + (TF2EccCoeffs['sixV06'] - 33.17/2.52*np.log(16.*v0ecc*v0ecc))*(v0ecc**6)))
        
        else:
            phi_Ecc = 0.
            
        if self.is_tidal:
            # Add tidal contribution if needed, as in PhysRevD.89.103012
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
            Lam_t, delLam    = utils.Lamt_delLam_from_Lam12(Lambda1, Lambda2, eta)
            
            phi_Tidal = (-0.5*39.*Lam_t)*(v**10.) + (-3115./64.*Lam_t + 6595./364.*Seta*delLam)*(v**12.)
            
        else:
            phi_Tidal = 0.
        
        phase = TF2OverallAmpl*(TF2coeffs['zero'] + TF2coeffs['one']*v + TF2coeffs['two']*v*v + TF2coeffs['three']*v**3 + TF2coeffs['four']*v**4 + (TF2coeffs['five'] + TF2coeffs['five_log']*np.log(v))*v**5 + (TF2coeffs['six'] + TF2coeffs['six_log']*np.log(v))*v**6 + TF2coeffs['seven']*v**7 + phi_Tidal + phi_Ecc)/(v**5.)
        
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
