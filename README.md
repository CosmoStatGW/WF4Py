# PyWF
User-friendly package implementing GW waveform models in pure python, used in the GWFAST code 

## Code Organization

```bash
PyWF/PyWF/
        ├── PyWF.py 
                Core: implementation of some GW waveform models present in 
		LALSimulation in pure Python, namely TaylorF2, IMRPhenomD, 
		IMRPhenomD_NRTidalv2, IMRPhenomHM and IMRPhenomNSBH
        ├── WFutils.py
		Auxiliary functions: constants, conversions, 
		spherical harmonics and parameter checks
        ├── PyWF_tutorial.ipynb
		Jupyter notebook with tutorial for the usage
        			
```		

## Summary
Each waveform is a derived of the abstract class "WaveFormModel", and has built in functions "Phi", "Ampl", "tau_star" and "fcut" to compute the phase, amplitude, time to coalescence and cut frequency for the chosen catalog of events.
The functions "Phi", "Ampl", "tau_star" accept as inputs the catalog as well as the frequencies at at which performing the computation.
The catalog has to be a dictionary containing the parameters of the events, such as 
    events = {'Mc':np.array([...]), 
              'dL':np.array([...]), 
              'iota':np.array([...]),
              'eta':np.array([...]),
              'chi1z':np.array([...]), 
              'chi2z':np.array([...]),
              'Lambda1':np.array([...]), 
              'Lambda2':np.array([...])
}
where Mc denotes the chirp mass (in units of solar masses), dL the luminosity distance (in Gpc), iota the inclination angle, eta the symmetric mass ratio, chi1z and chi2z the adimensional spin components of the two objects aligned with the orbital angular momentum and Lambda1 and Lambda2 the adimensional tidal deformability parameters of the objects in the tidal case.
Any waveform can then be easily used e.g. as 
    PyWF.IMRPhenomD().Ampl(fgrids, **events)
