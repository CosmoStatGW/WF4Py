# WF4Py
User-friendly package implementing GW waveform models in pure python, thus enabling parallelization over multiple events at a time. All the waveforms are accurately checked with their implementation in [LALSuite](<https://git.ligo.org/lscsoft/lalsuite>).

Developed by [Francesco Iacovelli](<https://github.com/FrancescoIacovelli>)

## Code Organization
The organisation of the repository is the following:

```bash
WF4Py/
        ├── WF4Py.py
            Core: implementation of some GW waveform models present in
				LALSimulation in pure Python
        ├── WFutils.py
			Auxiliary functions: constants, conversions,
				spherical harmonics and parameter checks
        ├── WF4Py_tutorial.ipynb
			Jupyter notebook with tutorial for the usage
        ├── WFfiles
    		Folder containing some needed text filed

```		

## Summary
Each waveform is a derived of the abstract class <span style="color:green">```WaveFormModel```</span>, and has built in functions <span style="color:blue">```Phi```</span>, <span style="color:blue">```Ampl```</span>, <span style="color:blue">```tau_star```</span> and <span style="color:blue">```fcut```</span> to compute the **phase**, **amplitude**, **time to coalescence** and **cut frequency**, respectively for the chosen catalog of events.

The functions accept as inputs the catalog as well as the frequencies at at which the computation has to be performed.

The catalog has to be a dictionary containing the parameters of the events, such as

```python
events = {'Mc':np.array([...]),
          'dL':np.array([...]),
          'iota':np.array([...]),
          'eta':np.array([...]),
          'chi1z':np.array([...]),
          'chi2z':np.array([...]),
          'Lambda1':np.array([...]),
          'Lambda2':np.array([...])}
```
where <span style="color:red">Mc</span> denotes the **chirp mass** (in units of *solar masses*), <span style="color:red">dL</span> the **luminosity distance** (in *Gpc*), <span style="color:red">iota</span> the **inclination angle**, <span style="color:red">eta</span> the **symmetric mass ratio**, <span style="color:red">chi1z</span> and <span style="color:red">chi2z</span> the **adimensional spin components** of the two objects aligned with the orbital angular momentum and <span style="color:red">Lambda1</span> and <span style="color:red"> Lambda2</span> the **adimensional tidal deformability** parameters of the objects in the tidal case.

Any waveform can then be easily used e.g. as

```python
WF4Py.IMRPhenomD().Ampl(fgrids, **events)
```

#### For a detailed tutorial refer to ```WF4Py_tutorial.ipynb```

## Available models
* (v1) <span style="color:green">```TaylorF2_RestrictedPN```</span> (1., 2., 3., 4.)
* (v1) <span style="color:green">```IMRPhenomD```</span> (5., 6.)
* (v1) <span style="color:green">```IMRPhenomD_NRTidalv2```</span> (5., 6., 7.)
* (v1) <span style="color:green">```IMRPhenomHM```</span> (8., 9.)
* (v1) <span style="color:green">```IMRPhenomNSBH```</span> (7., 10.)

## Testing
The adherence of all the models with their implementation in [LALSuite](<https://git.ligo.org/lscsoft/lalsuite>) is accuratly tested. As an example, we here report the comparison in the implementations of 

![alt text](<https://github.com/CosmoStatGW/WF4Py/blob/master/IMRPhenomHM_Comparison.png>)

## Bibliography  
1. A. Buonanno et al. (2009) [arXiv:0907.0700](<https://arxiv.org/abs/0907.0700>)
2. P. Ajith (2011) [arXiv:1107.1267](<https://arxiv.org/abs/1107.1267>)
3. C. K. Mishra et al. (2016) [arXiv:1601.05588](<https://arxiv.org/abs/1601.05588>)
4. L. Wade et al. (2014) [arXiv:1402.5156](<https://arxiv.org/abs/1402.5156>)
5. S. Husa et al. (2015) [arXiv:1508.07250](<https://arxiv.org/abs/1508.07250>)
6. S. Khan et al. (2015) [arXiv:1508.07253](<https://arxiv.org/abs/1508.07253>)
7. T. Dietrich et al. (2019) [arXiv:1905.06011](<https://arxiv.org/abs/1905.06011>)
8. L. London et al. (2018) [arXiv:1708.00404](<https://arxiv.org/abs/1708.00404>)
9. C. Kalaghatgi et al. (2019) [arXiv:1909.10010](<https://arxiv.org/abs/1909.10010>)
10. F. Pannarale et al. (2015) [arXiv:1509.00512](<https://arxiv.org/abs/1509.00512>)
