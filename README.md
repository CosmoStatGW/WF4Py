# WF4Py
User-friendly package implementing GW waveform models in pure python, thus enabling parallelization over multiple events at a time. All the waveforms are accurately checked with their implementation in [LALSuite](<https://git.ligo.org/lscsoft/lalsuite>).

Developed by [Francesco Iacovelli](<https://github.com/FrancescoIacovelli>).

This package is released together with the paper [](<>). When making use of it, please cite the paper and the present git repository. Bibtex:

```
@article{Iacovelli,
    author = "Iacovelli, Francesco and Mancarella, Michele and Foffa, Stefano and Maggiore, Michele",
    title = "{}",
    eprint = "",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "6",
    year = "2022",
}
```

## Code Organization
The organisation of the repository is the following:

```
WF4Py/WF4Py/
     ├── waveforms.py
            Import of the waveform models in the folder 'waveform_models/' for ease of use
     ├── waveform_models/
         	Core: implementation of various GW waveform models present in
				LALSimulation in pure Python
     ├── WFutils.py
			Auxiliary functions: constants, conversions,
				spherical harmonics and parameter checks
     ├── WFfiles
    		Folder containing some text files needed for waveform computation
WF4Py/
	├── WF4Py_tutorial.ipynb
		Jupyter notebook with tutorial for the usage

```		

## Summary

* [Overview and usage](https://github.com/CosmoStatGW/WF4Py#Overview-and-usage)
* [Installation](https://github.com/CosmoStatGW/WF4Py#Installation)
* [Available models](https://github.com/CosmoStatGW/WF4Py#Available-models)
* [Testing](https://github.com/CosmoStatGW/WF4Py#Testing)
* [Bibliography](https://github.com/CosmoStatGW/WF4Py#Bibliography)

## Overview and usage
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
where <span style="color:red">```Mc```</span> denotes the **chirp mass** (in units of *solar masses*), <span style="color:red">```dL```</span> the **luminosity distance** (in *Gpc*), <span style="color:red">```iota```</span> the **inclination angle**, <span style="color:red">```eta```</span> the **symmetric mass ratio**, <span style="color:red">```chi1z```</span> and <span style="color:red">```chi2z```</span> the **adimensional spin components** of the two objects aligned with the orbital angular momentum and <span style="color:red">```Lambda1```</span> and <span style="color:red"> ```Lambda2```</span> the **adimensional tidal deformability** parameters of the objects in the tidal case.

Any waveform can then be easily used e.g. as

```python
waveforms.IMRPhenomD().Ampl(fgrids, **events)
```

#### For a detailed tutorial refer to ```WF4Py_tutorial.ipynb```

## Installation
To install the package without cloning the git repository simply run

```
pip install git+https://github.com/CosmoStatGW/WF4Py
```
## Available models
* (v1) <span style="color:green">```TaylorF2_RestrictedPN```</span> (1., 2., 3., 4., 5.)
* (v1) <span style="color:green">```IMRPhenomD```</span> (6., 7.)
* (v1) <span style="color:green">```IMRPhenomD_NRTidalv2```</span> (6., 7., 8.)
* (v1) <span style="color:green">```IMRPhenomHM```</span> (9., 10.)
* (v1) <span style="color:green">```IMRPhenomNSBH```</span> (8., 11.)
* (v1) <span style="color:green">```IMRPhenomXAS```</span> (12.)

## Testing
The adherence of all the models with their implementation in [LALSuite](<https://git.ligo.org/lscsoft/lalsuite>) is accuratly tested. As an example, we here report the comparison in the implementations of ```IMRPhenomXAS```

![alt text](<https://github.com/CosmoStatGW/WF4Py/blob/master/IMRPhenomXAS_Comparison.png>)

## Bibliography  
1. A. Buonanno et al. (2009) [arXiv:0907.0700](<https://arxiv.org/abs/0907.0700>)
2. P. Ajith (2011) [arXiv:1107.1267](<https://arxiv.org/abs/1107.1267>)
3. C. K. Mishra et al. (2016) [arXiv:1601.05588](<https://arxiv.org/abs/1601.05588>)
4. L. Wade et al. (2014) [arXiv:1402.5156](<https://arxiv.org/abs/1402.5156>)
5. B. Moore et al. (2016) [arXiv:1605.00304](<https://arxiv.org/abs/1605.00304>)
6. S. Husa et al. (2015) [arXiv:1508.07250](<https://arxiv.org/abs/1508.07250>)
7. S. Khan et al. (2015) [arXiv:1508.07253](<https://arxiv.org/abs/1508.07253>)
8. T. Dietrich et al. (2019) [arXiv:1905.06011](<https://arxiv.org/abs/1905.06011>)
9. L. London et al. (2018) [arXiv:1708.00404](<https://arxiv.org/abs/1708.00404>)
10. C. Kalaghatgi et al. (2019) [arXiv:1909.10010](<https://arxiv.org/abs/1909.10010>)
11. F. Pannarale et al. (2015) [arXiv:1509.00512](<https://arxiv.org/abs/1509.00512>)
12. G. Pratten et al. (2020) [arXiv:2001.11412](https://arxiv.org/abs/2001.11412)
