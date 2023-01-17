[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18914.svg)](http://dx.doi.org/10.5281/zenodo.7060240) [![Documentation Status](https://readthedocs.org/projects/wf4py/badge/?version=latest)](https://wf4py.readthedocs.io/en/latest/?badge=latest)
# WF4Py
User-friendly package implementing GW waveform models in pure python, thus enabling parallelization over multiple events at a time. All the waveforms are accurately checked with their implementation in [LALSuite](<https://git.ligo.org/lscsoft/lalsuite>).

Developed by [Francesco Iacovelli](<https://github.com/FrancescoIacovelli>).

This package is released together with the papers [arXiv:2207.02771](<https://arxiv.org/abs/2207.02771>) and [arXiv:2207.06910](<https://arxiv.org/abs/2207.06910>), where detail of implementations and results can be found.

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
WF4Py/docs/ 
		Code documentation in Sphinx

```		

## Summary
* [Documentation](https://github.com/CosmoStatGW/WF4Py#Documentation)
* [Installation](https://github.com/CosmoStatGW/WF4Py#Installation)
* [Overview and usage](https://github.com/CosmoStatGW/WF4Py#Overview-and-usage)
* [Available models](https://github.com/CosmoStatGW/WF4Py#Available-models)
* [Testing](https://github.com/CosmoStatGW/WF4Py#Testing)
* [Citation](https://github.com/CosmoStatGW/WF4Py#Citation)
* [Bibliography](https://github.com/CosmoStatGW/WF4Py#Bibliography)

## Documentation

WF4Py has its documentation hosted on Read the Docs [here](<https://wf4py.readthedocs.io/en/latest/>), and it can also be built from the ```docs``` directory.

## Installation
To install the package without cloning the git repository simply run

```
pip install git+https://github.com/CosmoStatGW/WF4Py
```

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

#### For a detailed tutorial refer to [```WF4Py_tutorial.ipynb```](<https://github.com/CosmoStatGW/WF4Py/blob/master/WF4Py_tutorial.ipynb>)

## Available models
* (v1.0.0) <span style="color:green">```TaylorF2_RestrictedPN```</span> (1., 2., 3., 4., 5.)
* (v1.0.0) <span style="color:green">```IMRPhenomD```</span> (6., 7.)
* (v1.0.0) <span style="color:green">```IMRPhenomD_NRTidalv2```</span> (6., 7., 8.)
* (v1.0.0) <span style="color:green">```IMRPhenomHM```</span> (9., 10.)
* (v1.0.0) <span style="color:green">```IMRPhenomNSBH```</span> (8., 11.)
* (v1.0.0) <span style="color:green">```IMRPhenomXAS```</span> (12.)

## Testing
The adherence of all the models with their implementation in [LALSuite](<https://git.ligo.org/lscsoft/lalsuite>) is accuratly tested. As an example, we here report the comparison in the implementations of ```IMRPhenomXAS```

![alt text](<https://github.com/CosmoStatGW/WF4Py/blob/master/IMRPhenomXAS_Comparison.png>)

## Citation
If using this software, please cite this repository and the papers [arXiv:2207.02771](<https://arxiv.org/abs/2207.02771>) and [arXiv:2207.06910](<https://arxiv.org/abs/2207.06910>). Bibtex:

```
@article{Iacovelli:2022bbs,
    author = "Iacovelli, Francesco and Mancarella, Michele and Foffa, Stefano and Maggiore, Michele",
    title = "{Forecasting the Detection Capabilities of Third-generation Gravitational-wave Detectors Using GWFAST}",
    eprint = "2207.02771",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.3847/1538-4357/ac9cd4",
    journal = "Astrophys. J.",
    volume = "941",
    number = "2",
    pages = "208",
    year = "2022"
}
```

```
@article{Iacovelli:2022mbg,
    author = "Iacovelli, Francesco and Mancarella, Michele and Foffa, Stefano and Maggiore, Michele",
    title = "{GWFAST: A Fisher Information Matrix Python Code for Third-generation Gravitational-wave Detectors}",
    eprint = "2207.06910",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.3847/1538-4365/ac9129",
    journal = "Astrophys. J. Supp.",
    volume = "263",
    number = "1",
    pages = "2",
    year = "2022"
}
```

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
