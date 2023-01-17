import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='WF4Py',
    version='1.1.0',
    author='Francesco Iacovelli',
    author_email='francesco.iacovelli@unige.ch',
    description='Gravitational waves waveform models in pure Python language',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/CosmoStatGW/WF4Py',
    license='GNU GPLv3',
    python_requires='>=3.7',
    packages=['WF4Py', 'WF4Py/waveform_models'],#setuptools.find_packages(),
    include_package_data=True,
    package_data={'':['WFfiles/*.txt', 'WFfiles/*.h5']},
    install_requires=['numpy', 'scipy', 'h5py'],
)
