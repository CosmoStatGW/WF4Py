import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='WF4Py',
    version='1.0.0',
    author='Francesco Iacovelli',
    author_email='francesco.iacovelli@unige.ch',
    description='Gravitational waves waveform models in pure Python language',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/CosmoStatGW/WF4Py',
    license='GNU GPLv3',
    python_requires='>=3.7',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'abc', 'h5py'],
)
