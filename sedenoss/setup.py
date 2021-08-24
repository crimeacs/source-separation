import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['sedenoss']

setup(
    name='sedenoss',
    version='0.2',
    packages=find_packages(),
    url='https://github.com/IMGW-univie/source-separation',
    license='MIT License',
    author='Artemii Novoselov',
    author_email='artemii.novoselov@univie.ac.at',
    description='Separation and denoising of seismically-induced signals with dual-path recurrent neural network architecture'
)


