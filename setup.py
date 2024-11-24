from setuptools import setup, find_packages

setup(
    name='SynthWeave',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy',
    ],
    author='Dawid Wolkiewicz',
    description='A library for plug-in modality fusion for integration of multimodal feeds.'
)