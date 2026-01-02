from setuptools import setup, find_packages

setup(
    name="peru_production",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'seaborn',
        'xarray',
        'statsmodels',
        'cartopy'
    ]
) 