# setuptools
from setuptools import setup
# os
from os import path

# Read the README.md file
cwd = path.abspath( path.dirname(__file__) )
with open(path.join(cwd, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Perform the actual setup
setup(
    name='ACROPOLIS',
    version='1.2.1',
    python_requires=">=3.6",
    description='A generiC fRamework fOr Photodisintegration Of LIght elementS',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://acropolis.hepforge.org',
    author='Paul Frederik Depta, Marco Hufnagel, Kai Schmidt-Hoberg',
    license='GPL3',
    packages=[
        'acropolis'
    ],
    package_data={
        'acropolis': ["data/*"]
    },
    include_package_data=True,
    scripts=[
        'decay',
        'annihilation'
    ],
    install_requires=[
        'numpy>=1.19.1',
        'scipy>=1.5.2',
        'numba>=0.51.2'
    ]
)
