# setuptools
from setuptools import setup

setup(
    name='ACROPOLIS',
    version='1.2.1',
    python_requires=">=3.6",
    description='A generiC fRamework fOr Photodisintegration Of LIght elementS',
    url='https://github.com/skumblex/acropolis',
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
