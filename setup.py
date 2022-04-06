# setuptools
from setuptools import setup
# os
from os import path

# version
from acropolis.info import version, description, url, authors

# Read the README.md file
cwd = path.abspath( path.dirname(__file__) )
with open(path.join(cwd, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Perform the actual setup
setup(
    name='ACROPOLIS',
    version=version,
    python_requires='>=3.6',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=url,
    author=authors,
    license='GPL3',
    packages=[
        'acropolis'
    ],
    package_data={
        'acropolis': ['data/*']
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
