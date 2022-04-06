# ACROPOLIS

**A** generi**C** f**R**amework f**O**r **P**hotodisintegration **O**f **LI**ght element**S**

![Language: Python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)
![Version: 1.2.2](https://img.shields.io/badge/Current_Version-1.2.2-green.svg?style=flat-square)
![DevVersion: 1.3](https://img.shields.io/badge/Current_Dev_Version-1.3-orange.svg?style=flat-square)

![Logo](https://acropolis.hepforge.org/ACROPOLIS.png)
When using this code for your own scientific publications, please cite

- **ACROPOLIS: A generiC fRamework fOr Photodisintegration Of LIght elementS**\
  Paul Frederik Depta, Marco Hufnagel, Kai Schmidt-Hoberg\
  https://arxiv.org/abs/2011.06518
- **Updated BBN constraints on electromagnetic decays of MeV-scale particles**\
  Paul Frederik Depta, Marco Hufnagel, Kai Schmidt-Hoberg\
  https://arxiv.org/abs/2011.06519
- **BBN constraints on MeV-scale dark sectors. Part II. Electromagnetic decays**\
  Marco Hufnagel, Kai Schmidt-Hoberg, Sebastian Wild\
  https://arxiv.org/abs/1808.09324

The most recent version of the manual can always be found on GitHub in the manual/ folder. The respective publication on arXiv might be out-of-date, especially when new versions of the code become available.

# Abstract

The remarkable agreement between observations of the primordial light element abundances and the corresponding theoretical predictions within the standard cosmological history provides a powerful method to constrain physics beyond the standard model of particle physics (BSM). For a given BSM model these primordial element abundances are generally determined by (i) Big Bang Nucleosynthesis and (ii) possible subsequent disintegration processes. The latter potentially change the abundance values due to late-time high-energy injections which may be present in these scenarios. While there are a number of public codes for the first part, no such code is currently available for the second. Here we close this gap and present ACROPOLIS, A generiC fRamework fOr Photodisintegration Of LIght elementS. The widely discussed cases of decays as well as annihilations can be run without prior coding knowledge within example programs. Furthermore, due to its modular structure, ACROPOLIS can easily be extended also to other scenarios.

# Changelog

v1.2.2\
(April 6, 2022)
 - Implemented fixes for the issues #10 and #11 on GitHub
 - Made some initial plotting functions available in ``acropolis.plots``, which can be used to easily plot the results of parameter scans
 - Improved the output that is printed to the screen (especially for parameter scans if ``verbose=True``)
 - Updated the neutron lifetime to the PDG 2020 recommended value
 - Included some example files, e.g. for parameter scans, in the directory examples/
 - Included a new c-file tools/create_sm_abundance_file.c, which can be used with [``AlterBBN``](https://alterbbn.hepforge.org/) to generate the file ``abundance_file.dat`` for sm.tar.gz
 - Fixed a bug that prohibited running 2d parameter scans without 'fast' parameters
 - Fixed a bug that caused INFO messages to be printed even for ``verbose=False``

v1.2.1\
(February 16, 2021)
 - Fixed a bug in ``DecayModel``. Results that have been obtained with older versions can be corrected by multiplying the parameter ``n0a`` with an additional factor ``2.7012``. All results of our papers remain unchanged.
 - Updated the set of initial abundances to the most recent values returned by [``AlterBBN``](https://alterbbn.hepforge.org/) v2.2 (explicitly, we used ``failsafe=12``)

v1.2\
(January 15, 2021)
 - Speed improvements when running non-thermal nucleosynthesis (by a factor 7)
 - Modified the directory structure by moving ./data to ./acropolis/data to transform ``ACROPOLIS`` into a proper package, which can be installed via ``python3 -m pip install . --user`` (also putting the executables ``decay`` and ``annihilation`` into your ``PATH``)
 - Added the decay of neutrons and tritium to the calculation
 - For AnnihilationModel, it is now possible to freely choose the dark-matter density parameter (default is 0.12)


v1.1\
(December 1, 2020)
 - For the source terms it is now possible to specify arbitrary monochromatic and continuous contributions, meaning that the latter one is no longer limited to only final-state radiation of photons
 - By including additional JIT compilation steps, the runtime without database files was drastically decreased (by approximately a factor 15)
 - The previously mentioned performance improvements also allowed to drop the large database files alltogether, which results in a better user experience (all database files are now part of the git repo and no additional download is required) and a significantly reduced RAM usage (&#x223C;900MB &#x2192; &#x223C;20MB)
 - Fixed a bug, which could lead to NaNs when calculating heavily suppressed spectra with E<sub>0</sub> &#x226B; me<sup>2</sup>/(22T)
 - Added a unified way to print the final abundances in order to declutter the wrapper scripts. This makes it easier to focus on the actual important parts when learning how to use ``ACROPOLIS``
 - Moved from bytecode to simple text files for the remaining database file, as the former leads to unexpected behaviour on some machines
 - Added additional info and warning messages for the user's convenience

v1.0\
(November 12, 2020)
 - Initial release

# Installation from PyPI

*This is the recommended way to install ACROPOLIS.*

To install ACROPOLIS from PyPI, first make sure that ``pip`` is installed on your system and afterwards simply execute the command

```
python3 -m pip install ACROPOLIS --user
```

Once the installation is completed, the different modules of ACROPOLIS can directly be imported into our own Python code (just like e.g. ``numpy``). Additionally, the installation also ensures that the two executable ``decay`` and ``annihilation`` are copied into your ``PATH`` and that all dependencies are fulfilled.

If any dependencies of ACROPOLIS conflict with those for other programs in your work environment, it is strongly advised to utilize the capabilities of Python's virtual environments.


# Installation from GitHub

To install ACROPOLIS directly from source on GitHub, start by cloning the respective git repository via the command

```
git clone https://github.com/hep-mh/acropolis.git
```

Afterward, switch into the newly created main directory and run

```
python3 -m pip install . --user
```

# Usage without installation

In case you just want to use ACROPOLIS without any additional installation steps, it is necessary to manually check that all dependencies are fulfilled. As specified in ``setup.py``, ACROPOLIS depends on the following packages (older versions might work, but have not been thoroughly tested)

 - NumPy (> 1.19.1)
 - SciPy (>1.5.2)
 - Numba (> 0.51.1)

The most recent versions of these packages can be collectively installed via the command

```
python3 -m pip install numpy, scipy, numba --user
```

Afterwards, you can import the different modules into your own Python code, as long as said code resides in the ``acropolis`` directory (like ``decay`` and ``annihilation``). If you instead want to also use the different modules from other directories, please consider using one of the two previously mentioned installation methods.


# Using the example models

ACROPOLIS ships with two executables, ``decay`` and ``annihilation``, which wrap the scenarios discussed in section 4.1 and section 4.2 from the manual, respectively. Both of these files need to be called with six command-line arguments each, a list of which can be obtained by running the command of choice without any arguments at all. As an example, the following command runs the process of photodisintegration for an unstable mediator with a mass of 10MeV and a lifetime of 10<sup>5</sup>s that decays exclusively into photons and has an abundance of 10<sup>-10</sup> relative to photons at a reference temperature of 10MeV (*if you did not install ACROPOLIS via pip, you have to run this command from within the main directory and make sure to append an additional ``./`` to the beginning of the commands*)

```
decay 10 1e5 10 1e-10 0 1
```

On a similar note, the following command runs the process of photodisintegration for residual s-wave annihilations of a dark-matter particle with a mass of 10MeV and a cross-section of 10<sup>-25</sup> cm³/s that annihilates exclusively into photons

```
annihilation 10 1e-25 0 0 0 1
```

# Supported platforms

ACROPOLIS should work on any platform that supports ``python3`` and ``clang``, the latter of which is required for ``numba`` to work.
