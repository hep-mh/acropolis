# ACROPOLIS

**A generiC fRamework fOr Photodisintegration Of LIght elementS**

![Language: Python3](https://img.shields.io/badge/language-Python3-blue.svg?style=flat-square)
![Version: 1.1](https://img.shields.io/badge/current_version-1.1-blue.svg?style=flat-square)

When using this code, please cite the following papers

- https://arxiv.org/abs/2011.06518
- https://arxiv.org/abs/2011.06519
- https://arxiv.org/abs/1808.09324

The most recent version of the manual can always be found on GitHub at manual/manual.pdf. The respective publication on arXiv might be out-of-date, especially when new versions of the code become available.

# Abstract

The remarkable agreement between observations of the primordial light element abundances and the corresponding theoretical predictions within the standard cosmological history provides a powerful method to constrain physics beyond the standard model of particle physics (BSM). For a given BSM model these primordial element abundances are generally determined by (i) Big Bang Nucleosynthesis and (ii) possible subsequent disintegration processes. The latter potentially change the abundance values due to late-time high-energy injections which may be present in these scenarios. While there are a number of public codes for the first part, no such code is currently available for the second. Here we close this gap and present ACROPOLIS, A generiC fRamework fOr Photodisintegration Of LIght elementS. The widely discussed cases of decays as well as annihilations can be run without prior coding knowledge within example programs. Furthermore, due to its modular structure, ACROPOLIS can easily be extended also to other scenarios.

# Changelog

v1.1
 - For the source terms it is now possible to specify arbitrary monochromatic and continuous contributions, meaning that the latter one is no longer limited to only final-state radiation of photons
 - By including additional JIT compilation steps, the runtime without database files was drastically increased (by approximately a factor 15)
 - The previously mentioned performance improvements also allowed to drop the large database files alltogether, which results in a better user experience (all database files are now part of the git repo and no additional download is required) and a significantly reduced RAM usage (~900MB -> ~20MB)
 - Fixed a bug, which could lead to NaNs when calculating heavily suppressed spectra with E0 >> me2/(22*T)
 - Added a unified way to print the final abundances in order to declutter the wrapper scripts. This makes it easier to focus on the actual important parts when learning how to use ACROPOLIS
 - Moved from bytecode to simple text files for the remaining database file, as the former leads to unexpected behaviour on some machines
 - Added additional info and warning messages for the user's convenience

v1.0
 - Initial release

# Install the dependencies

ACROPOLIS is written in Python3.7 (remember that Python2 is dead) and depends on the following packages (older versions might work, but have not been thoroughly testes)

 - NumPy (> 1.19.1)
 - SciPy (>1.5.2)
 - Numba (> 0.51.1)

The most recent versions of these packages can be collectively installed at user-level, i.e. without the need for root access, by executing the command

```
python3 -m pip install numpy, scipy, numba --user
```

If these dependencies conflict with those for other programs in your work environment, it is strongly advised to utilise the capabilities of Python's virtual environments.

# Use the example models

Within the ACROPOLIS main directory there are two executables, ``decay`` and ``annihilation``, which wrap the scenarios discussed in section 4.1 and section 4.2 from the manual, respectively. Both of these files need to be called with six command-line arguments each, a list of which can be obtained by running the command of choice without any arguments at all. On that note, the following command runs the process of photodisintegration for an unstable mediator with a mass of 10MeV and a lifetime of 1e5s that decays exclusively into photons and has an abundance of 1e-10 relative to photons at a reference temperature of 10MeV

```
./decay 10 1e5 10 1e-10 0 1
```

On a similar note, the following command runs the process of photodisintegration for residual s-wave annihilations of a dark matter particle with a mass of 10MeV and a cross-section of 10e-25 cm³/s that annihialtes exclusively into photons

```
./annihilation 10 1e-25 0 0 0 1
```

# Supported platforms

ACROPOLIS has been tested on the following platforms (but lets be honest, it should run on every OS with a Python implementation)

| OS              | Version   | Arch   |
| --------------- | :-------: | :----: |
| Ubuntu          |  20.04    | x86_64 |
| KDE neon        |  5.20     | x86_64 |
| Kali GNU/Linux  |  2020.04  | x86_64 |
| macOS           |  10.15    | x86_64 |
| Windows 10      |  20H2     | x86_64 |
| Android         |  10.0     | arm64  |
