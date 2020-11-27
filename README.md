# ACROPOLIS

**A generiC fRamework fOr Photodisintegration Of LIght elementS**

# Abstract

The remarkable agreement between observations of the primordial light element abundances and the corresponding theoretical predictions within the standard cosmological history provides a powerful method to constrain physics beyond the standard model of particle physics (BSM). For a given BSM model these primordial element abundances are generally determined by (i) Big Bang Nucleosynthesis and (ii) possible subsequent disintegration processes. The latter potentially change the abundance values due to late-time high-energy injections which may be present in these scenarios. While there are a number of public codes for the first part, no such code is currently available for the second. Here we close this gap and present ACROPOLIS, A generiC fRamework fOr Photodisintegration Of LIght elementS. The widely discussed cases of decays as well as annihilations can be run without prior coding knowledge within example programs. Furthermore, due to its modular structure, ACROPOLIS can easily be extended also to other scenarios.

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

On a similar note, the following command runs the process of photodisintegration for residual s-wave annihilations of a dark matter particle with a mass of 10MeV that annihialtes exclusively into photons

```
./annihilation 10 1e-25 0 0 0 1
```
