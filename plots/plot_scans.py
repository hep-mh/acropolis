#! /usr/bin/env python3

# sys
import sys; sys.path.append('..')

# plots
from acropolis.plots import plt
from acropolis.plots import plot_scan_results


plot_scan_results(
    'data/annih_swave_ee.dat', output_file='annih_swave_ee.pdf',
    title=r'$b = 0,\;T_\text{kd}=0,\;\text{BR}_{\gamma\gamma}=1-\text{BR}_{e^+e^-} = 0$',
    labels=[r'$m_\chi\;[\mathrm{MeV}]$', r'$a\;[\mathrm{cm^3/s}]$'],
)

plot_scan_results(
    'data/annih_pwave_1e0MeV_ee.dat', output_file='annih_pwave_1e0MeV_ee.pdf',
    title=r'$a = 0,\;T_\text{kd}=1\;\mathrm{MeV},\;\text{BR}_{\gamma\gamma}=1-\text{BR}_{e^+e^-} = 0$',
    labels=[r'$m_\chi\;[\mathrm{MeV}]$', r'$b\;[\mathrm{cm^3/s}]$'],
)

#plot_scan_results('data/decay_1e7s_aa.dat')

#plot_scan_results('data/decay_50MeV_aa.dat')
