#! /usr/bin/env python3

# sys
import sys; sys.path.append('..')

# plots
from acropolis.plots import plt
from acropolis.plots import plot_scan_results, save_figure


# Define some label strings for...
# ...scans with AnnihilationModel...
mchi_tex = r'$m_\chi\;[\mathrm{MeV}]$'
a_tex    = r'$a\;[\mathrm{cm^3/s}]$'
b_tex    = r'$b\;[\mathrm{cm^3/s}]$'
# ...and scans with DecayModel
mphi_tex = r'$m_\phi\;[\mathrm{MeV}]$'
tau_tex  = r'$\tau_\phi\;[\mathrm{s}]$'
n0a_tex  = r'$(n_\phi/n_\gamma)|_{T=T_0}$'


# r'$b = 0,\;T_\text{kd}=0,\;\text{BR}_{\gamma\gamma}=1-\text{BR}_{e^+e^-} = 0$'
# r'$a = 0,\;T_\text{kd}=1\;\mathrm{MeV},\;\text{BR}_{\gamma\gamma}=1-\text{BR}_{e^+e^-} = 0$'
# r"$\tau_\phi = 10^7\,\mathrm{s},\;T_0=10\,\mathrm{MeV},\;\text{BR}_{\gamma\gamma}=1-\text{BR}_{e^+e^-} = 1$"
# r'$m_\phi = 50\,\mathrm{MeV},\;T_0=10\,\mathrm{MeV},\;\text{BR}_{\gamma\gamma}=1-\text{BR}_{e^+e^-} = 1$'


def _build_annih_title(mchi=None, a=None, b=None, tempkd=None, braa=None):
    return r''


def _build_decay_title(mphi=None, tau=None, temp0=None, n0a=None, braa=None):
    return r''


plot_scan_results(
    'data/annih_swave_ee.dat', output_file='annih_swave_ee.pdf',
    title=_build_annih_title(b=0, tempkd=0, braa=0), labels=(mchi_tex, a_tex)
)


plot_scan_results(
    'data/annih_pwave_1e0MeV_ee.dat', output_file='annih_pwave_1e0MeV_ee.pdf',
    title=_build_annih_title(a=0, tempkd=1, braa=0), labels=(mchi_tex, b_tex)
)


plot_scan_results(
    'data/decay_1e7s_aa.dat', output_file='decay_1e7s_aa.pdf',
    title=_build_decay_title(tau=1e7, temp0=10, braa=1), labels=(mphi_tex, n0a_tex)
)


_, ax = plot_scan_results(
    'data/decay_50MeV_aa.dat', output_file=None, save_pdf=False,
    title=_build_decay_title(mphi=50, temp0=10, braa=1) ,labels=(tau_tex, n0a_tex)
)

ax.text(4.0, -10.0, r"D/$^1$H low"               , color='0.6'           , fontsize=16)
ax.text(7.5, -11.0, r"$^3$He/D high"             , color='mediumseagreen', fontsize=16)
ax.text(6.3,  -4.2, r"$\mathcal{Y}_\text{p}$ low", color='dodgerblue'    , fontsize=16)
ax.text(7.3,  -8.0, r"D/$^1$H high"              , color='tomato'        , fontsize=16)

save_figure('decay_50MeV_aa.pdf')
