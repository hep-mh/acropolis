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


# TODO Log printing !!!
def _build_annih_title(mchi=None, a=None, b=None, tempkd=None, braa=None):
    eof = r',\;'

    # Build the title
    title = r'$'

    if a is not None:
        title += r'a = ' + str(a) + (r'\,\mathrm{cm^3/s}' if a != 0 else r'') + eof

    if b is not None:
        title += r'b = ' + str(b) + (r'\,\mathrm{cm^3/s}' if b != 0 else r'') + eof

    if mchi is not None:
        title += r'm_\chi = ' + str(mchi) + (r'\,\mathrm{MeV}' if mchi != 0 else r'') + eof

    if tempkd is not None:
        title += r'T_\text{kd} = ' + str(tempkd) + (r'\,\mathrm{MeV}' if tempkd != 0 else r'') + eof

    if braa is not None:
        title += r'\text{BR}_{\gamma\gamma} = 1-\text{BR}_{e^+e^-} = ' + str(braa) + eof

    if title.endswith(eof):
        title = title[:-len(eof)]

    return title + '$'


def _build_decay_title(mphi=None, tau=None, temp0=None, n0a=None, braa=None):
    eof = r',\;'

    # Build the title
    title = r'$'

    if mphi is not None:
        title += r'm_\phi = ' + str(mphi) + (r'\,\mathrm{MeV}' if mphi != 0 else r'') + eof

    if tau is not None:
        title += r'\tau_\phi = 10^' + str(tau) + (r'\,\mathrm{s}' if tau != 0 else r'') + eof

    if temp0 is not None:
        title += r'T_0 = ' + str(temp0) + (r'\,\mathrm{MeV}' if temp0 != 0 else r'') + eof

    if n0a is not None:
        title += r'(n_\phi/n_\gamma)|_{T=T_0} = ' + str(n0a) + eof

    if braa is not None:
        title += r'\text{BR}_{\gamma\gamma} = 1-\text{BR}_{e^+e^-} = ' + str(braa) + eof


    if title.endswith(eof):
        title = title[:-len(eof)]

    return title + '$'


plot_scan_results(
    'data/annih_swave_ee.dat', output_file='annih_swave_ee.pdf',
    title=_build_annih_title(b=0, tempkd=0, braa=0), labels=(mchi_tex, a_tex)
)


plot_scan_results(
    'data/annih_pwave_Tkd_1e+00MeV_ee.dat', output_file='annih_pwave_1e0MeV_ee.pdf',
    title=_build_annih_title(a=0, tempkd=1, braa=0), labels=(mchi_tex, b_tex)
)


plot_scan_results(
    'data/decay_tau_1e+07s_aa.dat', output_file='decay_1e7s_aa.pdf',
    title=_build_decay_title(tau=7, temp0=10, braa=1), labels=(mphi_tex, n0a_tex)
)


_, ax = plot_scan_results(
    'data/decay_mphi_5e+01MeV_aa.dat', output_file=None, save_pdf=False,
    title=_build_decay_title(mphi=50, temp0=10, braa=1) ,labels=(tau_tex, n0a_tex)
)

ax.text(4.0, -10.0, r"D/$^1$H low"               , color='0.6'           , fontsize=16)
ax.text(7.5, -11.0, r"$^3$He/D high"             , color='mediumseagreen', fontsize=16)
ax.text(6.3,  -4.2, r"$\mathcal{Y}_\text{p}$ low", color='dodgerblue'    , fontsize=16)
ax.text(7.3,  -8.0, r"D/$^1$H high"              , color='tomato'        , fontsize=16)

save_figure('decay_50MeV_aa.pdf')
