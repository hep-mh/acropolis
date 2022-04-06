#! /usr/bin/env python3

# sys
import sys; sys.path.append('../..')

# plots
from acropolis.plots import tex_title, tex_labels
from acropolis.plots import plot_scan_results, save_figure


# Set the data directory
data_dir = 'data/'


# s-wave, a vs mchi
plot_scan_results(
    data_dir + 'annih_swave_ee.dat', output_file='annih_swave_ee.pdf',
    title=tex_title(b=0, tempkd=0, braa=0), labels=tex_labels('mchi', 'a')
)


# p-wave, b vs mchi, tempkd = 1 MeV
plot_scan_results(
    data_dir + 'annih_pwave_Tkd_1e+00MeV_ee.dat', output_file='annih_pwave_1e0MeV_ee.pdf',
    title=tex_title(a=0, tempkd=1, braa=0), labels=tex_labels('mchi', 'b')
)


# decay, n0a vs mphi, tau = 1e7 s
plot_scan_results(
    data_dir + 'decay_tau_1e+07s_aa.dat', output_file='decay_1e7s_aa.pdf',
    title=tex_title(tau=1e7, temp0=10, braa=1), labels=tex_labels('mphi', 'n0a')
)


# decay, n0a vs tau, mphi = 50 MeV
_, ax = plot_scan_results(
    data_dir + 'decay_mphi_5e+01MeV_aa.dat', output_file=None, save_pdf=False,
    title=tex_title(mphi=50, temp0=10, braa=1), labels=tex_labels('tau', 'n0a')
)

# Add labels for the different abundances
ax.text(4.0, -10.0, r"D/$^1$H low"               , color='0.6'           , fontsize=16)
ax.text(7.5, -11.0, r"$^3$He/D high"             , color='mediumseagreen', fontsize=16)
ax.text(6.3,  -4.2, r"$\mathcal{Y}_\text{p}$ low", color='dodgerblue'    , fontsize=16)
ax.text(7.3,  -8.0, r"D/$^1$H high"              , color='tomato'        , fontsize=16)

save_figure('decay_50MeV_aa.pdf')
