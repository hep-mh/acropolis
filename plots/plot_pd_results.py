#! /usr/bin/env python3

# sys
import sys; sys.path.append('..')

# numpy
import numpy as np
# matplotlib
from matplotlib.lines import Line2D

# plots
from acropolis.plots import init_figure, save_figure


ref_values = {
    'NE_pd': 150,
    'NT_pd':  50
}

xlims = {
    'NE_pd': (10, 500),
    'NT_pd': (10, 200)
}

ylims = {
    'NE_pd': (1.4, 1.9),
    'NT_pd': (1.4, 1.9)
}

colors = {
    'annih': 'crimson',
    'decay': 'mediumorchid'
}


def invert(param):
    if param == 'NE_pd':
        return 'NT_pd'
    
    if param == 'NT_pd':
        return 'NE_pd'

    return None


# Loop over both parameters, i.e. 'NE_pd' and 'NT_pd'
for param in ['NE_pd', 'NT_pd']:
    iparam = invert(param)

    # Initialize the figure
    fig, ax = init_figure()

    # Loop over both data samples, i.e. 'decay' and 'annih'
    for run in ['annih', 'decay']:

        data = np.loadtxt(f'../tools/data/{param}_{run}.dat')
        # -->
        N_pd     = data[:,0]
        Y2H_low  = data[:,1]
        Y2H_mean = data[:,2]
        Y2H_high = data[:,3]

        ax.plot(N_pd, Y2H_low *1e5, '-' , color=colors[run])
        ax.plot(N_pd, Y2H_mean*1e5, '--', color=colors[run])
        ax.plot(N_pd, Y2H_high*1e5, '-.', color=colors[run])

    # Plot the reference line for the current parameter
    ax.plot([ ref_values[param] ]*2, [ *ylims[param] ], ':', color='black', zorder=-1)

    # Plot the reference value for the other paramater
    ax.text(xlims[param][1]*( 1 - 3e-2 ), ylims[param][0]*(1 + 2e-2), rf"$\texttt{{{iparam}={ref_values[iparam]}}}$", ha='right')

    # Plot the legend for the different line styles
    custom_lines = [
        Line2D([0], [0], linestyle='-' , color='black', lw=1),
        Line2D([0], [0], linestyle='--', color='black', lw=1),
        Line2D([0], [0], linestyle='-.', color='black', lw=1)
    ]
    # -->
    ax.legend(custom_lines, ['Low', 'Mean', 'High'], loc='upper right', fontsize=11, frameon=False)

    # Set the labels for the x- and y-axis
    ax.set_xlabel(rf'$\texttt{{{param}}}$')
    ax.set_ylabel(r'$Y_{{}^2\mathrm{H}}\;\;[\times 10^5]$')

    # Set the limits for the x- and y-axis
    ax.set_xlim(*xlims[param])
    ax.set_ylim(*ylims[param])

    # Apply a tighter layout
    fig.tight_layout()

    # Save the figure
    save_figure(f'{param}.pdf', show_fig=True)

