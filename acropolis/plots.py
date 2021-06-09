# math
from math import log10, floor, ceil
# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
# warnings
import warnings

# pprint
from acropolis.pprint import print_info
# params
from acropolis.params import NY


# Set the general style of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

# Include additional latex packages
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


_plot_number = 0

# The number of sigmas at which a
# point is considered excluded
_95cl = 1.95996 # 95% C.L.


def _get_abundances(data, i):
    i0 = i + 2

    # Extract the different abundances...
    mean, high, low = data[:,i0], data[:,i0+NY], data[:,i0+2*NY]
    # ...and calculate an estimate for the error
    diff = np.minimum( np.abs( mean - high ), np.abs( mean - low ) )

    return mean, diff


def _get_deviations(data):
    # Extract and sum up neutrons and protons
    mn, en     = _get_abundances(data, 0)
    mp, ep     = _get_abundances(data, 1)
    mH, eH     = mn + mp, np.sqrt( en**2. + ep**2. )

    # Extract and sum up lithium-7 and berylium-7
    mLi7, eLi7 = _get_abundances(data, 7)
    mBe7, eBe7 = _get_abundances(data, 8)
    m7,   e7   = mLi7 + mBe7, np.sqrt( eLi7**2. + eBe7**2. )

    # Extract deuterium
    mD  , eD   = _get_abundances(data, 2)

    # Extract and sum up tritium and helium-3
    mT  , eT   = _get_abundances(data, 3)
    mHe3, eHe3 = _get_abundances(data, 4)
    m3  , e3   = mT + mHe3, np.sqrt( eT**2. + eHe3**2. )

    # Extract helium-4
    mHe4, eHe4 = _get_abundances(data, 5)

    # Calculate the actual deviations
    with warnings.catch_warnings(record=True) as w:
        # Calculate the relevant abundance ratios
        mYp , eYp  = 4.*mHe4, 4.*eHe4
        mDH , eDH  = mD/mH, (mD/mH)*np.sqrt( (eD/mD)**2. + (eH/mH)**2. )
        mHeD, eHeD = m3/mD, (m3/mD)*np.sqrt( (e3/m3)**2. + (eD/mD)**2. )
        mLiH, eLiH = m7/mH, (m7/mH)*np.sqrt( (e7/m7)**2. + (eH/mH)**2. )

        # Calculate the corresponding deviations
        Yp  = (mYp  -  2.45e-1) / np.sqrt( ( 0.03e-1)**2. +  eYp**2. )
        DH  = (mDH  - 2.547e-5) / np.sqrt( (0.035e-5)**2. +  eDH**2. )
        HeD = (mHeD -   8.3e-1) / np.sqrt( (  1.5e-1)**2. + eHeD**2. )
        LiH = (mLiH -  1.6e-10) / np.sqrt( ( 0.3e-10)**2. + eLiH**2. )

        if len(w) == 1 and issubclass(w[0].category, RuntimeWarning):
            # Nothing to do here
            pass

    # Take care of potential NaNs
    HeD[mDH<0.035e-5] =  10
    DH [np.isnan(DH)] = -10

    # Return (without reshaping)
    return Yp, DH, HeD, LiH


def init_figure():
    fig = plt.figure(figsize=(.4*12, .4*11), dpi=150, edgecolor='white')
    ax  = fig.add_subplot(1, 1, 1)

    ax.tick_params(axis='both', which='both', labelsize=11, direction='in', width=0.5)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)

    return fig, ax


def set_tick_labels(ax, x, y):
    nint = lambda val: ceil(val) if val >= 0 else floor(val)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    xmin_log = nint( log10(xmin) )
    xmax_log = nint( log10(xmax) )
    ymin_log = nint( log10(ymin) )
    ymax_log = nint( log10(ymax) )

    nx = abs( xmax_log - xmin_log ) + 1
    ny = abs( ymax_log - ymin_log ) + 1

    # Set the ticks on the x-axis
    xticks_major  = np.linspace(xmin_log, xmax_log, nx)
    xticks_minor  = [ log10(i*10**j) for i in range(1, 10) for j in xticks_major ]
    xlabels       = [ r'$10^{' + f'{int(i)}' + '}$' for i in xticks_major ]

    xticks_major_locator   = FixedLocator(xticks_major)
    xticks_minor_locator   = FixedLocator(xticks_minor)
    xlabels_formatter      = FixedFormatter(xlabels)

    ax.xaxis.set_major_locator(xticks_major_locator)
    ax.xaxis.set_minor_locator(xticks_minor_locator)
    ax.xaxis.set_major_formatter(xlabels_formatter)
    ax.set_xlim(xmin_log, xmax_log)

    # Set the ticks on the y-axis
    yticks_major  = np.linspace(ymin_log, ymax_log, ny)
    yticks_minor  = [ log10(i*10**j) for i in range(1, 10) for j in yticks_major ]
    ylabels       = [ r'$10^{' + f'{int(i)}' + '}$' for i in yticks_major ]

    yticks_major_locator   = FixedLocator(yticks_major)
    yticks_minor_locator   = FixedLocator(yticks_minor)
    ylabels_formatter      = FixedFormatter(ylabels)

    ax.yaxis.set_major_locator(yticks_major_locator)
    ax.yaxis.set_minor_locator(yticks_minor_locator)
    ax.yaxis.set_major_formatter(ylabels_formatter)
    ax.set_ylim(ymin_log, ymax_log)


def save_figure(output_file=None):
    global _plot_number

    # If no name for the output file is given
    # simply enumerate the different plots
    if output_file is None:
        output_file = 'acropolis_plot_{}.pdf'.format(_plot_number)
        _plot_number += 1

    plt.savefig(output_file)

    print_info(
        "Figure has been saved as '{}'".format(output_file),
        "acropolis.plot.plot_scan_results"
    )


def plot_scan_results(data, output_file=None,
                      title='', labels=('', ''),
                      save_pdf=True, show_fig=False):
    # If data is a filename, load the data first
    if type(data) == str:
        data = np.loadtxt(data)

    # Get the set of input parameters...
    x, y = data[:,0], data[:,1]

    # ...and determine the shape of the data
    N  = len(x)
    Nx = (x == x[0]).sum()
    Ny = N//Nx

    shape = (Nx, Ny)

    # Calculate the abundance deviations
    Yp, DH, HeD, LiH = _get_deviations(data)

    # Reshape the input data...
    x   =   x.reshape(shape)
    y   =   y.reshape(shape)
    # ...and the deviation arrays
    Yp  =  Yp.reshape(shape)
    DH  =  DH.reshape(shape)
    HeD = HeD.reshape(shape)
    LiH = LiH.reshape(shape)

    # Extract the overall exclusion limit
    max = np.maximum( np.abs(DH), np.abs(Yp) )
    max = np.maximum( max, HeD )

    # Init the figure and...
    fig, ax = init_figure()
    # ...set the tick labels
    set_tick_labels(ax, x, y)

    # Plot the actual data
    cut = 1e10
    # Deuterium (filled)
    ax.contourf(np.log10(x), np.log10(y), DH,
        levels=[-cut, -_95cl, _95cl, cut],
        colors=['0.6','white', 'tomato'],
        alpha=0.2
    )
    # Helium-4 (filled)
    ax.contourf(np.log10(x), np.log10(y), Yp,
        levels=[-cut, -_95cl, _95cl, cut],
        colors=['dodgerblue','white', 'lightcoral'],
        alpha=0.2
    )
    # Helium-3 (filled)
    ax.contourf(np.log10(x), np.log10(y), HeD,
        levels=[_95cl, cut], # Only use He3/D as an upper limit
        colors=['mediumseagreen'],
        alpha=0.2
    )

    # Deuterium low (line)
    ax.contour(np.log10(x), np.log10(y), DH,
        levels=[-_95cl], colors='0.6', linestyles='-'
    )
    # Deuterium high (line)
    ax.contour(np.log10(x), np.log10(y), DH,
        levels=[_95cl], colors='tomato', linestyles='-'
    )
    # Helium-4 low (line)
    ax.contour(np.log10(x), np.log10(y), Yp,
        levels=[-_95cl], colors='dodgerblue', linestyles='-'
    )
    # Helium-3 high (line)
    ax.contour(np.log10(x), np.log10(y), HeD,
        levels=[_95cl], colors='mediumseagreen', linestyles='-'
    )
    # Overall high/low (line)
    ax.contour(np.log10(x), np.log10(y), max,
        levels=[_95cl], colors='black', linestyles='-'
    )

    # Set the title...
    ax.set_title( title, fontsize=11 )
    # ...and the axis labels
    ax.set_xlabel( labels[0] )
    ax.set_ylabel( labels[1] )

    # Set tight layout
    plt.tight_layout()

    if save_pdf == True:
        save_figure(output_file)

    if show_fig == True:
        plt.show()

    # Return figure and axis in case
    # further manipulation is desired
    return fig, ax
