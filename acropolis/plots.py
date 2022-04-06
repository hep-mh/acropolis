# math
from math import log10, floor, ceil
# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
# warnings
import warnings

# obs
from acropolis.obs import pdg2020
# pprint
from acropolis.pprint import print_info
# params
from acropolis.params import NY


# Set the general style of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

# Include additional latex packages
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{mathpazo}'


# A global variable counting the
# number of created plots, in order
# to provide unique plot identifiers
_plot_number = 0


# The number of sigmas at which a
# point is considered excluded
_95cl = 1.95996 # 95% C.L.


# DATA EXTRACTION ###################################################

def _get_abundance(data, i):
    # Add + 2 for the two parameters in the first two columns
    i0 = i + 2

    # Extract the different abundances...
    mean, high, low = data[:,i0], data[:,i0+NY], data[:,i0+2*NY]
    # ...and calculate an estimate for the error
    diff = np.minimum( np.abs( mean - high ), np.abs( mean - low ) )

    return mean, diff


def _get_deviations(data, obs):
    # Extract and sum up neutrons and protons
    mn, en     = _get_abundance(data, 0)
    mp, ep     = _get_abundance(data, 1)
    mH, eH     = mn + mp, np.sqrt( en**2. + ep**2. )

    # Extract and sum up lithium-7 and berylium-7
    mLi7, eLi7 = _get_abundance(data, 7)
    mBe7, eBe7 = _get_abundance(data, 8)
    m7,   e7   = mLi7 + mBe7, np.sqrt( eLi7**2. + eBe7**2. )

    # Extract deuterium
    mD  , eD   = _get_abundance(data, 2)

    # Extract and sum up tritium and helium-3
    mT  , eT   = _get_abundance(data, 3)
    mHe3, eHe3 = _get_abundance(data, 4)
    m3  , e3   = mT + mHe3, np.sqrt( eT**2. + eHe3**2. )

    # Extract helium-4
    mHe4, eHe4 = _get_abundance(data, 5)

    # Calculate the actual deviations
    with warnings.catch_warnings(record=True) as w:
        # Calculate the relevant abundance ratios
        mYp , eYp  = 4.*mHe4, 4.*eHe4
        mDH , eDH  = mD/mH, (mD/mH)*np.sqrt( (eD/mD)**2. + (eH/mH)**2. )
        mHeD, eHeD = m3/mD, (m3/mD)*np.sqrt( (e3/m3)**2. + (eD/mD)**2. )
        mLiH, eLiH = m7/mH, (m7/mH)*np.sqrt( (e7/m7)**2. + (eH/mH)**2. )

        # Calculate the corresponding deviations
        Yp  = (mYp  - obs[ 'Yp'].mean) / np.sqrt( obs[ 'Yp'].err**2. +  eYp**2. )
        DH  = (mDH  - obs[ 'DH'].mean) / np.sqrt( obs[ 'DH'].err**2. +  eDH**2. )
        HeD = (mHeD - obs['HeD'].mean) / np.sqrt( obs['HeD'].err**2. + eHeD**2. )
        LiH = (mLiH - obs['LiH'].mean) / np.sqrt( obs['LiH'].err**2. + eLiH**2. )

        if len(w) == 1 and issubclass(w[0].category, RuntimeWarning):
            # Nothing to do here
            pass

    # Take care of potential NaNs
    HeD[ mDH < obs['DH'].err ] =  10
    DH [     np.isnan(DH)    ] = -10

    # Return (without reshaping)
    return Yp, DH, HeD, LiH


# LATEX INFORMATION #################################################


_tex_data = {
    # DecayModel
    'mphi'  : (r'm_\phi'                                         , r'\mathrm{MeV}'   ),
    'tau'   : (r'\tau_\phi'                                      , r'\mathrm{s}'     ),
    'temp0' : (r'T_0'                                            , r'\mathrm{MeV}'   ),
    'n0a'   : (r'(n_\phi/n_\gamma)|_{T=T_0}'                     , r''               ),
    # AnnihilationModel
    'braa'  : (r'\text{BR}_{\gamma\gamma} = 1-\text{BR}_{e^+e^-}', r''               ),
    'mchi'  : (r'm_\chi'                                         , r'\mathrm{MeV}'   ),
    'a'     : (r'a'                                              , r'\mathrm{cm^3/s}'),
    'b'     : (r'b'                                              , r'\mathrm{cm^3/s}'),
    'tempkd': (r'T_\text{kd}'                                    , r'\mathrm{MeV}'   ),
}


def add_tex_data(key, tex, unit):
    global _tex_data

    _tex_data[key] = (tex, unit)


def tex_title(**kwargs):
    global _tex_data

    if _tex_data is None:
        return

    eof = r',\;'

    # Define a function to handle values
    # that need to be printed in scientific
    # notation
    def _val_to_string(val):
        if type(val) == float:
            power = log10( val )
            if power != int(power):
                # TODO
                pass

            return r'10^' + str( int(power) )

        return str( val )

    title = r''
    for key in kwargs.keys():
        # Extract the numerical value
        val     = kwargs[ key ]
        val_str = _val_to_string( val )
        # Extract the latex representation
        # of the parameter and its unit
        tex, unit = _tex_data[ key ]
        # If the value is 0, do not print units
        unit = '\,' + unit if val != 0 else r''

        title += tex + '=' + val_str + unit + eof

    if title.endswith(eof):
        title = title[:-len(eof)]

    return r'$' + title + r'$'


def tex_label(key):
    if key not in _tex_data.keys():
        return ''

    tex, unit = _tex_data[ key ]

    if unit != r'':
        unit = r'\;[' + unit + r']'

    return r'$' + tex + unit + r'$'


def tex_labels(key_x, key_y):
    return ( tex_label(key_x), tex_label(key_y) )


# FIGURE HANDLING ###################################################

def _init_figure():
    fig = plt.figure(figsize=(4.8, 4.4), dpi=150, edgecolor='white')
    ax  = fig.add_subplot(1, 1, 1)

    ax.tick_params(axis='both', which='both', labelsize=11, direction='in', width=0.5)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)

    return fig, ax


def _set_tick_labels(ax, x, y):
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

    xticks_major_locator = FixedLocator(xticks_major)
    xticks_minor_locator = FixedLocator(xticks_minor)
    xlabels_formatter    = FixedFormatter(xlabels)

    ax.xaxis.set_major_locator(xticks_major_locator)
    ax.xaxis.set_minor_locator(xticks_minor_locator)
    ax.xaxis.set_major_formatter(xlabels_formatter)
    ax.set_xlim(xmin_log, xmax_log)

    # Set the ticks on the y-axis
    yticks_major  = np.linspace(ymin_log, ymax_log, ny)
    yticks_minor  = [ log10(i*10**j) for i in range(1, 10) for j in yticks_major ]
    ylabels       = [ r'$10^{' + f'{int(i)}' + '}$' for i in yticks_major ]

    yticks_major_locator = FixedLocator(yticks_major)
    yticks_minor_locator = FixedLocator(yticks_minor)
    ylabels_formatter    = FixedFormatter(ylabels)

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
        "acropolis.plot.save_figure"
    )


def plot_scan_results(data, output_file=None, title='', labels=('', ''), save_pdf=True, show_fig=False, obs=pdg2020):
    # If data is a filename, load the data first
    if type(data) == str:
        data = np.loadtxt(data)

    # Get the set of input parameters...
    x, y = data[:,0], data[:,1]

    # ...and determine the shape of the data
    N  = len(x)
    Ny = (x == x[0]).sum()
    Nx = N//Ny

    shape = (Nx, Ny)

    # Calculate the abundance deviations
    Yp, DH, HeD, LiH = _get_deviations(data, obs)

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
    fig, ax = _init_figure()
    # ...set the tick labels
    _set_tick_labels(ax, x, y)

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
