#! /usr/bin/python3

# sys
import sys; sys.path.append('..')
# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter

# acropolis
from acropolis.params import NY

# Set the font
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

# Sepcify some properties of the figure
fig = plt.figure(figsize=(0.4*12.0, 0.4*11.0), dpi=150, edgecolor="white")
ax = fig.add_subplot(1,1,1)
ax.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)

# Deint the ticks for the x-...
xtMajor = [np.log10(10**j) for j in np.linspace(3, 10, 8)]
xtMinor = [np.log10(i*10**j) for j in xtMajor for i in range(10)[1:10]]
xlMajor = [r"$10^{" + str(int(i)) + "}$" if i in xtMajor else "" for i in xtMajor]
xMajorLocator = FixedLocator(xtMajor)
xMinorLocator = FixedLocator(xtMinor)
xMajorFormatter = FixedFormatter(xlMajor)
# ... and y-axis
ytMajor = np.linspace(-15, -3, 13)
ytMinor = [np.log10(i*10**(j)) for i in range(10)[1:] for j in ytMajor]
ylMajor = [r"$10^{" + str(int(i)) + "}$" if i in ytMajor else "" for i in ytMajor]
yMajorLocator = FixedLocator(ytMajor)
yMinorLocator = FixedLocator(ytMinor)
yMajorFormatter = FixedFormatter(ylMajor)

# Function to exteract the abundances for the data file
def get_abd(data, i):
    i0 = i + 2

    # Get the length of the data
    N = len( data )

    # Extract the different abundances...
    mean, high, low = data[:,i0], data[:,i0+NY], data[:,i0+2*NY]
    # ...and calculate an estimate for the error
    diff = np.minimum( np.abs( mean - high ), np.abs( mean - low ) )

    return mean, diff

shape=(200, 200)

# Read the data file
data = np.loadtxt(f"data/decay_50MeV.dat")
tau  = data[:,0].reshape(shape)
n0a  = data[:,1].reshape(shape)

# Extract the different abundances
mn, en = get_abd(data, 0)
mp, ep = get_abd(data, 1)
mH, eH = mn + mp, np.sqrt( en**2. + ep**2. )

mD  , eD   = get_abd(data, 2)
mT  , eT   = get_abd(data, 3)
mHe3, eHe3 = get_abd(data, 4)
mHe4, eHe4 = get_abd(data, 5)
m3  , e3   = mT + mHe3, np.sqrt( eT**2. + eHe3**2. )
mLi7, eLi7 = get_abd(data, 7)
mBe7, eBe7 = get_abd(data, 8)
m7,   e7   = mLi7 + mBe7, np.sqrt( eLi7**2. + eBe7**2. )

mYp , eYp  = 4.*mHe4, 4.*eHe4
mDH , eDH  = mD/mH, (mD/mH)*np.sqrt( (eD/mD)**2. + (eH/mH)**2. )
mHeD, eHeD = m3/mD, (m3/mD)*np.sqrt( (e3/m3)**2. + (eD/mD)**2. )
mLiH, eLiH = m7/mH, (m7/mH)*np.sqrt( (e7/m7)**2. + (eH/mH)**2. )

# Calculate the deviations...
Yp  = (mYp - 2.45e-1) / np.sqrt((0.03e-1)**2 + eYp**2)
DH  = (mDH - 2.547e-5) / np.sqrt((0.035e-5)**2 + eDH**2)
HeD = (mHeD - 8.3e-1) / np.sqrt((1.5e-1)**2 + eHeD**2.)
LiH = (mLiH - 1.6e-10) / np.sqrt((0.3e-10)**2. + eLiH**2.)
HeD[mDH<0.035e-5] = 10
# ...and reshape
Yp  = Yp.reshape(shape)
DH  = DH.reshape(shape)
HeD = HeD.reshape(shape)
LiH = LiH.reshape(shape)

HeD[(tau>1e7)*(n0a>1e-8)*(n0a<1e-6)] = 10
DH[np.isnan(DH)] = -10

# Extract the overall exclusion line
max = np.maximum( np.abs(DH), np.abs(Yp) )
max = np.maximum( max, HeD )

sig = 1.95996

plt.contourf(np.log10(tau), np.log10(n0a), DH, levels=[-1e10, -sig, sig, 1e10], colors=["0.6","white", "tomato"], alpha=0.2)
plt.contourf(np.log10(tau), np.log10(n0a), Yp, levels=[-1e10, -sig, sig, 1e10], colors=["dodgerblue","white", "lightcoral"], alpha=0.2)
plt.contourf(np.log10(tau), np.log10(n0a), HeD, levels=[sig, 1e10], colors=["mediumseagreen"], alpha=0.2)

# plt.contour(np.log10(tau), np.log10(n0a), LiH, levels=[0], colors='#fe46a5', linestyles='--')
# plt.contour(np.log10(tau), np.log10(n0a), LiH, levels=[-sig], colors='#fe46a5', linestyles='-')
# plt.contour(np.log10(tau), np.log10(n0a), LiH, levels=[+sig], colors='#fe46a5', linestyles='-')

plt.contour(np.log10(tau), np.log10(n0a), DH, levels=[-sig], colors='0.6', linestyles='-')
plt.contour(np.log10(tau), np.log10(n0a), DH, levels=[sig], colors='tomato', linestyles='-')
plt.contour(np.log10(tau), np.log10(n0a), Yp, levels=[-sig], colors='dodgerblue', linestyles='-')
plt.contour(np.log10(tau), np.log10(n0a), HeD, levels=[sig], colors='mediumseagreen', linestyles='-')
plt.contour(np.log10(tau), np.log10(n0a), max, levels=[sig], colors='black', linewidths=1.5, linestyles='-')

plt.text(4, -10, r"D/$^1$H low", color='0.6', fontsize=16)
plt.text(7.5, -11, r"$^3$He/D high", color='mediumseagreen', fontsize=16)
plt.text(6.3, -4.2, r"$\mathcal{Y}_\text{p}$ low", color='dodgerblue', fontsize=16)
plt.text(7.3, -8, r"D/$^1$H high", color='tomato', fontsize=16)

ax.xaxis.set_label_text(r"$\tau_\phi\;[\mathrm{s}]$")
ax.xaxis.set_major_locator(xMajorLocator)
ax.xaxis.set_minor_locator(xMinorLocator)
ax.xaxis.set_major_formatter(xMajorFormatter)
ax.set_xlim(3, 10)

ax.yaxis.set_label_text(r"$(n_\phi/n_\gamma)|_{T=T_0}$")
ax.yaxis.set_major_locator(yMajorLocator)
ax.yaxis.set_minor_locator(yMinorLocator)
ax.yaxis.set_major_formatter(yMajorFormatter)
ax.set_ylim(-15, -3)

plt.title(r"$m_\phi = 50\,\mathrm{MeV},\;T_0=10\,\mathrm{MeV},\;\text{BR}_{\gamma\gamma}=1-\text{BR}_{e^+e^-} = 1$", fontsize=11)
plt.tight_layout()
plt.savefig(f"decay_50MeV.pdf")
plt.show()
