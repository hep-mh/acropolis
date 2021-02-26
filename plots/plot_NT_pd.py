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

# Function to exteract the abundances for the data file
def get_abd(data, i):
    i0 = i + 1

    # Get the length of the data
    N = len( data )

    # Extract the different abundances...
    mean, high, low = data[:,i0], data[:,i0+NY], data[:,i0+2*NY]

    return mean, high, low

data = np.loadtxt("data/NT_pd.dat")

NT = data[:,0]
N = len(NT)

mD, hD, lD = get_abd(data, 2)

plt.plot([50]*2, [1, 2], color="black", linestyle="--")

for i in range(N):
    plt.plot(NT[i], mD[i]*1e5, '*', color="dodgerblue")
    plt.plot(NT[i], hD[i]*1e5, '*', color="crimson")
    plt.plot(NT[i], lD[i]*1e5, '*', color="mediumseagreen")

plt.text(160, 1.794, "low", color="mediumseagreen")
plt.text(160, 1.751, "mean", color="dodgerblue")
plt.text(160, 1.708, "high", color="crimson")

plt.text(135, 1.83, r"$\texttt{NE\_pd=150}$")


plt.xlabel(r"$\texttt{NT\_pd}$")
plt.xlim(0, 200)

plt.ylabel(r"$Y_\text{D}\;[\times 10^5]$")
plt.ylim(1.65, 1.85)

plt.title(r"$m_\phi = 50\,\mathrm{MeV},\;\tau_\phi = 10^5\,\mathrm{s},\;(n_\phi/n_\gamma)|_{T=10\,\mathrm{MeV}} = 10^{-8},\;\text{BR}_{\gamma\gamma}=1$", fontsize=10)
plt.tight_layout()
plt.savefig("NT_pd.pdf")
plt.show()
