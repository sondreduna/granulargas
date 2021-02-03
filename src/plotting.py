# importing useful packages

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import rc
import heapq 
import seaborn as sns

# Setting common plotting parameters

rc("text",usetex = True)
rc("font",family = "sans-serif")

fontsize = 25
newparams = {'axes.titlesize': fontsize,
             'axes.labelsize': fontsize,
             'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 
             'legend.fontsize': fontsize,
             'figure.titlesize': fontsize,
             'legend.handlelength': 1.5, 
             'lines.linewidth': 2,
             'lines.markersize': 7,
             'figure.figsize': (11, 7), 
             'figure.dpi':200,
             'text.usetex' : True,
             'font.family' : "sans-serif"
            }

plt.rcParams.update(newparams)

def boltzmann_dist(kT,m,v):
    return m*v/kT * np.exp(- m * v**2 /(2*kT))

def _plot_velocity_distribution(ensemble,title,savefig = "", compare = False):

    v_abs = np.sqrt(ensemble.get_v_square())

    fig = plt.figure()

    plt.title(title)

    sns.histplot(v_abs, stat = "density", color = "blue")
    
    if compare:

        v = np.linspace(0,np.max(v_abs),1000)
        kT = ensemble.kT()
        
        plt.plot(v,boltzmann_dist(kT,ensemble.M[0],v),
                 label = r"$p(v) = \frac{mv}{kT} \exp{\left(-\frac{m v^2}{2kT}\right)}$",
                 color = "black",
                 ls = "--")

    plt.grid(ls = "--")

    plt.xlabel(r"$v$")
    plt.ylabel(r"Particle density")
    
    plt.legend()
    plt.tight_layout()

    if savefig != "":
        fig.savefig(savefig)

def _plot_positions(ensemble,savefig = ""):
        

    fig, ax = plt.subplots(figsize = (8,8))
    cm = plt.get_cmap("viridis")
    
    for i in range(ensemble.N):
        ax.add_artist(plt.Circle((ensemble.particles[0,i],
                                ensemble.particles[1,i]),
                                ensemble.radii[i],
                                linewidth=0,
                                color = cm(1/(1 + ensemble.count[i]))))

    # boundary of box
    plt.hlines([0,1],[0,0],[1,1], ls = "--", color = "black")
    plt.vlines([0,1],[0,0],[1,1], ls = "--", color = "black")

    plt.grid(ls = "--")

    plt.tight_layout()

    if savefig != "":
        fig.savefig(savefig)

        plt.close()

def _plot_energy(ensemble, savefig = ""):

    fig = plt.figure()

    plt.plot(ensemble.E)
    plt.xlabel("Time")
    plt.ylabel("Energy")

    plt.tight_layout()

    if savefig != "":
        fig.savefig(savefig)

        plt.close()
        
