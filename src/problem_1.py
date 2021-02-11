from multiprocessing import Pool
from events import *
from plotting import *
import time as time

from scipy.stats import gaussian_kde, chisquare
from scipy.integrate import quad

def problem_1(v_0,N,count,seed = 42):

    np.random.seed(seed)
    
    gas = Gas(N,0.0005)
    theta = np.random.random(N) * 2* np.pi
    v = v_0 * np.array([np.cos(theta),np.sin(theta)])
    
    gas.set_velocities(v)

    gas.simulate(dt = 0.005,stopper = "equilibrium",stop_val = count)

    return gas, v
    
def problem_1_simple(v_0, N = 2000, count = 50,i = 0):

    gas, v = problem_1(v_0,N,count,int(i))

    np.save(f"../data/prob1/speed_{i}.npy",gas.speeds)
    np.save(f"../data/prob1/times_{i}.npy",gas.times)
    np.save(f"../data/prob1/v_0_{i}",v)
    
def problem_1_plot(gas):

    fig, ax = plt.subplots(ncols = 2, figsize = (20,7))

    v_0     = gas.v_0
    v_0_abs = np.sqrt(np.einsum('ij,ij->j',v_0,v_0))
    v_abs   = np.sqrt(gas.get_v_square())
        
    v = np.linspace(0,np.max(v_abs),1000)
    kT = gas.kT()

    xlim = [np.min(v_abs), np.max(v_abs)]
    
    ax[0].set_title(r"Initial velocity distribution")
    v_0_abs[0] *= 2  # hacky solution to make the histplot work
    sns.histplot(v_0_abs,stat = "density", color = "blue", ax = ax[0], edgecolor = None)
    ax[0].grid(ls = "--")
    ax[0].set_xlabel(r"$v$")
    ax[0].set_ylabel(r"Particle density")
    ax[0].set_xlim(xlim)
    plt.tight_layout()

    ax[1].set_title(r"Final velocity distribution")

    sns.histplot(v_abs, stat = "density", color = "blue", ax = ax[1],edgecolor = None)
    ax[1].plot(v,boltzmann_dist(kT,gas.M[0],v),
             label = r"$p(v) = \frac{mv}{kT} \exp{\left(-\frac{m v^2}{2kT}\right)}$",
             color = "black",
             ls = "--")

    ax[1].grid(ls = "--")

    ax[1].set_xlabel(r"$v$")
    ax[1].set_ylabel(r"Particle density")
    
    plt.legend()
    plt.tight_layout()

    fig.savefig("../fig/distribution.pdf")

    counts, bins = np.histogram(v_abs)
    bins = (bins[:-1] + bins[1:]) / 2
    exp_counts = boltzmann_dist(kT,gas.M[0],bins)

    chisq, p = chisquare(f_obs=counts, f_exp=exp_counts,ddof = 2)
    err = error(gas)
    np.savetxt("../data/p_value.txt",np.array([p,err]))


def deviation_plot(gas):

    fig = plt.figure()

    v_abs = np.sqrt(gas.get_v_square())
    v = np.linspace(np.min(v_abs),np.max(v_abs),1000)

    kernel = gaussian_kde(v_abs,'scott')

    fig = plt.figure()
    plt.plot(v,np.abs(kernel(v) - boltzmann_dist(gas.kT(),gas.M[0],v)), label = r"$|p(v) - \hat{p}(v)|$")

    plt.grid(ls = "--")
    plt.xlabel(r"$v$")
    plt.ylabel(r"$\texttt{err}$")

    plt.yscale("log")
    
    plt.legend()
    plt.tight_layout()
    fig.savefig("../fig/kde_diff.pdf")
    
def problem_1_para(v_0,N = 2000, count = 50):

    pool = Pool()

    results = [pool.apply_async(problem_1, [v_0,N,count,i]) for i in range(8)]
    answers = [results[i].get(timeout = None) for i in range(8)]

    sum_gas = Gas(1)
    sum_gas.N = 8*N
    sum_gas.M = np.full(8*N,answers[0][0].M[0])

    sum_gas.v_0       = np.concatenate([answers[i][1] for i in range(8)], axis = 1)
    sum_gas.particles = np.concatenate([answers[i][0].particles for i in range(8)], axis = 1)

    problem_1_plot(sum_gas)
    deviation_plot(sum_gas)


def error(gas):

    kT = gas.kT()
    v_abs   = np.sqrt(gas.get_v_square())
    m = gas.M[0]

    densities, bins = np.histogram(v_abs, density = True, bins = "auto")

    bins_centre = (bins[:-1] + bins[1:]) / 2
    width = bins[1]-bins[0]

    probs = np.array([quad(boltzmann_dist ,bins[i],bins[i+1], args = (kT,m))[0] for i in range(len(bins)-1)])
    
    err = 1/2 * np.sum(np.abs(densities - probs))
    return err     

def problem_1_dev(v_0,count = 50):

    Ns = np.arange(1000,50000,4000)
    Ns_para = np.array([i//8 for i in Ns])

    err = np.zeros(np.shape(Ns_para))

    for i,N in enumerate(Ns_para):
        pool = Pool()

        results = [pool.apply_async(problem_1, [v_0,N,count,i]) for i in range(8)]
        answers = [results[i].get(timeout = None) for i in range(8)]

        sum_gas = Gas(1)
        sum_gas.N = 8*N
        sum_gas.M = np.full(8*N,answers[0][0].M[0])

        sum_gas.v_0       = np.concatenate([answers[i][1] for i in range(8)], axis = 1)
        sum_gas.particles = np.concatenate([answers[i][0].particles for i in range(8)], axis = 1)
            
        err[i] = error(sum_gas)

    fig = plt.figure(figsize = (13,7))

    plt.plot(Ns,err,color = "blue",ls = "--",marker = "v",label = r"$\texttt{err}(N)$")

    plt.yscale("log")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\texttt{err}$")

    plt.grid(ls = "--")
    plt.legend()
    plt.tight_layout()

    fig.savefig("../fig/err.pdf")


""" 
def crater(N):

    r = np.sqrt(1/(4*(N-1)*np.pi))
    R = 5*r 

    m = 1
    M = 25

    x = np.arange(1.5*r, 1 - 1.5*r,3*r)
    y = np.arange(1.5*r, 0.5, 3*r)

    nx = np.size(x)
    ny = np.size(y)
    
    gas = Gas(N,r)
    gas.radii[0] = R
    gas.M[0] = M
    gas.M[1:] = m
    gas.xi = 1
    gas.particles[:,0] = np.array([0.5,0.75,0,-5])
    
    gas.simulate_savefigs(1,0.01,False)
    #gas.simulate(1,0.05,False)
    gas.plot_positions("../fig/crater.pdf")
"""

import sys

if __name__ == "__main__":

    label = sys.argv[1]
    problem_1_simple(v_0 = 1, N = 4000, count = 50, i = label)
