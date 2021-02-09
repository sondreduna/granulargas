from multiprocessing import Pool
from events import *
from plotting import *
import time as time

from scipy.stats import gaussian_kde

def problem_1(v_0,N,count,seed = 42):

    np.random.seed(seed)
    
    gas = Gas(N,0.0005)
    theta = np.random.random(N) * 2* np.pi
    v = v_0 * np.array([np.cos(theta),np.sin(theta)])
    
    gas.set_velocities(v)

    gas.simulate(dt = 1,stopper = "equilibrium",stop_val = count)

    return gas, v
    
def problem_1_simple(v_0, N = 2000, count = 50):

    gas, v = problem_1(v_0,N)

    gas.plot_velocity_distribution(r"\textbf{Final distribution}",
                                        "../fig/dist.pdf",
                                        compare = True)
    deviation_plot(gas)
    
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
