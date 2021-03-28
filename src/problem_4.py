from events import *
from plotting import *

from mpltools import annotation  # The special package mpltools may need to be installed
                                 # To get fancy slope marker

def distribute_balls(N,max_iter = 10000):

    """
    Function for distributing balls in 

    (x,y) \in [0,1] x [0,0.5] with r removed from the boundaries

    """

    X = np.zeros((2,N))
    r = (-1 + np.sqrt(np.pi*N))/(2*(N*np.pi - 1))

    X[:,0] = np.array([np.random.uniform(r,1-r),
                       np.random.uniform(r,0.5)])
    
    for i in range(1,N):
        X[:,i] = np.array([np.random.uniform(r,1-r),
                           np.random.uniform(r,0.5)])

        dists = np.sqrt(np.sum((X[:,:i] - X[:,i][:,None])**2, axis = 0))

        it = 0
        while np.size(dists[dists <= 2*r]) != 0:
            X[:,i] = np.array([np.random.uniform(r,1-r),
                               np.random.uniform(r,0.5)])
            it+=1
            dists = np.sqrt(np.sum((X[:,:i] - X[:,i][:,None])**2, axis = 0))
            if it > max_iter:
                print("Distribution of balls failed")
                return X,r
    return X,r 
    

def crater_sim(N,xi = 0.5, M = 10, path = "../fig/crater.pdf"):

    m = 1
    M = M

    X,r = distribute_balls(N)
    R = 20* r 
    
    gas = Gas(N+1,r)
    gas.radii[0] = R
    gas.M[0] = M
    gas.M[1:] = m
    gas.xi = xi

    gas.particles[:,0] = np.array([0.5,0.75,0,-1])
    gas.particles[:2,1:] = X
    
    gas.simulate(dt = 1, stopper = "dissipation", stop_val = 0.9, ret_vels = False)
    #gas.simulate_savefigs(1, 0.005, verbose = False, videoname = "../fig/crater")
    gas.plot_positions(path)
    
def crater(N,xi = 0.5,M = 25, v = 5):    
    m = 1
    M = M

    X,r = distribute_balls(N)
    R = 5 * r 
    
    gas = Gas(N+1,r)
    gas.radii[0] = R
    gas.M[0] = M
    gas.M[1:] = m
    gas.xi = xi

    gas.particles[:,0] = np.array([0.5,0.75,0,-v])
    gas.particles[:2,1:] = X

    gas.simulate(dt = 1, stopper = "dissipation", stop_val = 0.9, ret_vels = False)

    # the positions of all the particles
    x = gas.particles[:2,:]

    # plot the distribution of particles, for debugging 
    gas.plot_positions("../fig/crater.pdf")

    # number of particles involved in crater formation:  
    affected = np.count_nonzero(gas.count)
    return affected

def double_crater_test():
    m1 = 5
    m2 = 25

    crater_sim(4000,M = m1,path = "../fig/crater_1.pdf")
    crater_sim(4000,M = m2,path = "../fig/crater_2.pdf")

def mass_scan(num_masses = 10,label = 0,min_mass = 1, max_mass = 25):
    """
    Scan over M masses

    """

    M = np.linspace(min_mass,max_mass,num_masses)
    size = np.zeros(num_masses)

    for i, m_i in enumerate(M):
        size[i] = crater(2000,xi = 0.5, M = m_i)    

    np.save(f"../data/prob4/M_{label}.npy",M)
    np.save(f"../data/prob4/size_{label}.npy",size)

def vel_scan(num_vels = 10, label = 0):

    V = np.logspace(-1,2,num_vels)
    size = np.zeros(num_vels)

    for i, v_i in enumerate(V):
        size[i] = crater(2000,xi = 0.5, M = 20, v = v_i)

    np.save(f"../data/prob4/v_{label}.npy",V)
    np.save(f"../data/prob4/vs_{label}.npy",size)
    
def problem_4_plot():

    
    M = np.mean([np.load(f"../data/prob4/M_{i}.npy") for i in range(1,9)], axis = 0)
    size = np.mean([np.load(f"../data/prob4/size_{i}.npy") for i in range(1,9)], axis = 0)

    fig = plt.figure()
    
    plt.plot(M,size,label = r"$\mathcal{S}(m)$", ls = "--", marker = "o", color ="blue")
    plt.xlabel(r"mass $m$")
    plt.ylabel(r"Size of crater $\mathcal{S}$")

    plt.legend()
    plt.grid(ls = "--")
    plt.tight_layout()

    fig.savefig("../fig/mass_size.pdf")

    V    = np.mean([np.load(f"../data/prob4/v_{i}.npy") for i in range(1,9)], axis = 0)
    size = np.mean([np.load(f"../data/prob4/vs_{i}.npy") for i in range(1,9)], axis = 0)
    
    fig2, ax = plt.subplots()

    plt.plot(V,size,label = r"$\mathcal{S}(v)$", ls = "--", marker = "o", color ="blue")
    plt.xlabel(r"velocity $v$")
    plt.ylabel(r"Size of crater $\mathcal{S}$")

    annotation.slope_marker((5,500), (2,1), invert=True, 
                ax=ax,text_kwargs={"fontsize" : 22})

    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(ls = "--")
    plt.tight_layout()

    fig2.savefig("../fig/vel_size.pdf")

import sys

if __name__ == "__main__":

    label = int(sys.argv[1])
    #mass_scan(10,label,1,25)
    vel_scan(10,label)

    

