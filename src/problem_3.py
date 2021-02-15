from events import *
from plotting import *
from multiprocessing import Pool

def problem_3(v_0,N,count = 20,seed = 42, xi = 1, dt = 0.01):

    np.random.seed(seed)

    gas      = Gas(N,0.001)
    theta    = np.random.random(N) * 2 * np.pi
    v        = v_0 * np.array([np.cos(theta),np.sin(theta)])

    gas.set_velocities(v)
    gas.xi = xi
    
    mid = N//2
    gas.M[mid:] *= 4

    gas.simulate_saveE(dt = dt, stopper = "equilibrium", stop_val = count )

    return gas.E_avg

def problem_3_loop(v_0,N,count = 20, dt = 0.001):

    pool = Pool()
    res1 = pool.apply_async(problem_3, [v_0,N,count,1,1.0,dt])
    res2 = pool.apply_async(problem_3, [v_0,N,count,2,0.9,dt])
    res3 = pool.apply_async(problem_3, [v_0,N,count,3,0.8,dt])

    ans1 = res1.get(timeout = None)
    ans2 = res2.get(timeout = None)
    ans3 = res3.get(timeout = None)
    
    problem_3_plot(ans1,ans2,ans3,dt)

def problem_3_plot(E1,E2,E3,dt):

    fig, ax = plt.subplots(nrows = 3, figsize = (13,18))

    T1 = np.arange(0,np.size(E1[:,0])) * dt 
    T2 = np.arange(0,np.size(E2[:,0])) * dt
    T3 = np.arange(0,np.size(E3[:,0])) * dt

    ax[0].set_title(r"$\xi = 1.0$")
    ax[0].plot(T1,E1[:,0], label =r"mass = $m$", color = "blue")
    ax[0].plot(T1,E1[:,1], label =r"mass = $4m$", color = "red")
    ax[0].plot(T1,E1[:,2], label =r"All particles", color = "green")
    ax[0].set_xlabel(r"$t$")
    ax[0].set_ylabel(r"$\left\langle T \right\rangle$")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].grid(ls = "--")
    plt.tight_layout()

    ax[1].set_title(r"$\xi = 0.9$")
    ax[1].plot(T2,E2[:,0], label =r"mass = $m$", color = "blue")
    ax[1].plot(T2,E2[:,1], label =r"mass = $4m$", color = "red")
    ax[1].plot(T2,E2[:,2], label =r"All particles", color ="green")
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$\left\langle T \right\rangle$")
    ax[1].set_yscale("log")
    ax[1].legend()
    ax[1].grid(ls = "--")
    plt.tight_layout()

    ax[2].set_title(r"$\xi = 0.8$")
    ax[2].plot(T3,E3[:,0], label =r"mass = $m$", color = "blue")
    ax[2].plot(T3,E3[:,1], label =r"mass = $4m$", color = "red")
    ax[2].plot(T3,E3[:,2], label =r"All particles", color = "green")
    ax[2].set_xlabel(r"$t$")
    ax[2].set_ylabel(r"$\left\langle T \right\rangle$")
    ax[2].set_yscale("log")
    ax[2].grid(ls = "--")
    ax[2].legend()
    
    plt.tight_layout()

    plt.savefig("../fig/energy_avg.pdf")

import sys
    
if __name__ == "__main__":

    label = int(sys.argv[1])

    E1 = problem_3(v_0 = 2,N = 4000, count = 20, seed = label, dt = 0.001, xi = 1)
    E2 = problem_3(v_0 = 2,N = 4000, count = 20, seed = label, dt = 0.001, xi = 0.8)
    E3 = problem_3(v_0 = 2,N = 4000, count = 20, seed = label, dt = 0.001, xi = 0.9)

    np.save(f"../data/prob3/E1_{label}.npy",E1)
    np.save(f"../data/prob3/E2_{label}.npy",E2)
    np.save(f"../data/prob3/E3_{label}.npy",E3)        
