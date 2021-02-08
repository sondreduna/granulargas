from events import *
from plotting import *

def problem_3(v_0,N,count = 20,seed = 42, xi = 1, dt = 0.01):

    np.random.seed(seed)

    ensemble = Ensemble(N,0.0005)
    theta    = np.random.random(N) * 2 * np.pi
    v        = v_0 * np.array([np.cos(theta),np.sin(theta)])

    ensemble.set_velocities(v)
    ensemble.xi = xi
    
    mid = N//2
    ensemble.M[mid:] *= 4

    ensemble.simulate_saveE(dt = 0.01, stopper = "equilibrium", stop_val = count )

    return ensemble.E_avg

def problem_3_loop(v_0,N,count = 20, seed = 42, dt = 0.01):

    E1 = problem_3(v_0,N,count,seed,1.,dt)
    E2 = problem_3(v_0,N,count,seed,0.9,dt)
    E3 = problem_3(v_0,N,count,seed,0.8,dt)
    
    problem_3_plot(E1,E2,E3,dt)    
        
def problem_3_plot(E1,E2,E3,dt):

    fig, ax = plt.subplots(nrows = 3, figsize = (13,18))

    T1 = np.arange(0,np.size(E1[:,0])) * dt 
    T2 = np.arange(0,np.size(E2[:,0])) * dt
    T3 = np.arange(0,np.size(E3[:,0])) * dt

    ax[0].set_title(r"$\xi = 1.0$")
    ax[0].plot(T1,E1[:,0], label =r"mass = $m$")
    ax[0].plot(T1,E1[:,1], label =r"mass = $4m$")
    ax[0].plot(T1,E1[:,2], label =r"All particles")
    ax[0].set_xlabel(r"$t$")
    ax[0].set_ylabel(r"$\left\langle T \right\rangle$")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].grid(ls = "--")
    plt.tight_layout()

    ax[1].set_title(r"$\xi = 0.9$")
    ax[1].plot(T2,E2[:,0], label =r"mass = $m$")
    ax[1].plot(T2,E2[:,1], label =r"mass = $4m$")
    ax[1].plot(T2,E2[:,2], label =r"All particles")
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$\left\langle T \right\rangle$")
    ax[1].set_yscale("log")
    ax[1].legend()
    ax[1].grid(ls = "--")
    plt.tight_layout()

    ax[2].set_title(r"$\xi = 0.8$")
    ax[2].plot(T3,E3[:,0], label =r"mass = $m$")
    ax[2].plot(T3,E3[:,1], label =r"mass = $4m$")
    ax[2].plot(T3,E3[:,2], label =r"All particles")
    ax[2].set_xlabel(r"$t$")
    ax[2].set_ylabel(r"$\left\langle T \right\rangle$")
    ax[2].set_yscale("log")
    ax[2].grid(ls = "--")
    ax[2].legend()
    
    plt.tight_layout()

    plt.savefig("../fig/energy_avg.pdf")
    
    
    

    
