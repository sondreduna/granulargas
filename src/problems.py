from multiprocessing import Pool
from events import *
from plotting import *
import time as time

def test_function(N):

    x = np.random.random((2,N))
    y = np.random.random((2,N))
    np.einsum('ij,ij->j',x,y)


def problem_1(v_0,N,T,seed = 42):

    np.random.seed(seed)
    
    ensemble = Ensemble(N,0.0005)
    theta = np.random.random(N) * 2* np.pi
    v = v_0 * np.array([np.cos(theta),np.sin(theta)])
    
    ensemble.set_velocities(v)

    ensemble.simulate(T, dt = 1)
    kT = ensemble.kT()
    return ensemble
    
def problem_1_simple(v_0, N = 2000, T = 100):

    ensemble = problem_1(v_0,N,T)

    ensemble.plot_velocity_distribution(r"\textbf{Final distribution}",
                                        "../fig/dist.pdf",
                                        compare = True)

def problem_1_plot(ensemble):

    fig, ax = plt.subplots(ncols = 2, figsize = (22,7))

    v_0 = ensemble.v_0
    v_abs   = np.sqrt(ensemble.get_v_square())
    v_0_abs = np.sqrt(np.einsum('ij,ij->j',v_0,v_0))
    
    v = np.linspace(0,np.max(v_abs),1000)
    kT = ensemble.kT()

    xlim = [np.min(v_abs), np.max(v_abs)]
    
    ax[0].set_title(r"Initial velocity distribution")
    
    sns.histplot(v_0_abs,stat = "density", color = "blue", ax = ax[0])
    ax[0].grid(ls = "--")
    ax[0].set_xlabel(r"$v$")
    ax[0].set_ylabel(r"Particle density")
    ax[0].set_xlim(xlim)
    plt.tight_layout()

    ax[1].set_title(r"Final velocity distribution")

    sns.histplot(v_abs, stat = "density", color = "blue", ax = ax[1])
    ax[1].plot(v,boltzmann_dist(kT,ensemble.M[0],v),
             label = r"$p(v) = \frac{mv}{kT} \exp{\left(-\frac{m v^2}{2kT}\right)}$",
             color = "black",
             ls = "--")

    ax[1].grid(ls = "--")

    ax[1].set_xlabel(r"$v$")
    ax[1].set_ylabel(r"Particle density")
    
    plt.legend()
    plt.tight_layout()

    fig.savefig("../fig/distribution.pdf")


def deviation_plot(ensemble):

    fig = plt.figure()

    v_abs = np.sqrt(ensemble.get_v_square())

    ax = sns.histplot(v_abs,stat = "density", color = "blue")

    
    
    
def problem_1_para(v_0,N = 2000,T = 100):

    pool = Pool()

    results = [pool.apply_async(problem_1, [v_0,N,T,i]) for i in range(8)]
    answers = [results[i].get(timeout = None) for i in range(8)]

    sum_ensemble = Ensemble(1)
    sum_ensemble.N = 8*N
    sum_ensemble.M = np.full(8*N,answers[0].M[0])

    sum_ensemble.v_0       = np.concatenate([answers[i].v_0 for i in range(8)], axis = 1)
    sum_ensemble.particles = np.concatenate([answers[i].particles for i in range(8)], axis = 1)

    problem_1_plot(sum_ensemble)


    
def crater(N):

    r = np.sqrt(1/(4*(N-1)*np.pi))
    R = 5*r 

    m = 1
    M = 25

    x = np.arange(1.5*r, 1 - 1.5*r,3*r)
    y = np.arange(1.5*r, 0.5, 3*r)

    nx = np.size(x)
    ny = np.size(y)
    
    ensemble = Ensemble(N,r)
    ensemble.radii[0] = R
    ensemble.M[0] = M
    ensemble.M[1:] = m
    ensemble.xi = 1
    ensemble.particles[:,0] = np.array([0.5,0.75,0,-5])
    
    ensemble.simulate_savefigs(1,0.01,False)
    #ensemble.simulate(1,0.05,False)
    ensemble.plot_positions("../fig/crater.pdf")
