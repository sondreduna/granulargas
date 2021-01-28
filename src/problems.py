from multiprocessing import Pool
from events import *
import time as time

def test_function(N):

    x = np.random.random((2,N))
    y = np.random.random((2,N))
    np.einsum('ij,ij->j',x,y)


def problem_1(v_0,N,T):
    ensemble = Ensemble(N,0.0005)
    theta = np.random.random(N) * 2* np.pi
    v = v_0 * np.array([np.cos(theta),np.sin(theta)])

    ensemble.set_velocities(v)

    ensemble.simulate(T, dt = 1)
    kT = ensemble.kT()
    return ensemble, kT
    
def problem_1_simple(v_0, N = 2000, T = 100):

    ensemble = problem_1(v_0,N,T)

    ensemble.plot_velocity_distribution(ensemble.kT(),
                                        r"\textbf{Final distribution}",
                                        "../fig/dist.pdf",
                                        compare = True)
    
def problem_1_para(v_0,N = 2000,T = 100):

    pool = Pool()

    results = [pool.apply_async(problem_1, [v_0,N,T]) for i in range(8)]
    answers = [results[i].get(timeout = None) for i in range(8)]

    sum_ensemble = Ensemble(1)
    sum_ensemble.N = 8*N

    sum_ensemble.particles = np.concatenate([answers[i][0].particles for i in range(8)], axis = 1)


    kT = np.average([answers[i][1] for i in range(8)])
    sum_ensemble.plot_velocity_distribution(kT,r"\textbf{Final distribution}",
                                        "../fig/dist.pdf",
                                        compare = True)
    
    
