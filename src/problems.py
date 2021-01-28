from multiprocessing import Pool
from events import *
import time as time

def test_function(N):

    x = np.random.random((2,N))
    y = np.random.random((2,N))
    np.einsum('ij,ij->j',x,y)

def problem_1(v_0, N = 2000, T = 100):

    ensemble = Ensemble(N,0.0005)
    theta = np.random.random(N) * 2* np.pi
    v = v_0 * np.array([np.cos(theta),np.sin(theta)])

    ensemble.set_velocities(v)

    ensemble.simulate(T, dt = 1)
    ensemble.plot_velocity_distribution(r"\textbf{Final distribution}", "../fig/dist.pdf",compare = True)
    
def problem_1_para(v_0):

    pool = Pool()

    N = 1000
    T = 1000

    
    Ensembles = np.array([Ensemble(N,0.001) for i in range(8)])

    for i in range(8):
        theta = np.random.random(N) * 2* np.pi
        v = v_0 * np.array([np.cos(theta),np.sin(theta)])
        Ensembles[i].set_velocities(v)
    
    results   = [pool.apply_async(Ensembles[i].simulate , [T] ) for i in range(8)]
    answers   = [results[i].get(timeout = None)]
    
