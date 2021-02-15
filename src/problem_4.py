from events import *
from plotting import *

def distribute_balls(N):

    """
    Function for distributing balls in 

    (x,y) \in [0,1] x [0,0.5]

    """

    X = np.zeros((2,N))
    r = 1/2 * np.sqrt(1/(np.pi*N))
    
    for i in range(N):
        X[:,i] = np.array([np.random.uniform(1.5*r,1-1.5*r),
                           np.random.uniform(1.5*r,0.5-1.5*r)])

        dists = np.sqrt(np.sum((X[:,:i] - X[:,i][:,None])**2, axis = 0))
        
        while np.size(dists[dists <= 2*r]) != 0:
            X[:,i] = np.array([np.random.uniform(1.5*r,1-1.5*r),
                           np.random.uniform(1.5*r,0.5-1.5*r)])

    return X
    

