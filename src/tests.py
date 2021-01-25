from events import *

def one_particle_test_1():

    collection = Ensemble(1)

    collection.particles[2:,0] = np.array([1,1])

    collection.plot_positions()

    collection.simulate_savefigs(10,0.05,False)

def one_particle_test_2():

    collection = Ensemble(1)
    collection.xi = 0
    collection.particles[2:,0] = np.array([1,1])

    collection.plot_positions()

    collection.simulate_savefigs(10,0.05,False)


def two_particle_test_1():

    collection = Ensemble(2)

    collection.particles[:2,0] = np.array([0.2,0.5])
    collection.particles[:2,1] = np.array([0.6,0.5])

    collection.particles[2:,0] = np.array([1.,0])
    collection.particles[2:,1] = np.array([-1,0])

    collection.simulate_savefigs(10,0.05,True)

    print(collection.particles)
    
def two_particle_test_2():

    collection = Ensemble(2)
    collection.radii[0] = 0.02
    
    b = sum(collection.radii)/np.sqrt(2)
    
    collection.particles[:2,0] = np.array([0.2,0.5])
    collection.particles[:2,1] = np.array([0.6,0.5 + b])

    collection.particles[2:,0] = np.array([1.,0])
    collection.particles[2:,1] = np.array([-1,0])

    collection.simulate_savefigs(10,0.05,False)
    
    print(collection.particles)

def two_particle_test_3():

    collection = Ensemble(2)
    collection.xi = 0

    collection.particles[:2,0] = np.array([0.2,0.5])
    collection.particles[:2,1] = np.array([0.6,0.5])

    collection.particles[2:,0] = np.array([1.,0])
    collection.particles[2:,1] = np.array([-1.,0])

    collection.simulate_savefigs(10,0.05,False)

    print(collection.particles)

def constant_energy_check(N):

    collection = Ensemble(N)

    collection.set_velocities(np.random.random((2,N)))

    collection.simulate(100)
    collection.plot_energy("/home/sondre/Pictures/figs_simulation/energy.pdf")

    print(collection.E)

def many_particles_plot_test(N, T, verbose = True):
    
    collection = Ensemble(N)

    collection.set_velocities(np.random.random((2,N)))

    collection.simulate_savefigs(T,0.05,verbose)
    #print(collection.particles)

    
