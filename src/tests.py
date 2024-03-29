from events import *

def one_particle_test_1():

    collection = Gas(1)

    collection.particles[2:,0] = np.array([1,1])

    collection.plot_positions()

    collection.simulate_savefigs(10,0.05,verbose = False,videoname = "../fig/test_1")

def one_particle_test_2():

    collection = Gas(1)
    collection.xi = 0
    collection.particles[2:,0] = np.array([1,1])

    collection.plot_positions()

    collection.simulate_savefigs(10,0.05,verbose = False, videoname = "../fig/test_2")


def two_particle_test_1():

    collection = Gas(2)

    collection.particles[:2,0] = np.array([0.2,0.5])
    collection.particles[:2,1] = np.array([0.6,0.5])

    collection.particles[2:,0] = np.array([1.,0])
    collection.particles[2:,1] = np.array([-1,0])

    collection.simulate_savefigs(10,0.05,verbose = False , videoname = "../fig/test_3")

    print(collection.particles)
    
def two_particle_test_2():

    collection = Gas(2)
    collection.radii[0] = 0.02
    
    b = sum(collection.radii)/np.sqrt(2)
    
    collection.particles[:2,0] = np.array([0.2,0.5])
    collection.particles[:2,1] = np.array([0.6,0.5 + b])

    collection.particles[2:,0] = np.array([1.,0])
    collection.particles[2:,1] = np.array([-1,0])

    collection.simulate_savefigs(1,0.01,verbose = False, videoname = "../fig/test_4")
    
    print(collection.particles)

def two_particle_test_3():

    collection = Gas(2)
    collection.xi = 0

    collection.particles[:2,0] = np.array([0.2,0.5])
    collection.particles[:2,1] = np.array([0.6,0.5])

    collection.particles[2:,0] = np.array([1.,0])
    collection.particles[2:,1] = np.array([-1.,0])

    collection.simulate_savefigs(10,0.05,verbose = False, videoname = "../fig/test_5")

    print(collection.particles)

def constant_energy_check(N,T):

    collection = Gas(N, 0.001)

    collection.set_velocities(10 * np.random.random((2,N))) 

    collection.simulate(ret_vels = False)
    collection.plot_energy("/home/sondre/Pictures/figs_simulation/energy.pdf")

    print(collection.E)

def many_particles_plot_test(N, T):
    
    collection = Gas(N, 0.005)
    collection.xi = 1
    collection.set_velocities(-5 + 10 * np.random.random((2,N)))

    collection.simulate_savefigs(T,0.001,verbose = False,videoname="../fig/test_6" )
    #print(collection.particles)

    
def test_vel_dist(N,T):

    collection = Gas(N,0.001)

    v_0 = 0.1
    
    theta = np.random.random(N) * 2 * np.pi
    v = v_0 * np.array([np.cos(theta),np.sin(theta)])
    
    collection.set_velocities(v)

    collection.simulate()

    collection.plot_velocity_distribution(r"\textbf{Final distribution}", "../fig/dist.pdf",compare = True)
    

def snooker():

    board = Gas(11,0.01)

    board.particles[:,0] = np.array([0.2,0.5,1,0])

    board.M[0] = 100
    board.radii[0] = 0.02
    
    board.particles[:2,1] = np.array([0.5,0.5])
    board.particles[:2,2] = np.array([0.52,0.51])
    board.particles[:2,3] = np.array([0.52,0.49])
    board.particles[:2,4] = np.array([0.54,0.5])
    board.particles[:2,5] = np.array([0.54,0.52])
    board.particles[:2,6] = np.array([0.54,0.48])
    board.particles[:2,7] = np.array([0.56,0.51])
    board.particles[:2,8] = np.array([0.56,0.49])
    board.particles[:2,9] = np.array([0.56,0.53])
    board.particles[:2,10] = np.array([0.56,0.47])

    board.xi = 0.9
    board.simulate_savefigs(3,0.01,verbose = False, videoname = "../fig/snooker")

def brownian_motion():

    collection = Gas(501, 0.001)

    v_0 = 0.1
    
    theta = np.random.random(N) * 2 * np.pi
    v = v_0 * np.array([np.cos(theta),np.sin(theta)])

    collection.M[0] = 100
    collection.radii[0] = 0.02
    collection.set_velocities(v)

    collection.particles[2:,0] = np.array([0,0])

    board.simulate_savefigs(3,0.01,verbose = False, videoname = "../fig/browninan")
