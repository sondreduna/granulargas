import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import rc
import heapq

R = 0.01
M = 0.01

class Event:
    
    def __init__(self, time, i, j, event_type,insert_time):

        self.time        = time
        self.i           = i
        self.j           = j
        self.event_type  = event_type
        self.insert_time = insert_time

    def __lt__(self,other):
        return self.time < other.time

class Ensemble:
    """
    Wrapper for ensemble of particles
    
    
    """
    
    def __init__(self,N):
        """
        Setting the positions of N particles inside a square
        box of lengths 1 randomly according to U(0,1), and
        velocities to 0
        
        Parameters
        ----------
        
        N : int
            number of particles in ensemble
        
        """
        self.radii = np.full(N,R)
        self.particles = np.zeros((4,N), dtype = np.float64)
        
        self.particles[:2,:] = self.radii + np.random.random((2,N)) * (1 - self.radii)
        
        self.last_collision = np.zeros(N)
        self.N = N
        
        self.M = np.full(N,M)
        
        self.xi = 1
        
        self.events = []
        heapq.heapify(self.events)
        
    def set_velocities(self,v):
        self.particles[2:,:] = v
        
    def set_radii(self,r):
        self.radii = r
        
    def set_masses(self,m):
        self.M = m
        
    def plot_positions(self,savefig = ""):
        
        fig, ax = plt.subplots(figsize = (8,8))

        for i in range(self.N):
            
            ax.add_artist(plt.Circle((self.particles[0,i],
                                    self.particles[1,i]),
                                    self.radii[i],
                                    linewidth=0,
                                    color = "blue"))
        
        # boundary of box
        plt.hlines([0,1],[0,0],[1,1], ls = "--", color = "black")
        plt.vlines([0,1],[0,0],[1,1], ls = "--", color = "black")
        
        plt.grid(ls = "--")
        
        plt.tight_layout()
        
        if savefig != "":
            plt.savefig(savefig)

            plt.close()
    
    
    def wall_collision_time(self,i):
        
        v = self.particles[2:,i]
        
        delta_t_h = np.inf
        delta_t_v = np.inf 
        
        collision_type = "hor_wall"
        
        if v[0] > 0 :
            delta_t_v = (1 - self.radii[i] - self.particles[0,i])/v[0]
        elif v[0] < 0:
            delta_t_v = (self.radii[i] - self.particles[0,i])/v[0]
        
        if v[1] > 0 :
            delta_t_h = (1 - self.radii[i] - self.particles[1,i])/v[1]
        elif v[1] < 0:
            delta_t_h = (self.radii[i] - self.particles[1,i])/v[1]
            
        delta_t = delta_t_h
        if delta_t_v < delta_t_h:
            collision_type = "ver_wall"
            delta_t = delta_t_v
        
        return delta_t , collision_type

    def particle_collision_time(self,i):
        raise NotImplementedError
    def next_collision(self,i):
        raise NotImplementedError
    
    def new_velocities(self,event):
        
        if event.event_type == "ver_wall":
            self.particles[2,event.i] *= - self.xi
        elif event.event_type == "hor_wall":
            self.particles[3,event.i] *= - self.xi 
    
    def start_simulation(self):
        
        for i in range(self.N):
            collision = self.wall_collision_time(i)
            
            heapq.heappush(self.events, Event(collision[0],i,-1,collision[1],0))

    def is_valid(self,event,t):

        t_1 = self.last_collision[event.i]
        t_2 = self.last_collision[event.j]
        
        return (event.event_type == "pair") and ((t_1 <= t) and (t_2 <= t)) or (event.event_type != "pair") and (t_1 <= t)
            
    def simulate(self, T, save_snapshots = False):
        
        t = 0.0
        self.last_collision = np.zeros(self.N) # reset the time
        self.start_simulation()


        iter = 0
        while t < T:

            # popping the earliest event
            current = heapq.heappop(self.events)

            # checking whether the ith or jth particle of this event has
            # collided since the event was put in the queue

            if self.is_valid(current,t):
                
                time = current.time

                # updating the time of the last collision of the particles
                # involved in the collision

                self.particles[:2,:] += (time - t) * self.particles[2:,:] # move all particles
                self.new_velocities(current)         # setting new velocities
                t = time                             # updating the time
                
                if current.event_type == "pair":     # collision between a pair of particles
                    self.last_collision[[current.i,current.j]] = t
                else:
                    self.last_collision[current.i] = t # collision between particle and wall
                    
                new_collision = self.wall_collision_time(current.i)                                       # figure out the next collision for particle i
                heapq.heappush(self.events,Event(new_collision[0] + t,current.i, -1, new_collision[1],t)) # add new collision to queue


                """
                here: change new_collision above to
                a call here on self.next_collision(current) which calculates the next collision and 
                puts the result in the queue.
                """

                if save_snapshots:
                    self.plot_positions("./fig/img{0:0=3d}.png".format(iter))
                
                iter += 1
                
