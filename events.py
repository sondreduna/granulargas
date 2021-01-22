import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import rc
import heapq

class Event:
    
    def __init__(self, i, j, event_type):
        
        self.i = i
        self.j = j
        self.event_type = event_type 

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
        
        self.particles = np.zeros((4,N))
        self.particles[:2,:] = np.random.random((2,N))
        
        self.collided  = np.zeros(N, dtype = np.bool)
        self.N = N
        
        self.M = np.full(N,M)
        self.radii = np.full(N,R)
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
        
        fig = plt.figure(figsize = (8,8))
        
        plt.scatter(self.particles[0,:],self.particles[1,:], color = "blue", s = 2)
        
        # boundary of box
        plt.hlines([0,1],[0,0],[1,1], ls = "--", color = "black")
        plt.vlines([0,1],[0,0],[1,1], ls = "--", color = "black")
        
        plt.grid(ls = "--")
        
        plt.tight_layout()
        
        if savefig != "":
            plt.savefig(savefig)
    
    
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
    
    def new_velocities(self,event):
        
        if event.event_type == "ver_wall":
            self.particles[2,event.i] *= - self.xi
        elif event.event_type == "hor_wall":
            self.particles[3,event.i] *= - self.xi 
    
    def start_simulation(self):
        
        for i in range(self.N):
            collision = self.wall_collision_time(i)
            
            heapq.heappush(self.events, (collision[0], Event(i,0,collision[1])) )
            
    def simulate(self, T):
        
        t = 0
        
        self.start_simulation()
        
        while t < T:
        
            current = heapq.heappop(self.events)
        
            time = current[0]
            event = current[1]
            
            self.particles[:2,:] += (time - t)* self.particles[2:,:] # move all particles
            self.new_velocities(event)
            
            t = time
            
            new_collision = self.wall_collision_time(event.i)
            heapq.heappush(self.events,
                           (new_collision[0] + t, Event(event.i, 0, new_collision[1])))
            
            #self.plot_positions()
            #plt.show()
        
