import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import rc
import heapq
from prettytable import PrettyTable

R = 0.01
M = 0.01
eps = 1e-4

class Event:
    
    def __init__(self, time, i, j, event_type,insert_time):

        self.time        = time
        self.i           = i
        self.j           = j
        self.event_type  = event_type
        self.insert_time = insert_time

    def __lt__(self,other):
        return self.time < other.time

    def __repr__(self):
        x = PrettyTable()

        x.field_names = ["Event type", "i", "j","time of event","insert time"]

        if self.event_type == "pair":
            x.add_row([self.event_type,self.i,self.j,self.time,self.insert_time])

        else:
            x.add_row([self.event_type,self.i,None,self.time,self.insert_time])
        return x.get_string()
        

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
        self.N = N
        
        self.particles = np.zeros((4,N), dtype = np.float64)
        
        self.randomize_positions()
        
        self.last_collision = np.zeros(N)
        
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

    def randomize_positions_first(self):
        self.particles[:2,:] = self.radii + np.random.random((2,self.N)) * (1 - 2*self.radii)

        # loop to make sure none of the particles overlap
        
        for i in range(1,self.N):
            x_ij = self.particles[0,i] - self.particles[0,:i-1]
            y_ij = self.particles[1,i] - self.particles[1,:i-1]
            
            while np.any(x_ij**2 + y_ij**2 < (self.radii[:i-1] + self.radii[i])**2):
                self.particles[:2,i] = self.radii[i] + np.random.random(2) * (1 - 2*self.radii[i])

    def randomize_positions(self):

        # Distributing the particles uniformly, not overlapping,
        # assuming all radii are equal.
        r = self.radii[0]
        assert( np.all(self.radii == r ) )

        x = np.arange(3/2 * r,1 - 3/2 * r, 3 * r )
        y = np.arange(3/2 * r,1 - 3/2 * r, 3 * r )

        assert( self.N <= x.size **2 ) # check that we have enough points to draw from 
        
        for i in range(self.N):
            self.particles[:2,i] = np.array([np.random.choice(x, replace = False), np.random.choice(y,replace = False)])
        
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

        delta_t = np.inf
        other   = -1
        
        for j in range(self.N):
            if i != j:
                delta_x = self.particles[:2,j] - self.particles[:2,i]
                delta_v = self.particles[2:,j] - self.particles[2:,i]
                R_ij    = self.radii[i] + self.radii[j]
                d       = (delta_x @ delta_v)**2 - (delta_v @ delta_v) * ((delta_x @ delta_x) - R_ij**2)

                if delta_v @ delta_x < 0 and d > 0:
                    new_t =  - (delta_v @ delta_x + np.sqrt(d))/(delta_v @ delta_v)

                    if new_t < delta_t:
                        delta_t = new_t
                        other   = j
                    
        return delta_t, other
            
    def next_collision(self,i,t):
        delta_t = np.inf
        other   = -1

        wall = self.wall_collision_time(i)
        pair = self.particle_collision_time(i)
        
        if wall[0] < pair[0]:
            return Event(wall[0] + t,i,-1,wall[1],t)
        else:
            return Event(pair[0] + t ,i,pair[1],"pair",t)
    
    def new_velocities(self,event):
        
        if event.event_type == "ver_wall":
            self.particles[2,event.i] *= - self.xi
        elif event.event_type == "hor_wall":
            self.particles[3,event.i] *= - self.xi
        else:
            i = event.i
            j = event.j
            
            R_ij = np.sum(self.radii[[i,j]])
            delta_x_prime = self.particles[:2,j] - self.particles[:2,i]
            delta_v_prime = self.particles[2:,j] - self.particles[2:,i]
            self.particles[2:,i] = self.particles[2:,i] + ( (1+ self.xi) * \
                                      (self.M[j])/(self.M[i] + self.M[j]) * 1/R_ij**2 * \
                                      (delta_x_prime @ delta_v_prime) ) * delta_x_prime
            self.particles[2:,j] = self.particles[2:,j] - ( (1+ self.xi) * \
                                      (self.M[i])/(self.M[i] + self.M[j]) * 1/R_ij**2 * \
                                      (delta_x_prime @ delta_v_prime) ) * delta_x_prime
            
    
    def start_simulation(self):
        
        for i in range(self.N):
            new_event = self.next_collision(i,0)
            
            heapq.heappush(self.events, new_event)

    def is_valid(self,event,t):

        t_1 = self.last_collision[event.i]
        t_2 = self.last_collision[event.j]

        T = event.insert_time
        
        return ((event.event_type == "pair") and ((t_1 <= T) and (t_2 <= T))) or ((event.event_type != "pair") and (t_1 <= T))
            
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
                    heapq.heappush(self.events, self.next_collision(current.i,t))
                    heapq.heappush(self.events, self.next_collision(current.j,t))
                else:
                    self.last_collision[current.i] = t # collision between particle and wall
                    heapq.heappush(self.events, self.next_collision(current.i,t))

                """
                here: change new_collision above to
                a call here on self.next_collision(current) which calculates the next collision and 
                puts the result in the queue.
                """

                if save_snapshots:
                    self.plot_positions("./fig/img{0:0=3d}.png".format(iter))
                
                iter += 1
                
