import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import rc
import heapq
from prettytable import PrettyTable
import numba as nb
from numba import types
import plotting 


from utils import * 

# default mass value
M = 10

class Event:
    """
    Class for storing information about a collision event.

    Attributes
    ----------

    time : float
        Time of event
    i    : int
        Index of first particle involved in event
    j    : int 
        Index of first particle involved in event
    event_type : string
        Type of event: either pair : collision between a pair of particles, wall_h or wall_v : collision between horizontal or vertical wall
    count_i : int
        number of collisions of particle i when event created
    count_j : int
        number of collisions of particle j when event created


    Methods
    -------
    
    __lt__(other)
       self < other
    is_valid(count)
       checks if event is valid.
    
    """
    def __init__(self, time, i, j, event_type,count_i, count_j):

        self.time         = time
        self.i            = i
        self.j            = j
        self.event_type   = event_type
        self.count_i      = count_i
        self.count_j      = count_j

    def __lt__(self,other):
        """
        Less than operator for two events. Compares the time. 

        Parameters
        ----------
        
        other : Event
            event to compare
        """
        return self.time < other.time

    def __repr__(self):
        """
        Table string representation of event, useful for debugging. 
        """
        x = PrettyTable()

        x.field_names = ["Event type", "i", "j","time of event","count for particle i", "count for particle j"]

        if self.event_type == "pair":
            x.add_row([self.event_type,self.i,self.j,self.time,self.count_i,self.count_j])

        else:
            x.add_row([self.event_type,self.i,None,self.time,self.count_i,None])
        return x.get_string()

    def is_valid(self,count):
        """
        Checking if an event is valid, i.e. the counts of the involved 
        particles has not changed since last time.

        Parameters
        ----------
        count : np.array(int)
             count of particles from ensemble

        Returns
        -------
        _ : boolean
             True if valid, False otherwise
        """
        if self.event_type == "pair":
            return (count[self.i] == self.count_i) and (count[self.j] == self.count_j) and self.time > 0
        else:
            return (count[self.i] == self.count_i) and self.time > 0
		
class Gas:
    """
    A gas of particles.

    Attributes
    ----------
    
    radii : np.array(float)
        List of radii of particles
    N     : int 
        Number of particles in gas
    particles : np.array(float)
        Positions and velocities of particles, i.e. particles[:,i] = [x_i,y_i,vx_i,vy_i]
    count : np.array(int)
        Count of number of collisions for each particle
    M : np.array(float)
        List of masses of particles
    xi : float
        restitution coefficient
    events : heapq
        Priority queue of Events  

    Parameters
    ----------
    
    N : int
        Number of particles 
    R : float
        Initial set radius for all particles 
    
    """
    
    def __init__(self,N,R = 0.01):
       
        self.radii = np.full(N,R)
        self.N = N
        
        self.particles = np.zeros((4,N))
        
        self.randomize_positions()
        
        self.count = np.zeros(N)
        
        self.M = np.full(N,M)
        
        self.xi = 1
        
        self.events = []
        
        heapq.heapify(self.events)
            
    # setters and getters
    # -------------------
    
    def set_velocities(self,v):
        self.particles[2:,:] = v
        
    def set_radii(self,r):
        self.radii = r
        
    def set_masses(self,m):
        self.M = m

    def get_velocities(self):
        return self.particles[2:,:]

    def get_v_square(self):
        """
        Function for getting the square of the velocities

        """
        v = self.get_velocities()
        return np.einsum('ij,ij->j',v,v)

    def get_positions(self):
        return self.particles[:2,:]

    # -------------------
    
    def kT(self):
        """
        Function for calculating k_B T according to the 
        equipartition theorem : k_B * T = 2 * 1/2 * < 1/2 * m * v^2 > 

        """
        
        m = self.M[0]
        assert(np.all(self.M == m))

        v2 = self.get_v_square()
        
        return np.average(v2) * m / 2

    def total_energy(self):

        """
        Function for getting the total energy of the system

        """
        
        return 1/2 * np.dot(self.get_v_square(),self.M)

    
    
    # plotting functions
    # ------------------
    
    def plot_positions(self,savefig = ""):
        plotting._plot_positions(self,savefig)

    def plot_velocity_distribution(self, title, savefig= "", compare = False):
        plotting._plot_velocity_distribution(self,title,savefig,compare)

    def plot_energy(self, savefig = ""):
        plotting._plot_energy(self,savefig)

    # ------------------
    
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
        
    
    def wall_collision_time(self,i):

        """
        Function for calculating the time for the next collision 
        with a wall for particle i 

        Parameters
        ----------
        i : int
            particle index

        Returns 
        -------
        delta_t_h : float
            time to collision with horizontal wall

        delta_t_v : float
            time to collision with vertical wall 
        
        """
        
        v = self.particles[2:,i]
        
        delta_t_h = np.inf
        delta_t_v = np.inf 
        
        if v[0] > 0 :
            delta_t_v = (1 - self.radii[i] - self.particles[0,i])/v[0]
        elif v[0] < 0:
            delta_t_v = (self.radii[i] - self.particles[0,i])/v[0]
        
        if v[1] > 0 :
            delta_t_h = (1 - self.radii[i] - self.particles[1,i])/v[1]
        elif v[1] < 0:
            delta_t_h = (self.radii[i] - self.particles[1,i])/v[1]

        
        return delta_t_h, delta_t_v

    
    def particle_collision_time(self,i,t):

        """
        Function for calculating when a particle will collide with all other 
        particles in the gas. Updates the queue directly.

        Parameters
        ----------
        i : int 
            index of particle
        t : float 
            current time # TODO remove this and replace all occurences with self.t
       
        """

        T = np.full(self.N - 1, np.inf)
        mask = np.arange(self.N-1)
        mask[i:] += 1
        
        r_ij = self.radii[mask] + self.radii[i]
        
        delta_x = self.particles[:2,mask] - np.reshape(self.particles[:2,i],(2,1))
        delta_v = self.particles[2:,mask] - np.reshape(self.particles[2:,i],(2,1))

        vv = np.einsum('ij,ij->j',delta_v,delta_v)
        vx = np.einsum('ij,ij->j',delta_v,delta_x)
        xx = np.einsum('ij,ij->j',delta_x,delta_x)

        d = vx ** 2 - vv * ( xx - r_ij**2 )

        c_mask = (vx < 0 ) * (d > 0)
        
        T[c_mask] = - ( vx[c_mask] + np.sqrt(d[c_mask]) )/(vv[c_mask])

        T = T[c_mask]
        J = mask[c_mask]

        for j in range(np.size(T)):
            heapq.heappush(self.events,Event(T[j] + t ,i,J[j],"pair",self.count[i], self.count[J[j]]))
        
    def particle_collisions(self,i,t):
        """
        Function for calculating when a particle will collide with all other 
        particles in the gas. Updates the queue directly.

        Bad implementation of the above function 

        Parameters
        ----------
        i : int 
            index of particle
        t : float 
            current time # TODO remove this and replace all occurences with self.t
       
        """
        for j in range(self.N):
            if i != j:
                delta_x = self.particles[:2,j] - self.particles[:2,i]
                delta_v = self.particles[2:,j] - self.particles[2:,i]
                R_ij    = self.radii[i] + self.radii[j]
                d       = (delta_x @ delta_v)**2 - (delta_v @ delta_v) * ((delta_x @ delta_x) - R_ij**2)

                if delta_v @ delta_x < 0 and d > 0:
                    new_t =  - (delta_v @ delta_x + np.sqrt(d))/(delta_v @ delta_v)
                    heapq.heappush(self.events, Event(new_t + t ,i,j,"pair",self.count[i], self.count[j]))

                    
    def next_collision(self,i,t):
        """
        Function for calculating the next collision of particle i at time t
        
        Parameters
        ----------
        i : int
            index of particle
        t : float
            current time
    
        """

        wall_h, wall_v = self.wall_collision_time(i)
        heapq.heappush(self.events, Event(wall_h + t ,i,-1,"hor_wall", self.count[i], -1))
        heapq.heappush(self.events, Event(wall_v + t ,i,-1,"ver_wall", self.count[i], -1))
        #self.particle_collisions(i,t)
        self.particle_collision_time(i,t)
            
    
    def new_velocities(self,event):

        """
        Function for updating the velocities of the particles involved in an event.

        Parameters
        ----------
        
        event : Event
            previous event
        
        """
        
        if event.event_type == "ver_wall":
            self.particles[2,event.i] *= - self.xi
        elif event.event_type == "hor_wall":
            self.particles[3,event.i] *= - self.xi
        else:
            i = event.i
            j = event.j
            
            R_ij = self.radii[i] + self.radii[j]
            delta_x_prime = self.particles[:2,j] - self.particles[:2,i]
            delta_v_prime = self.particles[2:,j] - self.particles[2:,i]
            
            self.particles[2:,i] += ( (1+ self.xi) * \
                                      (self.M[j])/(self.M[i] + self.M[j]) * 1/R_ij**2 * \
                                      (delta_x_prime @ delta_v_prime) ) * delta_x_prime
            self.particles[2:,j] -= ( (1+ self.xi) * \
                                      (self.M[i])/(self.M[i] + self.M[j]) * 1/R_ij**2 * \
                                      (delta_x_prime @ delta_v_prime) ) * delta_x_prime
            
    
    def start_simulation(self):
        
        for i in range(self.N):
            self.next_collision(i,0)
            
    def simulate(self, dt = 0.1, stopper = "time", stop_val = 10, ret_vels = True):
    
        
        if stopper == "time":
            self.stop_criterion = StopAtTime(stop_val)
            
        elif stopper == "equilibrium":
            self.stop_criterion = StopAtEquilibrium(stop_val)

        elif stopper == "dissipation":
            self.stop_criterion = StopAtDissipation(self,stop_val)

        self.v_0 = self.particles[2:,:] # saving initial velocities
        
        # We don't know a priori how many iterations we are going to use
        # so we will let the energy E be dynamically sized.
                                
        self.E = []
        self.speeds = [np.copy(self.v_0)]
        self.times  = [0]

        progress_bar = tqdm( total = stop_val )
            
        # Let the current time be a member of the object so that we can
        # feed it into the stop criterion
                                
        self.t = 0.0 
        t_save = 0.0
        
        self.count = np.zeros(self.N) # reset counts
        self.start_simulation()       # does initial calculation of next collision

        total_count = 0
        
        while not self.stop_criterion.stop(self):
            
            # popping the earliest event

            current = heapq.heappop(self.events)

            # checking whether the ith or jth particle of this event has
            # collided since the event was put in the queue
            
            if current.is_valid(self.count):
                
                time = current.time

                # stops the loop if there is no next collision 
                if time == np.inf:
                    break

                # updates progress-bar
                # TODO: get rid of the if-check using a new type
               
                if stopper == "equilibrium":            
                    progress_bar.update(np.average(self.count) - total_count/self.N )
                    total_count = np.sum(self.count) # NB this is probably very inefficient    

                if stopper == "dissipation":
                    new_val = 1 - self.stop_criterion.current_energy/self.stop_criterion.init_energy
                    progress_bar.update(new_val - progress_bar.n)

                # moves forward dt to have outputs at equidistant points in time
   
                while t_save + dt < time:
                    if stopper == "time":
                        progress_bar.update(dt)
                    
                    time_step = t_save + dt - self.t
                     
                    self.particles[:2,:] += time_step * self.particles[2:,:] # move all particles forward

                    self.E.append(self.total_energy())       # saving energies 
                    self.speeds.append(np.copy(self.particles[2:,:])) # saving speeds
                    self.times.append(self.t)                # saving current time
                    
                    t_save += dt
                    self.t = t_save    

                # updating the time of the last collision of the particles
                # involved in the collision

                self.particles[:2,:] += (time - self.t) * self.particles[2:,:] # move all particles
                self.new_velocities(current)         # setting new velocities
                
                self.t = time                        # updating the time

                # finds the next collision of the involved particles
                
                if current.event_type == "pair":     # collision between a pair of particles
                    self.count[[current.i,current.j]] += 1
                    self.next_collision(current.i,self.t)
                    self.next_collision(current.j,self.t)
                    
                else:
                    self.count[current.i] += 1
                    self.next_collision(current.i,self.t)


        self.E = np.array(self.E)
        self.speeds = np.array(self.speeds)
        self.times  = np.array(self.times)
        
        progress_bar.close()

        if ret_vels:
            return self.particles[2:,:]

    def simulate_saveE(self, dt = 0.1, stopper = "time", stop_val = 10, ret_vels = True):
        
        if stopper == "time":
            self.stop_criterion = StopAtTime(stop_val)
            
        elif stopper == "equilibrium":
            self.stop_criterion = StopAtEquilibrium(stop_val)


        progress_bar = tqdm( total = stop_val )
            
        # Let the current time be a member of the object so that we can
        # feed it into the stop criterion
                                
        self.t = 0.0 
        t_save = 0.0
        
        self.count = np.zeros(self.N) # reset counts
        self.start_simulation()       # does initial calculation of next collision

        self.v_0 = self.particles[2:,:] # saving initial velocities
        
        # We don't know a priori how many iterations we are going to use
        # so we will let the energy E be dynamically sized.
                                
        self.E = []
        self.E_avg = [] 
        total_count = 0
        
        while not self.stop_criterion.stop(self):
            
            # popping the earliest event

            current = heapq.heappop(self.events)

            # checking whether the ith or jth particle of this event has
            # collided since the event was put in the queue
            
            if current.is_valid(self.count):
                
                time = current.time

                # stops the loop if there is no next collision 
                if time == np.inf:
                    break

                # updates progress-bar
                # TODO: get rid of the if-check using a new type
               
                if stopper == "equilibrium":            
                    progress_bar.update(np.average(self.count) - total_count/self.N )
                    total_count = np.sum(self.count) # NB this is probably very inefficient
                    self.E.append(self.total_energy())
        
                # moves forward dt to have outputs at equidistant points in time
                while t_save + dt < time:

                    progress_bar.update(dt)
                    
                    time_step = t_save + dt - self.t
                     
                    self.particles[:2,:] += time_step * self.particles[2:,:] # move all particles forward

                    total_E = self.total_energy()
                    self.E.append(total_E)
                    
                    vv = self.get_v_square()
                    
                    E_1 = 0.5 * self.M[0] * vv[self.M == self.M[0]]
                    E_2 = 0.5 * 4 * self.M[0] * vv[self.M == 4* self.M[0]]
                    
                    self.E_avg.append([np.average(E_1),np.average(E_2), 1/self.N * total_E])
    
                    t_save += dt
                    self.t = t_save    

                # updating the time of the last collision of the particles
                # involved in the collision

                self.particles[:2,:] += (time - self.t) * self.particles[2:,:] # move all particles
                self.new_velocities(current)         # setting new velocities
                
                self.t = time                        # updating the time

                # finds the next collision of the involved particles
                
                if current.event_type == "pair":     # collision between a pair of particles
                    self.count[[current.i,current.j]] += 1
                    self.next_collision(current.i,self.t)
                    self.next_collision(current.j,self.t)
                    
                else:
                    self.count[current.i] += 1
                    self.next_collision(current.i,self.t)


        self.E = np.array(self.E)
        self.E_avg = np.array(self.E_avg)
        progress_bar.close()

        if ret_vels:
            return self.particles[2:,:]    

    def simulate_savefigs(self,T,dt, verbose = False):
        """
        Simulates the system up to a time T, and saves a snapshot of 
        the particles' positions at every timestep dt.

        Parameters
        ----------
        T : float
            end time
        dt : float
            time step
        verbose : boolean
            True  = prints the current event for each step. Only for debugging purposes.
            False = does not print anything
     
        """

        self.stop_criterion = StopAtTime(T)
        
        self.t = 0.0
        t_save = 0.0
        
        self.start_simulation()
        self.count = np.zeros(self.N) # reset time
        self.v_0 = self.particles[2:,0]
        
        progress_bar = tqdm(total = T )

        it = 0

        # When restricting to a sum up to T, we know exactly how long E should be
        self.E = np.zeros(int(T/dt) + 1, dtype = np.float64)
        
        while not self.stop_criterion.stop(self):
            
            # popping the earliest event
            
            current = heapq.heappop(self.events)

            # checking whether the ith or jth particle of this event has
            # collided since the event was put in the queue
            
            if current.is_valid(self.count):
                
                time = current.time
                
                if verbose:
                    print(current)
                    print(it)


                # moves forward dt to have outputs at equidistant points in time
                while t_save + dt < current.time:
                    
                    time_step = t_save + dt - self.t
                    progress_bar.update(dt)
                    
                    self.particles[:2,:] += time_step * self.particles[2:,:] # move all particles forward
                    self.plot_positions("/home/sondre/Pictures/figs_simulation/img{0:0=3d}.png".format(it))
                    self.E[it] = self.total_energy()
                    it += 1
                    t_save += dt
                    self.t = t_save
                
                if time == np.inf:
                    break
                
                # updating the time of the last collision of the particles
                # involved in the collision
                
                self.particles[:2,:] += (time - self.t) * self.particles[2:,:] # move all particles
                self.new_velocities(current)         # setting new velocities

                # self.particles = np.around(self.particles,13) # rounding positions and velocities to 13 digits
                
                self.t = time                        # updating the time
                
                if current.event_type == "pair":     # collision between a pair of particles
                    self.count[[current.i,current.j]] += 1
                    self.next_collision(current.i,self.t)
                    self.next_collision(current.j,self.t)
                else:
                    self.count[current.i] += 1
                    self.next_collision(current.i,self.t)

        self.E = self.E[:it]
        progress_bar.close()
