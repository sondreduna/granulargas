import numpy as np
from tqdm import tqdm

class StopCriterion:
    """
    Abstract class for a stop criterion.

    Attributes
    ----------
    compare_val : undef 
    	value used for the comparison in stop(val)
    

    Methods
    -------
    stop(val)						
	function for stopping based on the type of 
        stop criterion
    """
    def __init__(self,compare_val):
        self.compare_val = compare_val
    
    def stop(self,val):
        raise NotImplementedError

class StopAtTime(StopCriterion):

    """
    Stop criterion, stopping when a time t has surpassed a
    given max-time

    """

    def stop(self,gas):
        return gas.t > self.compare_val

class StopAtEquilibrium(StopCriterion):

    """
    Stop criterion, stopping when the average number of collision
    per particle has surpassed a given number >> 1, default set to 100
    """

    def __init__(self,collision_max = 100):
        super().__init__(collision_max)

    def stop(self,gas):
        return np.average(gas.count) > self.compare_val

class StopAtDissipation(StopCriterion):

    """
    Stop criterion, stopping when a given fraction of the initial energy
    has dissipated. 
    """

    def __init__(self,gas,energy_fraction = 0.5):
        super().__init__(energy_fraction)
        self.init_energy = gas.total_energy()

    def stop(self,gas):
        self.current_energy = gas.total_energy()
        assert(self.init_energy != 0)
        return 1 - self.current_energy/self.init_energy > self.compare_val

