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

    def stop(self,ensemble):
        return ensemble.t > self.compare_val

class StopAtEquilibrium(StopCriterion):

    """
    Stop criterion, stopping when the average number of collision
    per particle has surpassed a given number >> 1, default set to 100
    """

    def __init__(self,collision_max = 100):
        super.__init__(collision_max)

    def stop(self,ensemble):
        return np.average(ensemble.count) > self.compare_val

class ProgressBar(tqdm):
    """
    A progress-bar qustomized to the 
    chosen stop criterion of the simulation 
    
    Not done with this yet.

    """

    def __init__(self,stopper,stop_val):
        super.__init__(total = stop_val)

    def update(self,val):
        raise NotImplementedError 
