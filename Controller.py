import LocationState as ls
import numpy as np
import LibFunctions as f


class Contoller:
    # the aim of this class is to be able to give it a way point 
    # and it determines a number of steps to take until 
    # the way point is reached
    def __init__(self):
        self.car = ls.CarModel()

        self.current_pos = ls.WayPoint()
        self.target = ls.WayPoint()

    def get_action(self):
        self.d_pos = f.sub_locations(self.current_pos.x, self.target.x)
        

