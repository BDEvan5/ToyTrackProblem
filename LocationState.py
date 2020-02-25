    
# these are minor helper data structures for the env
from copy import deepcopy
import LibFunctions as f
import numpy as np

class Location:
    def __init__(self, x=[0, 0]):
        self.x = x
        self.prev_x = [0, 0]
        self.dis = 100
        self.prev_dis = 100
        self.name = ""


class SingleSense:
    def __init__(self, dir=[0, 0], angle=0):
        self.dir = dir
        self.sense_location = [0, 0]
        self.val = 0 # always start open
        self.angle = angle

    def print_sense(self):
        print(str(self.dir) + " --> Val: " + str(self.val) + " --> Loc: " + str(self.sense_location))


class Sensing:
    def __init__(self, n):
        self.n = n  # number of senses
        self.senses = []

        d_angle = np.pi / n

        dir = [0, 0]
        for i in range(n):
            angle =  i * d_angle - np.pi /2 # -90 to +90
            dir[0] = np.sin(angle)
            dir[1] = - np.cos(angle)  # the - deal with positive being down
            dir = np.around(dir, decimals=4)

            sense = SingleSense(dir, angle)
            self.senses.append(sense)
                
    def print_sense(self):
        for e in self.senses:
            e.print_sense()


class CarState(Location, Sensing):
    def __init__(self, n=10, v=[0, 0], x=[0, 0]):
        Location.__init__(self, x) # it would appear that you must innitialise an inherited object 
        Sensing.__init__(self, n)
        self.v = v

    def set_sense_locations(self, dx):
        # keep dx here so that the sensing distance can be set by the env
        for sense in self.senses:
            sense.sense_location = f.add_locations(self.x, sense.dir, dx)




# class EnvState:
#     # this class holds the current track information
#     def __init__(self):
#         self.step = 0
#         self.



