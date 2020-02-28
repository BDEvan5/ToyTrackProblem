    
# these are minor helper data structures for the env
from copy import deepcopy
import LibFunctions as f
import numpy as np


class WayPoint:
    def __init__(self):
        self.x = [0.0, 0.0]
        self.v = 0.0
        self.theta = 0.0

    def set_point(self, x, v, theta):
        self.x = x
        self.v = v
        self.theta = theta

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

    def update_sense_offsets(self, offset):
        dir = [0, 0]
        for i, sense in enumerate(self.senses):
            dir[0] = np.sin(sense.angle + offset)
            dir[1] = - np.cos(sense.angle + offset)
            dir = np.around(dir, decimals=4)

            sense.dir = dir


class CarModel:
    def __init__(self):
        self.m = 1
        self.L = 1
        self.J = 1

    def get_delta(self, f):
        a = f[0] / self.m
        theta_dot = f[1] * self.L / self.J

        return a, theta_dot


class CarState(WayPoint, Sensing, CarModel):
    def __init__(self, n=10):
        WayPoint.__init__(self) 
        Sensing.__init__(self, n)
        self.car = None

    def set_sense_locations(self, dx):
        # keep dx here so that the sensing distance can be set by the env
        for sense in self.senses:
            sense.sense_location = f.add_locations(self.x, sense.dir, dx)

    def update_state(self, a, theta_dot, dt=1):
        self.x[0] += self.v * dt * np.sin(self.theta)
        self.x[1] += self.v * dt * np.cos(self.theta)

        self.v += a * dt
        self.theta += theta_dot

        self.update_sense_offsets(self.theta)

    def chech_new_state(self, f=[0, 0], dt=1):
        x = [0.0, 0.0]
        x[0] = self.x[0] + self.v * dt * np.sin(self.theta)
        x[1] = self.x[1] + self.v * dt * np.cos(self.theta)

        return x



