    
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

    def print_point(self):
        print("X: " + str(self.x) + " -> v: " + str(self.v) + " -> theta: " +str(self.theta))

class Path:
    def __init__(self):
        self.route = []
    
    def add_way_point(self, x, v=0, theta=0):
        wp = WayPoint()
        wp.set_point(x, v, theta)
        self.route.append(wp)

    def print_route(self):
        for wp in self.route:
            print("X: (%d;%d), v: %d, th: %d" %(wp.x[0], wp.x[1], wp.v, wp.theta))

class SingleSense:
    def __init__(self, direc=[0, 0], angle=0):
        self.dir = direc
        self.sense_location = [0, 0]
        self.val = 0 # always start open
        self.angle = angle

    def print_sense(self):
        print(str(self.dir) + " --> Val: " + str(self.val) + " --> Loc: " + str(self.sense_location))


class Sensing:
    def __init__(self, n=10):
        self.n = n  # number of senses
        self.senses = []

        d_angle = np.pi / n

        direc = [0, 0]
        for i in range(n):
            angle =  i * d_angle - np.pi /2 # -90 to +90
            direc[0] = np.sin(angle)
            direc[1] = - np.cos(angle)  # the - deal with positive being down
            direc = np.around(direc, decimals=4)

            sense = SingleSense(direc, angle)
            self.senses.append(sense)
                
    def print_sense(self):
        for e in self.senses:
            e.print_sense()

    def update_sense_offsets(self, offset):
        direc = [0, 0]
        # print(offset)
        for i, sense in enumerate(self.senses):
            direc[0] = np.sin(sense.angle + offset)
            direc[1] = - np.cos(sense.angle + offset)
            direc = np.around(direc, decimals=4)

            sense.dir = direc
        






class CarState(WayPoint, Sensing):
    def __init__(self, n=10):
        WayPoint.__init__(self)
        Sensing.__init__(self, n)
        self.car = None

    def set_sense_locations(self, dx):
        # keep dx here so that the sensing distance can be set by the env
        # print("Old sense")
        # self.print_sense()
        self.update_sense_offsets(self.theta)
        for sense in self.senses:
            sense.sense_location = f.add_locations(self.x, sense.dir, dx)
        # print("new Sense")
        # self.print_sense()

class EnvState:
    def __init__(self):
        self.action = [0, 0]
        self.reward = 0
        self.distance_to_target = 0
        self.done = False

class SimulationState():
    def __init__(self):
        self.car_state = CarState()
        self.step = 0

    def _add_car_state(self, car_state):
        self.car_state = car_state

    def _add_env_state(self, env_state):
        self.env_state = env_state

    def print_step(self, i):
        msg0 = str(i)
        msg1 = " State; x: " + str(np.around(self.x,2)) + " v: " + str(self.v) + "@ " + str(self.theta)
        msg2 = " Action: " + str(np.around(self.action,3))
        msg3 = " Reward: " + str(self.reward)

        print(msg0 + msg1 + msg2 + msg3)
        # self.print_sense()

