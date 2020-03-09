    
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


class SingleSense:
    def __init__(self, dir=[0, 0], angle=0):
        self.dir = dir
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





class CarState(WayPoint, Sensing):
    def __init__(self, n=10):
        WayPoint.__init__(self)
        Sensing.__init__(self, n)
        self.car = None

    def set_sense_locations(self, dx):
        # keep dx here so that the sensing distance can be set by the env
        for sense in self.senses:
            sense.sense_location = f.add_locations(self.x, sense.dir, dx)


class EnvState:
    def __init__(self):
        self.action = [0, 0]
        self.reward = 0
        self.distance_to_target = 0
        self.done = False

class SimulationState(CarState, EnvState):
    def __init__(self):
        super().__init__(n=10)
        self.step = 0

    def _add_car_state(self, car_state):
        self.x = car_state.x
        self.v = car_state.v
        self.theta = car_state.theta

    def _add_env_state(self, env_state):
        self.action = env_state.action
        self.reward = env_state.reward
        self.distance_to_target = env_state.distance_to_target

    def print_step(self, i):
        msg0 = str(i)
        msg1 = " State; x: " + str(np.around(self.x,2)) + " v: " + str(self.v) + "@ " + str(self.theta)
        msg2 = " Action: " + str(np.around(self.action,3))
        msg3 = " Reward: " + str(self.reward)

        print(msg0 + msg1 + msg2 + msg3)
        # self.print_sense()

