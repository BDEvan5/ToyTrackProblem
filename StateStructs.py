from copy import deepcopy
import LibFunctions as f
import numpy as np

class WayPoint:
    def __init__(self):
        self.x = [0.0, 0.0]
        self.v = 0.0
        self.theta = 0.0

    def print_point(self):
        print("X: " + str(self.x) + " -> v: " + str(self.v) + " -> theta: " +str(self.theta))


class SingleSense:
    def __init__(self, direc=[0, 0], angle=0):
        self.dir = direc
        self.sense_location = [0, 0]
        self.val = 0 # always start open
        self.angle = angle

    def print_sense(self):
        print(str(self.dir) + " --> Val: " + str(self.val) + " --> Loc: " + str(self.sense_location))

class Sensing:
    def __init__(self, n):
        self.n = n  # number of senses
        self.senses = []
        # self.senses.append(SingleSense((0, 0), 0))

        d_angle = np.pi / (n - 1)

        direc = [0, 0]
        angle = 0
        for i in range(n):
            sense = SingleSense()
            self.senses.append(sense)

        for i in range(n):
            angle =  i * d_angle - np.pi /2 # -90 to +90
            direc[0] = np.sin(angle)
            direc[1] = - np.cos(angle)  # the - deal with positive being down
            direc = np.around(direc, decimals=4)

            self.senses[i].dir = direc
            self.senses[i].angle = angle
            # print(self.senses[i].dir)
        
        # self.senses[0].print_sense()
        # for i in range(n):
            # print(self.senses[i].dir)
            # print(self.senses[i].angle)


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
        
    def get_sense_observation(self):
        obs = np.zeros(self.n)
        for i, sense in enumerate(self.senses):
            obs[i] = sense.val
        # print(obs)
        # self.print_sense()
        return obs # should return an array of 1 or 0 for senses

    def print_directions(self):
        arr = np.zeros((self.n, 2))
        for i, sen in enumerate(self.senses):
            arr[i] = sen.dir
        print(arr)


class SingleRange:
    def __init__(self, angle):
        self.val = 0 # distance to wall
        self.angle = angle
        self.dr = 0 # derivative of change of length

class Ranging:
    def __init__(self, n):
        self.n = n
        self.ranges = []

        dth = np.pi / n

        for i in range(n):
            ran = SingleRange(dth * i - np.pi/2)
            self.ranges.append(ran)

    def _get_range_obs(self):
        obs = np.zeros(self.n)
        for i, ran in enumerate(self.ranges):
            obs[i] = ran.val

        return obs

    def get_range_state_num(self):
        obs = self._get_range_obs()

        num = 0
        for i in range(len(obs)-1): #last sense doesn't work
            num += obs[i] * (2**i)
        return int(num)

    def print_ranges(self):
        obs = self._get_range_obs()
        print(obs)


class CarState(WayPoint, Sensing, Ranging):
    def __init__(self, n):
        WayPoint.__init__(self)
        Ranging.__init__(self, n)
        Sensing.__init__(self, n)

    def get_state_observation(self):
        bin_scale = 10
        state = []
        state.append(self.v)
        state.append(self.theta)
        # consider adding action here
        for ran in self.ranges:
            r_val = np.around((ran.val/bin_scale), 0)
            state.append(r_val)
            dr_val = np.around(ran.dr, 0)
            state.append(dr_val)

    def get_sense_state_num(self):
        # state = []
        # state.append(self.v)
        # state.append(self.theta)

        obs = self.get_sense_observation()
        obs_n = 0
        for i in range(len(obs)): 
            obs_n += obs[i] * (2**i)
        return int(obs_n)

    def set_sense_locations(self, dx):
        self.update_sense_offsets(self.theta)
        for sense in self.senses:
            sense.sense_location = f.add_locations(self.x, sense.dir, dx)


class EnvState:
    def __init__(self):
        self.action = [0, 0]
        self.reward = 0
        self.distance_to_target = 0
        self.done = False

class SimulationState():
    def __init__(self):
        self.car_state = CarState(5)
        self.env_state = EnvState()
        self.step = 0

    def _add_car_state(self, car_state):
        self.car_state = car_state

    def _add_env_state(self, env_state):
        self.env_state = env_state

    # def print_step(self, i):
    #     msg0 = str(i)
    #     msg1 = " State; x: " + str(np.around(self.x,2)) + " v: " + str(self.v) + "@ " + str(self.theta)
    #     msg2 = " Action: " + str(np.around(self.action,3))
    #     msg3 = " Reward: " + str(self.reward)

    #     print(msg0 + msg1 + msg2 + msg3)
    #     # self.print_sense()












