import matplotlib.pyplot as plt
import numpy as np
import time
import StateStructs as ls
from copy import deepcopy
import LibFunctions as f
import logging
from copy import deepcopy
import pickle
import datetime
import os

class WayPoint:
    def __init__(self):
        self.x = [0.0, 0.0]
        self.v = 0.0
        self.theta = 0.0

    def print_point(self):
        print("X: " + str(self.x) + " -> v: " + str(self.v) + " -> theta: " +str(self.theta))


class SingleRange:
    def __init__(self, angle):
        self.val = 0 # distance to wall
        self.angle = angle
        self.dr = 0 # derivative of change of length

class Ranging:
    def __init__(self, n):
        self.n = n
        self.ranges = []

        dth = np.pi / (n-1)

        for i in range(n):
            angle = dth * i - np.pi/2
            ran = SingleRange(angle)
            self.ranges.append(ran)

    def _get_range_obs(self):
        obs = np.zeros(self.n)
        for i, ran in enumerate(self.ranges):
            obs[i] = ran.val

        return obs

    def print_ranges(self):
        obs = self._get_range_obs()
        print(obs)


class CarState(WayPoint, Ranging):
    def __init__(self, n):
        WayPoint.__init__(self)
        Ranging.__init__(self, n)
        self.cur_distance = 0.0
        self.prev_distance = 0.0
        self.glbl_wp = WayPoint() # this is what is being navigated towards

    def get_state_observation(self):
        # max normalisation constants
        max_v = 5
        max_theta = np.pi
        max_range = 100
        max_dis_glbl_wp = 100 # distance to next wp

        state = []
        state.append(self.v/max_v)
        state.append(self.theta/max_theta)

        x_glbl_bf = self.relocate_glbl_wp()
        state.append(x_glbl_bf[0]/max_dis_glbl_wp)
        state.append(x_glbl_bf[1]/max_dis_glbl_wp)
        for ran in self.ranges:
            r_val = np.around((ran.val/max_range), 4)
            state.append(r_val)

        state = np.array(state)
        state = state[None, :]
        return state

    def relocate_glbl_wp(self):
        # convers the glbl wp to the relative frame of the car
        #only worry about position of glbl wp, do orientation later #TODO 
        r = f.get_distance(self.glbl_wp.x, self.x)
        x = [0, 0]
        x[0] = - np.sin(self.theta) * r #toCheck sin convention
        x[1] = - np.cos(self.theta) * r
        # print("relocated x: " + str(x) + " -> th: " + str(self.theta) + " -> x: " + str(self.x))
        return x

    def get_distance_difference(self):
        return self.cur_distance - self.prev_distance

    def reset_state(self, start_location):
        self.x = deepcopy(start_location)
        self.v = 0
        self.theta = 0


class EnvState:
    def __init__(self):
        self.control_action = [0, 0]
        self.agent_action = 1 # straight
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


class SimMem:
    def __init__(self, logger=None):
        self.steps = []
        self.logger = logger

        self.step = 0

    def add_step(self, car_state, env_state):
        SimStep = ls.SimulationState()
        SimStep._add_car_state(deepcopy(car_state))
        SimStep._add_env_state(deepcopy(env_state))
        self.steps.append(SimStep)
        self.log_step(SimStep)
        self.step += 0

    def log_step(self, step):
        msg0 =  str(step.step) + ": ----------------------" + str(step.step)
        msg1 = "State: x->" + str(step.car_state.x) + "v-> [" + str(step.car_state.v) + "] theta->" + str(step.car_state.theta)
        msg2 = "Action: " + str(step.env_state.control_action)
        # msg3 = "Reward: " + str(step.reward)

        self.logger.debug(msg0)
        self.logger.debug(msg1)
        self.logger.debug(msg2)
        # self.logger.debug(msg3)

    def print_ep(self):
        for i, step in enumerate(self.steps):
            step.print_step(i)

    def save_ep(self, f_name):
        save_file_name =  f_name # + str(datetime.datetime.now())
        
        s_file = open(save_file_name, 'ab')

        pickle.dump(self.steps, s_file)

        s_file.close()

    def load_ep(self, f_name):
        save_file_name = f_name
        s_file = open(save_file_name, 'rb')

        self.steps = pickle.load(s_file)

        s_file.close()












