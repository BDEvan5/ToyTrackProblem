import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import time
# import multiprocessing as mp 
import LocationState as ls
from copy import deepcopy
import LibFunctions as f
import logging


class RaceEnv:
    def __init__(self, track, logger, dx=5, sense_dis=5):
        self.dx = dx # this is how close it must be to stop
        self.ds = sense_dis # this is how far the sensor can look ahead
        self.logger = logger
        self.track = track

        self.state_space = 2
        self.action_space = 10

        self.state = ls.CarState(self.action_space)
        self.car = ls.CarModel()
    
    def step(self, action):
        # self.state.increment_location()

        # new_x = self._get_next_state(action)
        # new_x, new_v = self._get_next_controlled_state(action, dt=1)
        new_x = self.state.chech_new_state(action)
        coll_flag = self._check_collision(new_x)

        if not coll_flag: # no collsions
            a, t_dot = self.car.get_delta(action)
            self.state.update_state(a, t_dot)

        self._update_senses()
        done = self._check_done()
        reward = self._get_reward(coll_flag)

        return self.state, reward, done

    def reset(self):
        # resets to starting location
        self.state.x = self.track.start_location
        self.state.v = 0
        self._update_senses()
        self.reward = 0
        return self.state

    def _get_reward(self, coll_flag):
        dis = f.get_distance(self.state.x, self.track.end_location) 

        reward = 100 - dis  # reward increases as distance decreases
        if coll_flag:
            reward = -50

        self.state.dis = dis
        return  reward

    def _check_done(self):
        dis = f.get_distance(self.track.end_location, self.state.x)

        if dis < self.dx:
            print("Final distance is: %d" % dis)
            return True
        return False

    def _check_collision(self, x):
        b = self.track.boundary
        ret = 0
        for o in self.track.obstacles:
            if o[0] < x[0] < o[2]:
                if o[1] < x[1] < o[3]:
                    msg = "Boundary collision --> x: %d;%d"%(x[0],x[1])
                    ret = 1
        if x[0] < b[0] or x[0] > b[2]:
            msg = "X wall collision --> x: %d, b:%d;%d"%(x[0], b[0], b[2])
            ret = 1
        if x[1] < b[1] or x[1] > b[3]:
            msg = "Y wall collision --> y: %d, b:%d;%d"%(x[1], b[1], b[3])
            ret = 1
        if ret == 1:
            # print(msg)
            self.logger.info(msg)
        return ret

    def _update_senses(self):
        self.state.set_sense_locations(self.ds)
        b = self.track.boundary
        # self.state.print_sense()
        for sense in self.state.senses:
            if b[0] < sense.sense_location[0] < b[2]:
                if b[1] < sense.sense_location[1] < b[3]:
                    sense.val = 0
                    # It is within the boundary
                else:
                    # hits te wall boundary
                    sense.val = 1

            for o in self.track.obstacles:
                if o[0] < sense.sense_location[0] < o[2]:
                    if o[1] < sense.sense_location[1] < o[3]:
                        sense.val = 1
                        # if it hits an obstacle   

    def _get_next_state(self, action):
        # this takes a direction to move in and moves there
        new_x = f.add_locations(action, self.state.x)
        return new_x

    def _get_next_controlled_state(self, a, dt):
        dd = dv = [0, 0]
        for i in range(2):
            dd[i] = a[i] * (dt ** 2) 
            dd[i] += self.state.v[i] * dt 
            dv[i] = a[i] * dt 
        
        new_x = f.add_locations(self.state.x, dd)
        new_v = f.add_locations(self.state.v, dv)
        return new_x, new_v


class TrackData:
    def __init__(self):
        self.boundary = None
        self.obstacles = []

        self.start_location = [0, 0]
        self.end_location = [0, 0]

    def add_locations(self, x_start, x_end):
        self.start_location = x_start
        self.end_location = x_end

    def add_obstacle(self, obs):
        self.obstacles.append(obs)

    def add_boundaries(self, b):
        self.boundary = b


