import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import time
# import multiprocessing as mp 
import LocationState as ls
from copy import deepcopy
import LibFunctions as f
import logging


class RaceTrack:
    def __init__(self, logger, dx=10, sense_dis=5):
        self.dx = dx # this is how close it must be to stop
        self.ds = sense_dis # this is how far the sensor can look ahead
        self.logger = logger

        self.state_space = 2
        self.action_space = 10

        self.state = ls.CarState(self.action_space)
        
        self.start_location = ls.Location()
        self.end_location = ls.Location()   

        self.obstacles = []        
        self.boundary = None

    def add_locations(self, x_start, x_end):
        self.start_location.x = x_start
        self.end_location.x = x_end

    def add_obstacle(self, obs):
        self.obstacles.append(obs)

    def add_boundaries(self, b):
        self.boundary = b

    def step(self, action):
        # self.prev_s = deepcopy(self.state)
        self.state.prev_x = deepcopy(self.state.x)
        self.state.prev_dis = deepcopy(self.state.dis)

        new_x = self._get_next_state(action)
        coll_flag = self._check_collision(new_x)

        if not coll_flag: # no collsions
            self.state.x = new_x

        self._update_senses()
        done = self._check_done()
        reward = self._get_reward()

        return self.state, reward, done

    def reset(self):
        # resets to starting location
        self.state.x = self.start_location.x
        self._update_senses()
        self.reward = 0
        return self.state

    def _get_reward(self):
        dis = f.get_distance(self.state.x, self.end_location.x) 

        if dis < self.state.prev_dis:
            # it moved closer
            reward  = 10
        else:
            reward = -10
            # it moved further away

        self.state.dis = dis
        return  reward

    def _check_done(self):
        dis = f.get_distance(self.end_location.x, self.state.x)

        if dis < self.dx:
            print("Final distance is: %d" % dis)
            return True
        return False

    def _check_collision(self, x):
        b = self.boundary
        ret = 0
        for o in self.obstacles:
            if o[0] < x[0] < o[2]:
                if o[1] < x[1] < o[3]:
                    msg = "Boundary collision --> x: %d;%d"%(x[0],x[1])
                    ret = 1
        if x[0] < b[0] or x[0] > b[2]:
            msg = "X wall collision --> x: %d, b:%d;%d"%(x[0], b[0], b[2])
            ret = 1
        if x[1] < b[1] or x[1] > b[3]:
            msg = "Y wall collision --> x: %d, b:%d;%d"%(x[1], b[1], b[3])
            ret = 1
        if ret == 1:
            # print(msg)
            self.logger.info(msg)
        return ret

    def _update_senses(self):
        self.state.set_sense_locations(self.ds)
        b = self.boundary
        # self.state.print_sense()
        for sense in self.state.senses:
            if b[0] < sense.sense_location[0] < b[2]:
                if b[1] < sense.sense_location[1] < b[3]:
                    sense.val = 0
                    # It is within the boundary
                else:
                    # hits te wall boundary
                    sense.val = 1

            for o in self.obstacles:
                if o[0] < sense.sense_location[0] < o[2]:
                    if o[1] < sense.sense_location[1] < o[3]:
                        sense.val = 1
                        # if it hits an obstacle   

    def _get_next_state(self, action):
        # dd = [0, 0]
        # for i in range(2):
        #     dd[i] = action[i] * self.dt * self.dt

        # this is where dynamics can be added

        new_x = f.add_locations(action, self.state.x)

        return new_x


