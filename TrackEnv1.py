import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import time
import LocationState as ls
import EpisodeMem as em
from copy import deepcopy
import LibFunctions as f
import logging
from copy import deepcopy



class RaceEnv:
    def __init__(self, track, car, logger, dx=5, sense_dis=8):
        self.dx = dx # this is how close it must be to stop
        self.ds = sense_dis # this is how far the sensor can look ahead
        self.logger = logger
        self.track = track
        self.dt = 0.5 # update frequency

        self.state_space = 2
        self.action_space = 5

        self.car_state = ls.CarState(self.action_space)
        self.env_state = ls.EnvState()
        self.car = car
        self.sim_mem = em.SimMem(self.logger)

    def step(self, action):
        new_x = self.car.chech_new_state(self.car_state, action, self.dt)
        coll_flag = self.track._check_collision_hidden(new_x)

        if not coll_flag: # no collsions
            self.car.update_controlled_state(self.car_state, action, self.dt)

        self._update_senses()
        self.env_state.done = self._check_done(coll_flag)
        self._get_reward(coll_flag)
        self._update_ranges()
        self.env_state.action = action

        self.sim_mem.add_step(self.car_state, self.env_state)

        return self.car_state, self.env_state.reward, self.env_state.done

    def control_step(self, action):
        new_x = self.car.chech_new_state(self.car_state, action)
        coll_flag = self.track._check_collision_hidden(new_x)

        if not coll_flag: # no collsions
            self.car.update_controlled_state(self.car_state, action, self.dt)

        self._update_senses()
        self.env_state.done = self._check_done()
        self.env_state.action = action

        self.sim_mem.add_step(self.car_state, self.env_state)

        return self.car_state, self.env_state.done

    def reset(self):
        # resets to starting location
        self.car_state.x = deepcopy(self.track.start_location)
        self.car_state.v = 0
        self.car_state.theta = 0
        self._update_senses()
        self.reward = 0
        self.sim_mem.steps.clear()
        print("Mem cleared" + str(self.sim_mem.steps))
        return self.car_state

    def _get_reward(self, coll_flag):
        dis = f.get_distance(self.car_state.x, self.track.end_location) 

        reward = 0 # reward increases as distance decreases
        if coll_flag:
            reward = -50
        # if self.env_state.done:
        #     reward = 50

        self.env_state.distance_to_target = dis
        self.env_state.reward = reward

    def _check_done(self, coll_flag):
        dis = f.get_distance(self.track.end_location, self.car_state.x)

        if coll_flag:
            print("Ended in collision")
            return True

        if dis < self.dx:
            print("Final distance is: %d" % dis)
            return True
        return False

    def _update_senses(self):
        self.car_state.set_sense_locations(self.ds)
        b = self.track.boundary
        # self.car_state.print_sense()
        for sense in self.car_state.senses:
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
            for o in self.track.hidden_obstacles:
                if o[0] < sense.sense_location[0] < o[2]:
                    if o[1] < sense.sense_location[1] < o[3]:
                        sense.val = 1
        # self.car_state.print_sense()

    def _get_next_state(self, action):
        # this takes a direction to move in and moves there
        new_x = f.add_locations(action, self.car_state.x)
        return new_x

    def get_ep_mem(self):
        # self.sim_mem.print_ep()
        return self.sim_mem

    def _update_ranges(self):
        dx = 5 # search size
        curr_x = self.car_state.x
        th_car = self.car_state.theta
        for ran in self.car_state.ranges:
            th = th_car + ran.angle
            crash_val = 0
            i = 0
            while crash_val == 0:
                r = dx * i
                addx = [dx * i * np.sin(th), -dx * i * np.cos(th)] # check
                x_search = f.add_locations(curr_x, addx)
                crash_val = self.track._check_collision_hidden(x_search)
                i += 1
            ran.val = (i - 2) * dx # sets the last distance before collision 
        # self.car_state.print_ranges()


class TrackData(ls.Path):
    def __init__(self):
        ls.Path.__init__(self)
        self.boundary = None
        self.obstacles = []
        self.hidden_obstacles = []

        self.start_location = [0, 0]
        self.end_location = [0, 0]

    def add_locations(self, x_start, x_end):
        self.start_location = x_start
        self.end_location = x_end

    def add_obstacle(self, obs):
        self.obstacles.append(obs)

    def add_boundaries(self, b):
        self.boundary = b

    def _check_collision_hidden(self, x):
        b = self.boundary
        ret = 0
        for i, o in enumerate(self.obstacles):
            if o[0] < x[0] < o[2]:
                if o[1] < x[1] < o[3]:
                    msg = "Obstacle collision %d --> x: %d;%d"%(i, x[0],x[1])
                    ret = 1
        for i, o in enumerate(self.hidden_obstacles):
            if o[0] < x[0] < o[2]:
                if o[1] < x[1] < o[3]:
                    msg = "Hidden Obstacle collision %d --> x: %d;%d"%(i, x[0],x[1])
                    ret = 1
        if x[0] < b[0] or x[0] > b[2]:
            msg = "X wall collision --> x: %d, b:%d;%d"%(x[0], b[0], b[2])
            ret = 1
        if x[1] < b[1] or x[1] > b[3]:
            msg = "Y wall collision --> y: %d, b:%d;%d"%(x[1], b[1], b[3])
            ret = 1
        # if ret == 1:
            # print(msg)
            # self.logger.info(msg)
        return ret

    def _check_collision(self, x):
        b = self.boundary
        ret = 0
        for i, o in enumerate(self.obstacles):
            if o[0] < x[0] < o[2]:
                if o[1] < x[1] < o[3]:
                    msg = "Obstacle collision %d --> x: %d;%d"%(i, x[0],x[1])
                    ret = 1
        if x[0] < b[0] or x[0] > b[2]:
            msg = "X wall collision --> x: %d, b:%d;%d"%(x[0], b[0], b[2])
            ret = 1
        if x[1] < b[1] or x[1] > b[3]:
            msg = "Y wall collision --> y: %d, b:%d;%d"%(x[1], b[1], b[3])
            ret = 1
        return ret

    def _check_path_collision(self, x0, x1):
        # write this one day to check  points along line
        b = self.boundary
        ret = 0
        for i, o in enumerate(self.obstacles):
            if o[0] < x1[0] < o[2]:
                if o[1] < x1[1] < o[3]:
                    msg = "Obstacle collision %d --> x: %d;%d"%(i, x1[0],x1[1])
                    ret = 1
        
        if x[0] < b[0] or x1[0] > b[2]:
            msg = "X wall collision --> x: %d, b:%d;%d"%(x1[0], b[0], b[2])
            ret = 1
        if x1[1] < b[1] or x1[1] > b[3]:
            msg = "Y wall collision --> y: %d, b:%d;%d"%(x1[1], b[1], b[3])
            ret = 1
        # if ret == 1:
            # print(msg)
            # self.logger.info(msg)
        return ret

    def add_hidden_obstacle(self, obs):
        self.hidden_obstacles.append(obs)

    def get_ranges(self, x, th):
        # x is location, th is orientation 
        # given a range it determine the distance to each object 
        b = self.boundary
        ret = 0
        for i, o in enumerate(self.obstacles):
            if o[0] < x[0] < o[2]:
                if o[1] < x[1] < o[3]:
                    msg = "Obstacle collision %d --> x: %d;%d"%(i, x[0],x[1])
                    ret = 1
        if x[0] < b[0] or x[0] > b[2]:
            msg = "X wall collision --> x: %d, b:%d;%d"%(x[0], b[0], b[2])
            ret = 1
        if x[1] < b[1] or x[1] > b[3]:
            msg = "Y wall collision --> y: %d, b:%d;%d"%(x[1], b[1], b[3])
            ret = 1
        return ret


class CarModel:
    def __init__(self):
        self.m = 1
        self.L = 1
        self.J = 5
        self.b = [0.5, 0.2]

        self.max_v = 0
        self.friction = 0.1
        
    def set_up_car(self, max_v):
        self.max_v = max_v
    
    def update_controlled_state(self, state, action, dt):
        a = action[0]
        th = action[1]
        state.v += a * dt - self.friction * state.v

        state.theta = th # assume no th integration to start
        r = state.v * dt
        state.x[0] += r * np.sin(state.theta)
        state.x[1] += - r * np.cos(state.theta)

        state.update_sense_offsets(state.theta)
           
    def chech_new_state(self, state, action, dt):
        x = [0.0, 0.0]
        a = action[0]
        th = action[1]
        v = a * dt - self.friction * state.v + state.v

        r = v * dt
        x[0] = r * np.sin(th) + state.x[0]
        x[1] = - r * np.cos(th) + state.x[1]
        # print(x)
        return x


