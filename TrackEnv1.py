import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import time
# import multiprocessing as mp 
import LocationState as ls
import EpisodeMem as em
from copy import deepcopy
import LibFunctions as f
import logging
from copy import deepcopy



class RaceEnv:
    def __init__(self, track, car, logger, dx=5, sense_dis=5):
        self.dx = dx # this is how close it must be to stop
        self.ds = sense_dis # this is how far the sensor can look ahead
        self.logger = logger
        self.track = track

        self.state_space = 2
        self.action_space = 10

        self.car_state = ls.CarState(self.action_space)
        self.env_state = ls.EnvState()
        self.car = car
        self.sim_mem = em.SimMem(self.logger)

    def step(self, action):
        new_x = self.car_state.chech_new_state(action)
        coll_flag = self._check_collision(new_x)

        if not coll_flag: # no collsions
            a, t_dot = self.car.get_delta(action)
            self.car_state.update_state(a, t_dot)

        self._update_senses()
        self.env_state.done = self._check_done()
        self._get_reward(coll_flag)

        return self.car_state, self.env_state.reward, self.env_state.done

    def control_step(self, action):
        new_x = self.car.chech_new_state(self.car_state, action)
        coll_flag = self._check_collision(new_x)

        if not coll_flag: # no collsions
            a, t_dot = self.car.get_delta(action)
            self.car.update_state(self.car_state, a, t_dot)

        self._update_senses()
        self.env_state.done = self._check_done()
        self.env_state.action = action

        self.sim_mem.add_step(self.car_state, self.env_state)
        # print(self.env_state.action)

        return self.car_state, self.env_state.done

    def reset(self):
        # resets to starting location
        self.car_state.x = deepcopy(self.track.start_location)
        self.car_state.v = 0
        self.car_state.theta = 0
        self._update_senses()
        self.reward = 0
        return self.car_state

    def _get_reward(self, coll_flag):
        dis = f.get_distance(self.car_state.x, self.track.end_location) 

        reward = 100 - dis  # reward increases as distance decreases
        if coll_flag:
            reward = -50

        self.env_state.distance_to_target = dis
        self.env_state.reward = reward

    def _check_done(self):
        dis = f.get_distance(self.track.end_location, self.car_state.x)

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
        # self.car_state.print_sense()

    def _get_next_state(self, action):
        # this takes a direction to move in and moves there
        new_x = f.add_locations(action, self.car_state.x)
        return new_x

    # def _get_next_controlled_state(self, a, dt):
    #     dd = dv = [0, 0]
    #     for i in range(2):
    #         dd[i] = a[i] * (dt ** 2) 
    #         dd[i] += self.car_state.v[i] * dt 
    #         dv[i] = a[i] * dt 
        
    #     new_x = f.add_locations(self.car_state.x, dd)
    #     new_v = f.add_locations(self.car_state.v, dv)
    #     return new_x, new_v


class TrackData:
    def __init__(self):
        self.boundary = None
        self.obstacles = []

        self.start_location = [0, 0]
        self.end_location = [0, 0]

        self.point_list = []

    def add_locations(self, x_start, x_end):
        self.start_location = x_start
        self.end_location = x_end

    def add_obstacle(self, obs):
        self.obstacles.append(obs)

    def add_boundaries(self, b):
        self.boundary = b

    def add_way_points(self, point_list):
        self.point_list = point_list


class CarModel:
    def __init__(self):
        self.m = 1
        self.L = 1
        self.J = 5
        self.b = [0.5, 0.2]

    def get_delta(self, f):
        a = f[0] / self.m
        theta_dot = f[1] * self.L / self.J
        # theta_dot = np.arctan((f[0]/ np.abs(f[1])))

        return a, theta_dot
    
    def update_state(self, car_state, a, theta_dot, dt=1):
        # self.x[0] += self.v * dt * np.sin(self.theta)
        # self.x[1] += self.v * dt * np.cos(self.theta)

        # self.v += np.abs(a * dt)
        # self.theta += theta_dot
        # x = [0, 0]
        car_state.x[0] += a * dt**2
        car_state.x[1] += theta_dot *dt **2

        # self.update_sense_offsets(self.theta)
        
        

    def chech_new_state(self, car_state, f=[0, 0], dt=1):
        x = [0.0, 0.0]
        x[0] = car_state.x[0] + car_state.v * dt * np.sin(car_state.theta)
        x[1] = car_state.x[1] + car_state.v * dt * np.cos(car_state.theta)

        return x


