import matplotlib.pyplot as plt
import numpy as np
import time
from copy import deepcopy
import LibFunctions as f
import logging
from copy import deepcopy


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



class TrackData(Path):
    def __init__(self):
        Path.__init__(self)
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

        # update range offsets somewhere
           
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












