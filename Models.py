from StateStructs import CarState
import numpy as np
import time
from copy import deepcopy
import LibFunctions as f
import logging
from copy import deepcopy
import math
import copy 



class TrackConfig:
    # this holds methods for track configuration

    def straight_track(self):
        start_location = [50.0, 95.0]
        end_location = [50.0, 10.0]
        o1 = (0, 0, 30, 100)
        o2 = (70, 0, 100, 100)
        o3 = (35, 60, 51, 70)
        o4 = (49, 30, 65, 40)
        b = (1, 1, 99, 99)

        self.add_locations(start_location, end_location)
        self.boundary = b
        self.add_wall(o1)
        self.add_wall(o2)

    def single_corner(self):
        start_location = [80.0, 95.0]
        end_location = [5.0, 20.0]
        o1 = (0, 0, 100, 5)
        o2 = (0, 35, 65, 100)
        o3 = (95, 0, 100, 100)
        b = (1, 1, 99, 99)

        self.add_locations(start_location, end_location)
        self.boundary = b
        self.add_obstacle(o1)
        self.add_obstacle(o2)
        self.add_obstacle(o3)

    def simple_maze(self):
        start_location = [95.0, 85.0]
        end_location = [10.0, 10.0]
        o1 = (20, 0, 40, 70)
        o2 = (60, 30, 80, 100)
        b = (1, 1, 99, 99)

        self.add_locations(start_location, end_location)
        self.boundary = b
        self.add_wall(o1)
        self.add_wall(o2)


class TrackData(TrackConfig):
    def __init__(self):
        TrackConfig.__init__(self)
        self.boundary = None
        self.obstacles = []
        self.hidden_obstacles = []

        self.start_location = [0, 0]
        self.end_location = [0, 0]

    def add_locations(self, x_start, x_end):
        self.start_location = x_start
        self.end_location = x_end

    def add_wall(self, obs):
        self.obstacles.append(obs)

    def add_boundary(self, b):
        self.boundary = b

    def add_obstacle(self):
        o = Obstacle([15, 10])
        o.bounding_box = [40, 20, 60, 80]
        self.hidden_obstacles.append(o)

    def _check_collision_hidden(self, x):
        b = self.boundary
        ret = 0
        for i, o in enumerate(self.obstacles):
            if o[0] < x[0] < o[2]:
                if o[1] < x[1] < o[3]:
                    msg = "Obstacle collision %d --> x: %d;%d"%(i, x[0],x[1])
                    ret = 1
        for i, obstacle in enumerate(self.hidden_obstacles):
            o = obstacle.get_obstacle_shape()
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
            if o[0] <= x[0] <= o[2]:
                if o[1] <= x[1] <= o[3]:
                    msg = "Obstacle collision %d --> x: %d;%d"%(i, x[0],x[1])
                    ret = 1
        if x[0] <= b[0] or x[0] >= b[2]:
            msg = "X wall collision --> x: %d, b:%d;%d"%(x[0], b[0], b[2])
            ret = 1
        if x[1] <= b[1] or x[1] >= b[3]:
            msg = "Y wall collision --> y: %d, b:%d;%d"%(x[1], b[1], b[3])
            ret = 1
        # if ret == 1:
        #     print(msg)
        return ret

    def get_heat_map(self):
        heat_map = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                if self._check_collision([i, j]):
                    heat_map[i, j] = 1

        # returns a 100 by 100 grid of obstacle values
        return heat_map

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

    def set_up_random_obstacles(self):
        for obs in self.hidden_obstacles:
            obs.set_random_location()

    def check_line_collision(self, x1, x2):
        n_checks = 20
        m = f.get_gradient(x1, x2)
        dx = (x2[0] - x1[0]) / n_checks
        for i in range(n_checks):
            x_add = [i * dx, i * m * dx]
            x_search = f.add_locations(x_add, x1)
            if self._check_collision(x_search):
                return True
        return False

    def check_hidden_line_collision(self, x1, x2):
        n_checks = 20
        m = f.get_gradient(x1, x2)
        dx = (x2[0] - x1[0]) / n_checks
        for i in range(n_checks):
            x_add = [i * dx, i * m * dx]
            x_search = f.add_locations(x_add, x1)
            if self._check_collision_hidden(x_search):
                return True
        return False


class Obstacle:
    def __init__(self, size=[0, 0]):
        self.x_size = size[0]
        self.y_size = size[1]

        self.location = [0, 0]
        self.shape = np.zeros(4)
        self.bounding_box = None

    def get_obstacle_shape(self): # used by track to get shape
        x1 = self.location[0] - self.x_size /2
        x2 = self.location[0] + self.x_size /2
        y1 = self.location[1] - self.y_size /2
        y2 = self.location[1] + self.y_size /2

        self.shape = [x1, y1, x2, y2]
        return self.shape

    def set_random_location(self):
        x1 = self.bounding_box[0] + self.x_size / 2
        y1 = self.bounding_box[1] + self.y_size / 2
        x2 = self.bounding_box[2] - self.x_size / 2
        y2 = self.bounding_box[3] - self.y_size / 2

        x_loc = np.random.randint(x1, x2)
        y_loc = np.random.randint(y1, y2)

        self.location = [x_loc, y_loc]



        

    
    



if __name__ == "__main__":
    myTrack = TrackData()
    myTrack.straight_track()
    myTrack.add_obstacle()

    for o in myTrack.obstacles:
        print(o)







