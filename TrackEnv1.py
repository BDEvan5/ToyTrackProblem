import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import time
# import multiprocessing as mp 
import LocationState as ls
from copy import deepcopy


class RaceTrack:
    def __init__(self, interface, dt=0.5, dx=1, sense_dis=5):
        self.track = interface
        self.dt = dt
        self.dx = dx
        self.ds = sense_dis

        self.state = ls.State()
        self.prev_s = ls.State()
        
        self.start_location = ls.State()
        self.end_location = ls.State()   

        self.obstacles = []        

    def add_locations(self, x_start, x_end):
        self.start_location.set_location(x_start)
        self.end_location.set_location(x_end)
        self.track.location.x = self.track.scale_input(x_start)
        self.track.set_end_location(x_end)

    def add_obstacle(self, obs):
        self.track.add_obstacle(obs)
        self.obstacles.append(obs)

    def step(self, action):
        
        self._get_next_state(action)
        self._update_senses()

        done = self._check_done()
        reward = self._check_distance_to_target()   

        return self.state, reward, done

    def reset(self):
        # resets to starting location
        self.state.set_state(self.start_location.x)
        self._update_senses()
        self.reward = 0
        return self.state

    def render(self):
        # this function sends the state to the interface
        x = deepcopy(self.state)
        self.track.q.put(x)

    def _check_done(self):
        dis = [0.0, 0.0]
        for i in range(2):
            dis[i] = self.end_location.x[i] - self.state.x[i] 
        distance_to_target = np.linalg.norm(dis)
        # print(distance_to_target)

        if distance_to_target < self.dx:
            print("Final distance is: " % distance_to_target)
            return True
        return False

    def _check_distance_to_target(self):
        dx = (np.power(self.state.x[0] - self.end_location.x[0], 2))
        dy = (np.power(self.state.x[1] - self.end_location.x[1], 2))
        dis = np.sqrt((dx)+(dy))

        return dis

    def _get_next_state(self, action):
        self.prev_s.x = deepcopy(self.state.x)
        # self.prev_s = self.state.copy()
        dv = [0, 0]
        dd = [0, 0]
        for i in range(2):
            # this performs manual integration, I am not fully sure this is correct but I think it is
            dv[i] = action[i] * self.dt
            dd[i] = action[i] * self.dt * self.dt

        # come back to a more accurate model one day, but at the moment there are problems

        self.state.update_state(dv, dd)
        self._check_next_state()
        self._check_collision()
        self._update_senses()

    def _check_next_state(self):
        # this will be exapnded to a bigger function that will include friction etc
        for i in range(2):
            # this in effect provides the bounce off the wall idea
            # it flips the position and velocity
            if self.state.x[i] < 0:
                self.state.x[i] = -self.state.x[i]
                self.state.v[i] = - self.state.v[i]
            # check outer boundary
            if self.state.x[i] > self.track.size[i]:
                self.state.x[i] = 2*self.track.size[i] -self.state.x[i]
                self.state.v[i] = - self.state.v[i]

        self._check_collision()

    def _check_collision(self):
        x = self.state.x
        for o in self.obstacles:
            if o[0] < x[0] < o[2]:
                if o[1] < x[1] < o[3]:
                    print("Collision")
                    # this just resets to previos postion.
                    self.state.x = self.prev_s.x
                    self.state.v = [0, 0]

    def _update_senses(self):
        self.state.set_sense_locations(self.ds)
        # self.state.print_sense()
        for sense in self.state.senses:
            for o in self.obstacles:
                if o[0] < sense.sense_location[0] < o[2]:
                    if o[1] < sense.sense_location[1] < o[3]:
                        sense.val = 1
                        # print("Sense going to collide")
                else:
                    sense.val = 0

            





