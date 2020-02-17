import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import time
# import multiprocessing as mp 
import LocationState as ls


class RaceTrack:
    def __init__(self, interface, dt=0.5):
        self.track = interface
        self.dt = dt

        self.state = ls.State()

        start_location = [80, 80]
        self.start_location = ls.State(x=start_location)
        self.end_location = ls.State(x=[20, 20])           
        self.track.location.x = self.track.scale_input(start_location)

    def add_locations(self, start_location, end_location):
        self.start_location = start_location
        self.end_location = end_location

    def step(self, action):
        # action space is (ax, ay)
        # this just simulates what will happen if the car accelerates at this rate for the time step
        # d represents the change
        self._get_next_state(action)

        done = self._check_done()
        reward = self._check_distance_to_target()

        # this is needed so that the result can be visualised
        # this will show every 10th step
        time.sleep(self.dt/10)        

        return self.state, reward, done

    def reset(self):
        self.state.set_state()
        self.reward = 0


        # reset other variables here

    def render(self):
        # this function sends the state to the interface
        x = self.state.x.copy()  
        
        self.track.q.put(x)

    def _check_done(self):
        if self.state == self.end_location:
            return True
        return False

    def _check_distance_to_target(self):
        dx = (np.power(self.state.x[0] - self.end_location.x[0], 2))
        dy = (np.power(self.state.x[1] - self.end_location.x[1], 2))
        dis = np.sqrt((dx)+(dy))

        return dis

    def _get_next_state(self, action):
        dv = [0, 0]
        dd = [0, 0]
        for i in range(2):
            # this performs manual integration, I am not fully sure this is correct but I think it is
            dv[i] = action[i] * self.dt
            dd[i] = action[i] * self.dt * self.dt

        self.state.update_state(dv, dd)
        self._check_next_state()

    def _check_next_state(self):
        # this will be exapnded to a bigger function that will include friction etc
        for i in range(2):
            # this in effect provides the bounce off the wall idea
            # it flips the position and velocity
            if self.state.x[i] < 0:
                self.state.x[i] = -self.state.x[i]
                self.state.v[i] = - self.state.v[i]
            # check lower boundary
            if self.state.x[i] > self.track.size[i]:
                self.state.x[i] = 2*self.track.size[i] -self.state.x[i]
                self.state.v[i] = - self.state.v[i]






