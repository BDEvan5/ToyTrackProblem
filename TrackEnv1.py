import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import time
import multiprocessing


class RaceTrack:
    def __init__(self, interface, dt=0.5):
        self.start_location = State(x=[80, 80])
        self.end_location = State(x=[20, 20])

        self.state = State()

        self.track = interface
        self.dt = dt

    def add_locations(self, start_location, end_location):
        self.start_location = start_location
        self.end_location = end_location

    def step(self, action):
        # action space is (ax, ay)
        # this just simulates what will happen if the car accelerates at this rate for the time step
        # d represents the change
        dv = [0, 0]
        dd = [0, 0]
        for i in range(2):
            # this performs manual integration, I am not fully sure this is correct but I think it is
            dv[i] = action[i] * self.dt
            dd[i] = action[i] * self.dt * self.dt

        self.state.update_state(dv, dd)
        # print(dd)

        done = self.check_done()
        reward = self.check_distance_to_target()

        time.sleep(0.1)        

        return self.state, reward, False

    def reset(self):
        self.state.set_state()
        self.reward = 0


        # reset other variables here

    def render(self):
        #this function provides visual representation
        x = self.state.x.copy()
        print(self.state.x)

        for i in range(2):
            if x[i] <0:
                x[i] = -x[i]
            x[i] = x[i] *200
            
        self.track.q.put(x)

    def check_done(self):
        if self.state == self.end_location:
            return True
        return False

    def check_distance_to_target(self):
        dx = (np.power(self.state.x[0] - self.end_location.x[0], 2))
        dy = (np.power(self.state.x[1] - self.end_location.x[1], 2))
        dis = np.sqrt((dx)+(dy))

        return dis





    

class Location:
    def __init__(self, x=[0, 0]):
        self.x = x

    def set_location(self, x):
        self.x = x

    def get_location(self):
        return self.x


class State(Location):
    def __init__(self, v=[0, 0], x=[0, 0]):
        Location.__init__(self, x) # it would appear that you must innitialise an inherited object 
        self.v = v

    def set_v(self, v):
        self.v = v
    
    def update_state(self, dv, dd):
        for i in range(2):
            self.x[i] += dd[i]
            self.v[i] += dv[i]


    def set_state(self, x=[0, 0], v=[0, 0] ):
        self.x = x
        self.v = v




