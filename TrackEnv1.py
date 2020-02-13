import matplotlib.pyplot as plt
import numpy as np


class RaceTrack:
    def __init__(self, dt=0.5):
        self.start_location = State(x=[80, 80])
        self.end_location = State(x=[20, 20])

        self.state = State()

        plt.ion()
        self.fig = plt.figure()
        self.axes = self.fig.gca()

        self.dt = dt

    def show_track(self):
        x = [0, 0, 100, 100]
        y = [0, 100, 0, 100]

        self.axes.plot(x, y, 'ro', color='black')
        self.axes.plot(self.end_location.x, self.end_location.y, 'x', color='green', markersize=10)
        self.axes.plot(self.start_location.x, self.end_location.y, 'x', color='green', markersize=10)

        plt.draw()
        plt.pause(0.01)

    def draw_car(self):
        self.axes.plot(self.state.location.x, self.state.location.y, '*', color='red', markersize=10)


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

        done = self.check_done
        reward = self.check_distance_to_target()

        next_state = self.state.copy()

        return next_state.x, reward, False

    def reset(self):
        self.state.set_state()
        self.reward = 0

        # reset other variables here

    def check_done(self):
        if self.state == self.end_location:
            return True
        return False

    def check_distance_to_target(self):
        dx = (np.power(self.state.x[0] - self.end_location.x[0], 2))
        dy = (np.power(self.state.x[1] - self.end_location.x[1], 2))
        dis = np.sqrt((dx)+(dy))

        return dis


class Car:
    # the car is part of the environment
    def __init__(self):
        self.mass = 1000




    

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
        self.x += dd
        self.v += dv

    def set_state(self, x=[0, 0], v=[0, 0] ):
        self.x = x
        self.v = v

