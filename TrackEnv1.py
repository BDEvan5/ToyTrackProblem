import matplotlib.pyplot as plt
import numpy as np

class RaceTrack:
    def __init__(self):
        self.car = None
        self.end_location = None


        self.fig = plt.figure()
        self.axes = self.fig.gca()

    def show_track(self):
        x = [0, 0, 100, 100]
        y = [0, 100, 0, 100]

        self.axes.plot(x, y, 'ro', color='black')
        self.axes.plot(self.car.location.x, self.car.location.y, '*', color='red', markersize=10)
        self.axes.plot(self.end_location.x, self.end_location.y, 'x', color='green', markersize=10)
        

        plt.show()

    def add_car(self, car):
        self.car = car

    def add_end_point(self, location):
        self.end_location = location




class SimpleCarBase:
    def __init__(self):
        self.mass = 1000

        self.location = Location()
        self.location.set_location(40, 60)

    


class Location:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

        self.vx = 0
        self.vy = 0

    def set_location(self, x, y):
        self.x = x
        self.y = y

    def get_location(self):
        return [self.x, self.y]

    def set_v(self, vx, vy):
        self.vx = vx
        self.vy = vy


    