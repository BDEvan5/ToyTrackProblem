import numpy as np 
from Interface import Interface



class WayPoint:
    def __init__(self):
        self.x = [0.0, 0.0]
        self.v = 0.0
        self.theta = 0.0

    def set_point(self, x, v, theta):
        self.x = x
        self.v = v
        self.theta = theta

    def print_point(self, text=None):
        # print("X: " + str(self.x) + " -> v: " + str(self.v) + " -> theta: " +str(self.theta))
        if text is None:
            print(f"X: [{self.x[0]:.2f} ; {self.x[1]:.2f}] -> V: {self.v:.2f} -> theta: {self.theta:.2f}")
        else:
            print(f"[{text}] X: [{self.x[0]:.2f} ; {self.x[1]:.2f}] -> V: {self.v:.2f} -> theta: {self.theta:.2f}")

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

    def __len__(self):
        return len(self.route)

    def __iter__(self): #todo: fix this
        for obj in self.route:
            return obj

    def show(self, track):
        interface = Interface(track, 100)
        interface.show_planned_path(self)

    def __eq__(self, other):
        dx = self.x[0] - other.x[0]
        dy = self.x[1] - other.x[1]
        # strictly not true but done for smoothing ease
        if dx == 0 or dy == 0:
            return True
        return False
