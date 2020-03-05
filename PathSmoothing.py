import numpy as np 

class TrackSmoothing:
    def __init__(self, track, car):
        self.car = car
        self.track = track

        self.path_in = None

    def add_path(self, planned_path):
        self.path_in = planned_path

    def add_velocity(self):
        # set up last wp in each cycle
        for i, wp in enumerate(self.path_in):
            if i == 0:
                last_wp = wp
                continue
            dx = wp.x[0] - last_wp.x[0]
            dy = wp.x[1] - last_wp.x[1]
            if dy != 0:
                gradient = dx/dy  #flips to make forward theta = 0
            else:
                gradient = 1000
            last_wp.theta = np.arctan(gradient)  # gradient to next point
            last_wp.v = self.car.max_v * (np.pi - last_wp.theta) / np.pi

            last_wp = wp

        self.path_in[len(self.path_in)-1].theta = 0 # set the last point
        self.path_in[len(self.path_in)-1].v = self.car.max_v
    
    def get_path(self):
        return self.path_in