import numpy as np
from PathPlanner import A_StarPathFinder, RRT_PathFinder
import LibFunctions as f
from copy import deepcopy

class Classical:
    def __init__(self, track, car):
        self.track = track
        self.car = car

        self.path_finder = A_StarPathFinder(track)
        self.path_finder = RRT_PathFinder(track)
        self.path = None # path to hold path in

        self.last_action_n = 0

    def plan_path(self):
        self.path = self.path_finder.run_search(5)
        # self.smooth_track()
        self.add_velocity()

    def get_single_path(self):
        self.track.add_way_point(self.track.start_location)
        self.track.add_way_point(self.track.end_location)
        self.track.route[1].v = self.car.max_v

    def add_velocity(self):
        # set up last wp in each cycle
        path = self.path.route
        for i, wp in enumerate(path):
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

        path[len(path)-1].theta = 0 # set the last point
        path[len(path)-1].v = self.car.max_v

    def smooth_track(self):
        weight_data = 0.2
        weight_smooth = 0.05
        tolerance = 0.00001

        path = deepcopy(self.path.route)
        new_path = []
        for pt in path:
            p = deepcopy(pt)
            p.x = [0, 0]
            new_path.append(p)
        new_path[0] = deepcopy(path[0])
        new_path[len(new_path)-1] = deepcopy(path[len(new_path)-1])


        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path)-1):
                for j in range(2):
                    aux = new_path[i].x[j]
                    aux = new_path[i].x[j]
                    new_path[i].x[j] += weight_data * (path[i].x[j] - new_path[i].x[j])
                    new_path[i].x[j] += weight_smooth * (new_path[i-1].x[j] + new_path[i+1].x[j] - 2*new_path[i].x[j])
                    change += abs(aux - new_path[i].x[j])

        self.path.route = new_path

        # for i in range(len(path)):
        #     print('[' +', '.join('%.3f'%x for x in path[i].x) +'] -> [' +', '.join('%.3f'%x for x in new_path[i].x) + ']')

    def get_action(self, state):
        location = state[0:2] # check this state vector
        
        while self.check_wp_increment(location):
            self.last_action_n += 1

        wp_return = self.path.route[self.last_action_n]
        print(f"Location: {location} WP:  {wp_return.x}")
        return wp_return

    def check_wp_increment(self, location):
        last_wp = self.path.route[self.last_action_n].x
        next_wp = self.path.route[self.last_action_n + 1].x
        # last_dis = f.get_distance(last_wp, location)
        wp_dis = f.get_distance(last_wp, next_wp)
        next_dis = f.get_distance(next_wp, location)

        if next_dis < wp_dis * 1.2 : # 1.2 is for error to make the circle a bit bigger
            return True
        return False

    def reset(self):
        # possibly find path here except it wastes computation time.
        self.last_action_n = 0


