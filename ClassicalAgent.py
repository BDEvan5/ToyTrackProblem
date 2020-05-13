import numpy as np
from PathPlanner import RTT_StarPathFinder, WayPoint
import LibFunctions as f
from copy import deepcopy
from Interface import Interface

class Classical:
    def __init__(self, track, car):
        self.track = track
        self.car = car

        self.path = None # path to hold path in

        self.last_action_n = 0
        self.rrt = RTT_StarPathFinder(track)

    def get_base_action(self, state):
        location = state[0:2] # check this state vector
        
        while self.check_wp_increment(location):
            self.last_action_n += 1

        wp_return = self.path.route[self.last_action_n]
        print(f"Location: {location} WP:  {wp_return.x}")
        return wp_return

    def get_action(self, state):
        location = state[0:2]
        path = self.plan_rrt(location)

        pt = min(4, len(path)-1)

        action = path[pt-1] # the third rrt path pt

        action_wp = WayPoint()
        grad = f.get_gradient(path[pt-1], path[pt])
        theta = np.arctan(grad)
        v = self.car.max_v * (np.pi - theta) / np.pi
        action_wp.set_point(action, v, theta)

        return action_wp

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

    def plan_rrt(self, location):
        # write method here that uses random trees to plan a path to a receding horizon around obs
        horizon = 3 # way points ahead

        wp_n = min(self.last_action_n +horizon, len(self.path))
        
        start = location 
        end = self.path.route[wp_n].x

        path = self.rrt.run_wp_search(start, end)
        return path 

