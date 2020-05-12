import numpy as np
from PathPlanner import A_StarPathFinder
import LibFunctions as f
from copy import deepcopy
from Interface import Interface

class Classical:
    def __init__(self, track, car):
        self.track = track
        self.car = car

        self.path = None # path to hold path in

        self.last_action_n = 0

    def get_base_action(self, state):
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

    def plan_rrt(self):
        # write method here that uses random trees to plan a path to a receding horizon around obs
        
