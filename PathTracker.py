import numpy as np
from PathPlanner import A_StarPathFinder, WayPoint
from PathOptimisation import optmise_track_path
import LibFunctions as f




class ControlSystem:
    def __init__(self):
        self.k_th_ref = 0.1 # amount to favour v direction

    def __call__(self, state, destination):
        # print(glbl_wp.x)
        # print(state.x)
        x_ref = destination.x 
        v_ref = destination.v 
        th_ref = destination.theta

        # run v control
        e_v = v_ref - state.v # error for controler
        a = self._acc_control(e_v)

        # run th control
        x_ref_th = self._get_xref_th(state.x, x_ref)
        e_th = th_ref * self.k_th_ref + x_ref_th * (1- self.k_th_ref) # no feedback
        th = self._th_controll(e_th)

        action = [a, th]
        return action

    def _acc_control(self, e_v):
        # this function is the actual controller
        k = 0.25
        friction_c = 1
        return k * e_v + friction_c

    def _th_controll(self, e_th):
        # theta controller to come here when dth!= th
        # e_th = min(np.pi/2, e_th)
        # e_th = max(-np.pi/2, e_th) # clips the action to pi 

        return e_th

    def _get_xref_th(self, x1, x2):
        dx = x2[0] - x1[0]
        dy = x2[1] - x1[1]
        # self.logger.debug("x1: " + str(x1) + " x2: " + str(x2))
        # self.logger.debug("dxdy: %d, %d" %(dx,dy))
        if dy != 0:
            ret = np.abs(np.arctan(dx / dy))
        else:
            ret = np.pi / 2

        # sort out the sin
        sign = 1
        if dx < 0:
            sign = -1
        if dy > 0: # dy is opposite to normal
            ret = np.pi - np.abs(ret)
        return ret * sign


class Tracker:
    def __init__(self, path):
        self.path = path.route
        self.n_inds = len(self.path) -1

        self.pind = 0

        self.control_system = ControlSystem()

    def act(self, state):
        HORIZON = 1
        location = state.x

        ind = self.get_nearest_ind(location)
        if ind > self.pind: # pind is where I am now and want to work forward from.
            self.pind = ind

        ind = min(ind + HORIZON, self.n_inds) # checks that it isn't past the end indicie  
        destination = self.path[ind]

        destination.print_point(f"Destination: {ind}")

        ref_action = self.control_system(state, destination)

        return ref_action

    def get_nearest_ind(self, location):
        SEARCH_PTS = 5
        d = [f.get_distance(location, wpt.x) for wpt in self.path[self.pind:(self.pind+SEARCH_PTS)]]

        pt = min(d)
        ind = d.index(pt) + self.pind

        return ind







