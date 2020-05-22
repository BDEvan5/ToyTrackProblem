import numpy as np
from PathPlanner import A_StarPathFinder, WayPoint
import LibFunctions as f




class ControlSystem:
    def __init__(self):
        self.k_th_ref = 0.1 # amount to favour v direction
        self.k_th = 0.8
        self.L = 1

    def __call__(self, state, destination):
        # print(glbl_wp.x)
        # print(state.x)
        x_ref = destination.x 
        v_ref = destination.v 
        th_ref = destination.theta

        x = state[0]
        y = state[1]
        v = state[2]
        theta = state[3]

        # run v control
        e_v = v_ref - v # error for controler
        a = self._acc_control(e_v)

        # run th control
        x_ref_th = self._get_xref_th([x, y], x_ref)
        th_ref_combined = th_ref * self.k_th_ref + x_ref_th * (1- self.k_th_ref) # no feedback
        # print(f"Theta ref: {th_ref_combined}")
        new_v = v + a
        # e_th = abs(th_ref_combined) - abs(state.theta)
        e_th = th_ref_combined - theta
        if e_th > np.pi: # there is a problem here, the angle between -3.14 and + 3.14 is wrong
            e_th = 2 * np.pi - e_th
        if e_th < - np.pi:
            e_th = e_th + 2 * np.pi

        delta = np.arctan(e_th * self.L / (new_v)) * 1

        if abs(delta) > 1:
            print(f"Probelms: delta trying to be {delta} which is >1") 
            # raise ValueError

        action = [a, delta]
        return action

    def _acc_control(self, e_v):
        # this function is the actual controller
        k = 0.25
        friction_c = 1
        return k * e_v + friction_c

    def _th_controll(self, e_th):
        # theta controller to come here when dth!= th

        th = e_th * self.k_th

        th = max(-np.pi, e_th)
        th = min(np.pi, e_th)

        return th

    def _get_xref_th(self, x1, x2):
        dx = x2[0] - x1[0]
        dy = x2[1] - x1[1]
        # # self.logger.debug("x1: " + str(x1) + " x2: " + str(x2))
        # # self.logger.debug("dxdy: %d, %d" %(dx,dy))
        if dy != 0:
            ret = np.abs(np.arctan(dx / dy))
        else:
            ret = np.pi / 2

        # grad = f.get_gradient(x1, x2)
        # ret = abs(np.arctan((grad ** (-1))))

        # sort out the sin
        sign = 1
        if dx < 0.1:
            sign = -1
        if dy > 0: # dy is opposite to normal
            ret = np.pi - np.abs(ret)
        return abs(ret) * sign


class Tracker:
    def __init__(self, path):
        self.path = path.route
        self.n_inds = len(self.path) -1

        self.pind = 0
        self.control_system = ControlSystem()

    def act(self, state):
        # self.pind = # the points I have passed
        location = state[0:4]
        # nn_state = state[2::]

        car_dist = f.get_distance(self.path[self.pind].x, state[0:2])
        ds = f.get_distance(self.path[self.pind].x, self.path[self.pind+1].x)
        if car_dist > ds:
            self.pind += 1

        destination = self.path[self.pind+1] # next point

        ref_action = self.control_system(location, destination)



        return ref_action





