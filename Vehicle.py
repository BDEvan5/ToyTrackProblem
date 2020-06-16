"""
This is the vehicle with planners and controllers.
There are three levels of planning
This will be one day implemented in ros on the actual vehicle
"""
import numpy as np 

from PathPlanner import A_StarFinderMod
# import agent
import LibFunctions as lib

""" This is basically a wrapper for the three different systems """
class Vehicle:
    def __init__(self):
        self.local_planner = LocalPlanner()
        self.control_system = ControlSystem()

        # self.agent = Agent()

    def get_action(self, state):
        target = self.local_planner(state)

        # write code here to modify the action based on the agent
        # write function to get relative desitnation then use that for nn and controller

        action = self.control_system(state, target)

        return action

    def plan_path(self, track, load):
        name = track.name + "_path_list.npy"
        if load:
            full_path = np.load(name)
        else:
            path = A_StarFinderMod(track, 1)
            path = reduce_path_diag(path) #(n,2)
            full_path = add_velocity(path)  #(n, 4)
            np.save(name, full_path)

        self.local_planner.set_path(full_path)

        return full_path

    def reset(self):
        self.local_planner.pind = 0

class LocalPlanner:
    def __init__(self):
        self.path = None
        self.pind = 0
        self.n_inds = 0

    def __call__(self, state):
        assert self.path is not None, "No path set in local planner"

        d_car_pind = lib.get_distance(self.path[self.pind][0:2], state[0:2])
        d_wpts = lib.get_distance(self.path[self.pind][0:2], self.path[self.pind+1][0:2])

        while d_car_pind > d_wpts:
            if self.pind < (self.n_inds - 2):
                self.pind += 1
            d_car_pind = lib.get_distance(self.path[self.pind], state[0:2])
            d_wpts = lib.get_distance(self.path[self.pind][0:2], self.path[self.pind+1][0:2])

        target = self.path[self.pind + 1]

        return target

    def set_path(self, path):
        self.path = path
        self.n_inds = len(self.path)

class ControlSystem:
    def __init__(self):
        self.k_th_ref = 0.1 # amount to favour v direction
        self.k_th = 0.8
        self.L = 1

    def __call__(self, state, destination):
        x_ref = destination[0:2]
        v_ref = destination[2] 
        th_ref = destination[3]

        x = state[0]
        y = state[1]
        v = state[2] * 5
        theta = state[3] * np.pi # this undoes the scaling

        # run v control
        e_v = v_ref - v # error for controler
        a = self._acc_control(e_v)

        # run th control
        x_ref_th = self._get_xref_th([x, y], x_ref)
        th_ref_combined = th_ref * self.k_th_ref + x_ref_th * (1- self.k_th_ref) # no feedback
        new_v = v + a
        e_th = th_ref_combined - theta
        if e_th > np.pi:
            e_th = 2 * np.pi - e_th
        if e_th < - np.pi:
            e_th = e_th + 2 * np.pi

        delta = np.arctan(e_th * self.L / (new_v)) * 1

        if abs(delta) > 1.5:
            print(f"Probelms: delta trying to be {delta} which is >1.5") 
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
        if dy != 0:
            ret = np.abs(np.arctan(dx / dy))
        else:
            ret = np.pi / 2

        # sort out the sign
        sign = 1
        if dx < 0.1:
            sign = -1
        if dy > 0: # dy is opposite to normal
            ret = np.pi - np.abs(ret)
        return abs(ret) * sign



# path helpers
def reduce_path_diag(path):
    new_path = []
    new_path.append(path[0]) # starting pos
    pt1 = path[0]
    for i in range(2, len(path)-1): 
        pt2 = path[i]
        if pt1[0] == pt2[0] or pt1[1] == pt2[1]:
            continue
        if abs(pt1[1] - pt2[1]) == abs(pt1[0] - pt2[0]): # if diagonal
            continue

        new_path.append(path[i-1]) # add corners
        pt1 = path[i-1]

    new_path.append(path[-2]) # add end
    new_path.append(path[-1]) # add end
     
    print(f"Path Reduced from: {len(path)} to: {len(new_path)}  points by straight analysis")

    return new_path

def reduce_diagons(path):
    new_path = []
    skip_pts = []
    tol = 0.2
    look_ahead = 5 # number of points to consider on line
    # consider using a distance lookahead too

    for i in range(len(path) - look_ahead):
        pts = []
        for j in range(look_ahead):
            pts.append(path[i + j]) # generates a list of current points

        grads = []
        for j in range(1, look_ahead):
            m = f.get_gradient(pts[0], pts[j])
            grads.append(m)

        for j in range(look_ahead -2):
            ddm = abs((grads[j] - grads[-1])) / (abs(grads[-1]) + 0.0001) # div 0
            if ddm > tol: # the grad is within tolerance
                continue
            index = i + j + 1
            if index in skip_pts: # no repeats
                continue

            skip_pts.append(j + i + 1)        

    for i in range(len(path)):
        if i in skip_pts:
            continue
        new_path.append(path[i])

    print(f"Number of skipped pts: {len(skip_pts)}")

    return new_path

def add_velocity(path):
    new_path = np.zeros((len(path), 4))
    new_path[:, 0:2] = path

    v = 5
    for i in range(len(path)-1):
        gradient = lib.get_gradient(path[i-1], path[i])
        new_path[i, 2] = np.arctan(gradient) - np.pi/2 # th
        new_path[i, 3] = v # cosnt velocity for the moment

    new_path[-1, 2] = 0
    new_path[-1, 3] = v

    return new_path

