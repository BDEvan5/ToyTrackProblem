import LibFunctions as f 
import numpy as np 
from copy import deepcopy
from Interface import Interface
from Models import TrackData
from PathPlanner import get_practice_path, Path

from collections import deque
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from scipy import  interpolate as si
from scipy import optimize as so
# from tensorflow.keras.optimizers import RMSprop
# import tensorflow as tf


class PathSmoother:
    def __init__(self):
        self.path = None
        self.track = None

    def reduce_path(self):
        path = self.path
        new_path = []
        new_path.append(path[0]) # starting pos
        pt1 = path[0]
        for i in range(2, len(path)):
            pt2 = path[i]
            if pt1[0] != pt2[0] and pt1[1] != pt2[1]:
                new_path.append(path[i-1]) # add corners
                pt1 = path[i-1]
        new_path.append(path[-1]) # add end

        self.path = new_path

    def expand_path(self):
        path = self.path
        new_path = []
        pt = path[0]
        for i in range(len(path)-1):
            next_pt = path[i+1]

            new_path.append(pt)
            new_pt = [(pt[0]+ next_pt[0])/2, (pt[1]+ next_pt[1])/2]
            new_path.append(new_pt)

            pt = next_pt

        new_path.append(path[-1])

        self.path = new_path

    def show_path(self):
        # helper for debugging
        interface = Interface(track, 100)
        interface.show_planned_path(path)

    def optimize_path(self):
        path = self.path
        bounds = []

        start1 = tuple((self.path[0][0], self.path[0][0]))
        end1 = tuple((self.path[-1][0], self.path[-1][0]))
        start2 = tuple((self.path[0][1], self.path[0][1]))
        end2 = tuple((self.path[-1][1], self.path[-1][1]))
        bounds.append(start1)
        bounds.append(start2)
        for _ in range((len(path)-2)*2):
            bounds.append((0, 100))
        bounds.append(end1)
        bounds.append(end2)

        cons = {'type': 'eq', 'fun':path_constraint}
        
        res = optimize.minimize(self.path_cost, path, bounds=bounds, constraints=cons, method='SLSQP')
        # res = optimize.minimize(path_cost, path)
        print(res)
        path_res = res.x

        new_path = []
        path_opti = Path()
        for i in range(0,len(path_res), 2):
            new_pt = (path_res[i], path_res[i+1])
            new_path.append(new_pt)
            path_opti.add_way_point(new_pt)

        # path opti has actual path in it
        self.path = new_path
        return new_path

    def path_cost(self, path_list):
        path = []
        for i in range(0,len(path_list), 2):
            new_pt = (path_list[i], path_list[i+1])
            path.append(new_pt)

        cost = 0
        for i in range(len(path)-2):
            pt1 = path[i]
            pt2 = path[i+1]
            pt3 = path[i+2]

            dis = f.get_distance(pt1, pt2) 
            angle = f.get_angle(pt1, pt2, pt3)

            cost += dis + (angle ** 2)

        return cost

    def path_constraint(self, path_list):
        track = self.track
        path = []
        for i in range(0,len(path_list), 2):
            new_pt = (path_list[i], path_list[i+1])
            path.append(new_pt)

        ret = 0
        for i in range(len(path)):
            if track._check_collision(path[i]):
                ret += 1 

        return ret

    def run_smoothing(self, track, path_list):
        self.track = track
        self.path = path_list

        self.reduce_path()
        self.expand_path()
        # self.expand_path()

        self.optimize_path()

        return self.path

    def plot_path(self, path):
        fig, ax = plt.subplots()

        paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

        ax.autoscale()
        ax.margins(0.1)
        plt.show()


"""
Additional functions
"""

def reduce_path(path):
    new_path = []
    new_path.append(path[0]) # starting pos
    pt1 = path[0]
    for i in range(2, len(path)):
        pt2 = path[i]
        if pt1[0] != pt2[0] and pt1[1] != pt2[1]:
            new_path.append(path[i-1]) # add corners
            pt1 = path[i-1]
    new_path.append(path[-1]) # add end

    return new_path

def expand_path(path):
    new_path = []
    pt = path[0]
    for i in range(len(path)-1):
        next_pt = path[i+1]

        new_path.append(pt)
        new_pt = [(pt[0]+ next_pt[0])/2, (pt[1]+ next_pt[1])/2]
        new_path.append(new_pt)

        pt = next_pt

    new_path.append(path[-1])

    return new_path

def show_path(path):
    track = TrackData()
    track.simple_maze()

    interface = Interface(track, 100)
    interface.show_planned_path(path)

def plot_new_path(path, new_path):
    fig, ax = plt.subplots()

    paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
    lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
    ax.add_collection(lc2)
    paths = [(new_path[i], new_path[i+1]) for i in range(len(new_path)-1)]
    lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
    ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()

def path_constraint(path_list):
    track = TrackData()
    track.simple_maze()

    path = []
    for i in range(0,len(path_list), 2):
        new_pt = (path_list[i], path_list[i+1])
        path.append(new_pt)

    ret = 0
    for i in range(len(path)):
        ret += track._check_collision(path[i])
    # print(ret)

    return ret # 0 for no col or 1 for col

def optimise_path(path):
    track = preprocess_heat_map()

    def path_cost(path_list):
        path = []
        for i in range(0,len(path_list), 2):
            new_pt = (path_list[i], path_list[i+1])
            path.append(new_pt)
        # path = path_list

        dis_cost = 0
        obs_cost = 0
        c_distance = 0.1
        for i in range(len(path)-2):
            pt1 = path[i]
            pt2 = path[i+1]
            pt3 = path[i+2]

            if pt1[0] > 100 or pt1[1] > 100:
                dis_cost += 1000
            else:
                dis = f.get_distance(pt1, pt2) 
                angle = f.get_angle(pt1, pt2, pt3)

                dis_cost += dis
                dis_cost += (angle ** 2) * 0.5
                obs_cost += (track[int(pt1[0])-1, int(pt1[1])-1] ** 2) * c_distance
        
        dis_cost += f.get_distance(path[-2], path[-1])

        cost = dis_cost + obs_cost
        print(f"Distance Cost: {dis_cost} --> Obs cost: {obs_cost} --> Total cost: {cost}")

        return cost

    bounds = set_up_bounds(path)    
    path = np.asarray(path)
    path = path.flatten()
    res = so.minimize(path_cost, path, bounds=bounds, method='trust-constr')

    print(res)
    path_res = res.x


    new_path = []
    path_opti = Path()
    for i in range(0,len(path_res), 2):
        new_pt = (path_res[i], path_res[i+1])
        new_path.append(new_pt)
        path_opti.add_way_point(new_pt)
    # else:
    #     path_opti = Path()
    #     for pt in path_res:
    #         path_opti.add_way_point(pt)
    #     new_path = path_res

    # path opti has actual path in it

    return new_path, path_opti

def minimize_w_tf(path):
    track = preprocess_heat_map()

    path  = tf.convert_to_tensor(path)

    def path_cost_tf():
        dis_cost = 0
        obs_cost = 0
        c_distance = 2
        for i in range(len(path)-2):
            pt1 = path[i]
            pt2 = path[i+1]
            pt3 = path[i+2]

            dis = f.get_distance(pt1, pt2) 
            angle = f.get_angle(pt1, pt2, pt3)

            dis_cost += dis
            dis_cost += (angle ** 2)
            obs_cost += track[int(pt1[0])-1, int(pt1[1])-1] * c_distance
        
        dis_cost += f.get_distance(path[-2], path[-1])

        print(f"Distance Cost: {dis_cost} --> Obs cost: {obs_cost}")
        cost = dis_cost + obs_cost
        cost = tf.convert_to_tensor(cost, dtype=tf.float32)

        return cost

    # bounds = set_up_bounds(path)    

    opti = RMSprop()
    trainable_vars = path[1:-1]
    for i in range(500):
        opti.minimize(loss=path_cost_tf, var_list=trainable_vars) # not the first and last variables

    path_opti = Path()
    for i in range(0,len(path), 2):
        path_opti.add_way_point([path[i], path[i+1]])

    # path opti has actual path in it

    return path, path_opti

    
def set_up_bounds(path):
    bounds = []

    start1 = tuple((path[0][0], path[0][0]))
    end1 = tuple((path[-1][0], path[-1][0]))
    start2 = tuple((path[0][1], path[0][1]))
    end2 = tuple((path[-1][1], path[-1][1]))
    bounds.append(start1)
    bounds.append(start2)
    for _ in range((len(path)-2)*2):
        bounds.append((0, 100))
    bounds.append(end1)
    bounds.append(end2)

    return bounds

def preprocess_heat_map():
    track = TrackData()
    track.simple_maze()
    track_map = track.get_heat_map()

    for _ in range(5): # blocks up to 5 away will start to have a gradient
        for i in range(1, 98):
            for j in range(1, 98):
                left = track_map[i-1, j]
                right = track_map[i+1, j]
                up = track_map[i, j+1]
                down = track_map[i, j-1]

                track_map[i, j] = max(sum((left, right, up, down)) / 3, track_map[i, j])

    return track_map 
            

def show_track_map():
    track = preprocess_heat_map()
    x = np.array([i for i in range(100)])
    y = np.array([i for i in range(100)])

    fig = plt.figure(3)
    ax = plt.gca()

    im = ax.imshow(track)
    plt.show()


# run functions
def run_path_opti():
    # show_track_map()
    path = get_practice_path()

    path = reduce_path(path)
    path = expand_path(path)
    # path = expand_path(path)

    path_list, path_obj = optimise_path(path)
    # old_opti(path)

    show_path(path_obj)

def run_tf_opti():
    path = get_practice_path()

    path = reduce_path(path)
    path = expand_path(path)

    path_list, path_opti = minimize_w_tf(path)

    show_path(path_opti)


if __name__ == "__main__":
    run_path_opti()
    # run_tf_opti()

