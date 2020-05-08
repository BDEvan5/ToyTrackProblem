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

def path_cost(path_list):
    track = TrackData()
    track.simple_maze()
    path = []
    for i in range(0,len(path_list), 2):
        new_pt = (path_list[i], path_list[i+1])
        path.append(new_pt)

    cost = 0
    c_distance = 1
    for i in range(len(path)-2):
        pt1 = path[i]
        pt2 = path[i+1]
        pt3 = path[i+2]

        dis = f.get_distance(pt1, pt2) 
        angle = f.get_angle(pt1, pt2, pt3)

        cost += dis
        # cost += dis + (angle ** 2)
        dis_to_obs = track.get_obstacle_distance(pt1)
        print(dis_to_obs)
        cost += c_distance * dis_to_obs **2
    
    cost += f.get_distance(path[-2], path[-1])

    return cost

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

    cons = {'type': 'eq', 'fun':path_constraint}
    
    # res = so.minimize(path_cost, path, bounds=bounds, constraints=cons)
    res = so.minimize(path_cost, path, bounds=bounds)
    print(res)
    path_res = res.x

    new_path = []
    path_opti = Path()
    for i in range(0,len(path_res), 2):
        new_pt = (path_res[i], path_res[i+1])
        new_path.append(new_pt)
        path_opti.add_way_point(new_pt)

    # path opti has actual path in it

    return new_path, path_opti


# run functions
def run_path_opti():
    path = get_practice_path()

    path = reduce_path(path)
    path = expand_path(path)

    path_list, path_obj = optimise_path(path)
    # old_opti(path)

    show_path(path_obj)



if __name__ == "__main__":
    run_path_opti()

