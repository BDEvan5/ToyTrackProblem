import LibFunctions as f 
import numpy as np 
from Models import TrackData
from PathPlanner import get_practice_path, Path, convert_list_to_path

import matplotlib.pyplot as plt
from scipy import optimize as so


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
    new_path.remove(new_path[1])
    new_path.remove(new_path[-2])

    return new_path

def preprocess_heat_map(track=None):
    if track is None:
        track = TrackData()
        track.simple_maze()
    track_map = track.get_heat_map()

    for _ in range(1): # blocks up to 5 away will start to have a gradient
        for i in range(1, 98):
            for j in range(1, 98):
                left = track_map[i-1, j]
                right = track_map[i+1, j]
                up = track_map[i, j+1]
                down = track_map[i, j-1]

                # logical directions, not according to actual map orientation
                left_up = track_map[i-1, j+1] *3
                left_down = track_map[i-1, j-1]*3
                right_up = track_map[i+1, j+1]*3
                right_down = track_map[i+1, j-1]*3

                obs_sum = sum((left, right, up, down, left_up, left_down, right_up, right_down))
                track_map[i, j] = max(obs_sum / 16, track_map[i, j])

    return track_map 
   
def calcTrajOpti(path):
    traj_map = preprocess_heat_map()

    def trajectory_objective(traj):
        A = 20

        traj = np.reshape(traj, (-1, 2))
        distances = np.array([f.get_distance(traj[i], traj[i+1]) for i in range(len(traj)-1)])
        cost_distance = np.sum(distances)

        obstacle_costs = np.array([traj_map[int(pt[0]-1), int(pt[1])] ** 0.5 for pt in traj])
        obs_cost = np.sum(obstacle_costs)

        cost = cost_distance + obs_cost * A
        print(f"Cost: {cost} -> dis: {cost_distance} -> Obs: {obs_cost}")
        return cost

    def traj_constraint(traj):
        traj = np.reshape(traj, (-1, 2))
        # put constraint in here for the next state being related to the turning angle *t

    start1 = tuple((path[0][0], path[0][0]))
    end1 = tuple((path[-1][0], path[-1][0]))
    start2 = tuple((path[0][1], path[0][1]))
    end2 = tuple((path[-1][1], path[-1][1]))
    bnd = [(0, 100)]*((len(path)-2)*2)
    bnd.insert(0, start1)
    bnd.insert(1, start2)
    bnd.insert(-1, end1)
    bnd.insert(-1, end2)

    x = so.minimize(fun=trajectory_objective, x0=path, bounds=bnd)
    print(f"Message: {x.message}")
    print(f"Success: {x.success}")
    print(f"Result: {x.fun}")
    path = np.reshape(x.x, (-1, 2))
    print(f"Path: {path}")

    path_obj = convert_list_to_path(path)
    path_obj.show()

    return path


def run_traj_opti():
    path = get_practice_path()
    path = reduce_path(path)
    path = expand_path(path)
    path = expand_path(path)

    path = calcTrajOpti(path)

    # path = convert_list_to_path(path)
    # path.show()






if __name__ == "__main__":
    run_traj_opti()
