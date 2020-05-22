import LibFunctions as f 
import numpy as np 
from Models import TrackData
from PathPlanner import get_practice_path, convert_list_to_path
from StateStructs import Path

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
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
        dis = f.get_distance(pt1, pt2)
        # if dis < 5:
        #     continue
        

        new_path.append(path[i-1]) # add corners
        pt1 = path[i-1]

    new_path.append(path[-2]) # add end
    new_path.append(path[-1]) # add end
     
    print(f"Path Reduced from: {len(path)} to: {len(new_path)}  points")

    # path = reduce_diagons(new_path)
    # print(len(path))

    # path = reduce_diagons(path)
    # print(len(path))

    # path = reduce_diagons(path)
    # print(len(path))

    return new_path

def reduce_diagons(new_path):
    two_new_path = []
    for i in range(0, len(new_path)-2):
        pt1 = new_path[i]
        pt2 = new_path[i+1]
        pt3 = new_path[i+2]

        m1 = f.get_gradient(pt1, pt2)
        m2 = f.get_gradient(pt1, pt3)
        dm = abs(abs(m1) - abs(m2))
        if dm > 0.25: # tolerance for different grads
            two_new_path.append(pt1)
        elif f.get_distance(pt1, pt2) > 15:
            two_new_path.append(pt1) # not too far apart

    two_new_path.append(new_path[-2])
    two_new_path.append(new_path[-1])

    return two_new_path


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

def preprocess_heat_map(track=None, show=False):
    # if track is None:
    #     track = TrackData()
    #     track.simple_maze()
    track_map = track.get_heat_map()
    track_map = np.asarray(track_map, dtype=np.float32)

    for _ in range(10): # blocks up to 5 away will start to have a gradient
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

    if show:
        show_heat_map(track_map)

    return track_map 

def show_heat_map(track_map, view3d=True):
    if view3d:
        xs = [i for i in range(100)]
        ys = [i for i in range(100)]
        X, Y = np.meshgrid(xs, ys)

        fig = plt.figure()


        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, track_map, cmap='plasma')
        fig.colorbar(surf)
    else:
        ax = plt.gca()
        im = ax.imshow(track_map)

    plt.show()

    
 
def calcTrajOpti(path):
    traj_map = preprocess_heat_map()

    def trajectory_objective(traj):
        A = 200
        traj = np.reshape(traj, (-1, 2))
        for pt in traj:
            if pt[0] > 100 or pt[1] > 100:
                return 20 * pt[0] * pt[1]

    
        distances = np.array([f.get_distance(traj[i], traj[i+1]) for i in range(len(traj)-1)])
        cost_distance = np.sum(distances)

        obstacle_costs = np.array([traj_map[int(pt[0]-1), int(pt[1])] ** 0.5 for pt in traj])
        obs_cost = np.sum(obstacle_costs) * A

        cost = cost_distance + obs_cost
        # print(f"Cost: {cost} -> dis: {cost_distance} -> Obs: {obs_cost}")
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

    mthds = ['trust-constr', 'Powell', 'Nelder-Mead', 'SLSQP']
    x0 = np.array(path).flatten()
    x = so.minimize(fun=trajectory_objective, x0=x0, bounds=bnd, method=mthds[0])
    print(f"Message: {x.message}")
    print(f"Success: {x.success}")
    print(f"Result: {x.fun}")
    path = np.reshape(x.x, (-1, 2))
    # print(f"Path: {path}")

    path_obj = convert_list_to_path(path)
    # path_obj.show()

    return path

def optimise_track_trajectory(path, track):
    # traj_map = preprocess_heat_map(track, False)
    traj_map = preprocess_heat_map(track, True)
 
    def trajectory_objective(traj):
        A = 1
        traj = np.reshape(traj, (-1, 2))
        for pt in traj:
            if pt[0] > 100 or pt[1] > 100:
                return 20 * pt[0] * pt[1]

    
        # distances = np.array([f.get_distance(traj[i], traj[i+1]) for i in range(len(traj)-1)])
        # cost_distance = np.sum(distances)

        obstacle_costs = np.array([traj_map[int(pt[0]-1), int(pt[1])] ** 0.5 for pt in traj])
        obs_cost = np.sum(obstacle_costs) * A

        # cost = cost_distance + obs_cost
        # print(f"Cost: {cost} -> dis: {cost_distance} -> Obs: {obs_cost}")
        return obs_cost

    def get_bounds():
        start1 = tuple((path[0][0], path[0][0]))
        end1 = tuple((path[-1][0], path[-1][0]))
        start2 = tuple((path[0][1], path[0][1]))
        end2 = tuple((path[-1][1], path[-1][1]))
        bnd = [(0, 100)]*((len(path)-2)*2)
        bnd.insert(0, start1)
        bnd.insert(1, start2)
        bnd.insert(-1, end1)
        bnd.insert(-1, end2)

        return bnd

    bnds = get_bounds()

    mthds = ['trust-constr', 'Powell', 'Nelder-Mead', 'SLSQP']
    x0 = np.array(path).flatten()
    x = so.minimize(fun=trajectory_objective, x0=x0, bounds=bnds, method=mthds[3])

    print(f"Message: {x.message}")
    print(f"Success: {x.success}")
    print(f"Result: {x.fun}")
    path = np.reshape(x.x, (-1, 2))
    # print(f"Path: {path}")

    path_obj = convert_list_to_path(path)
    # path_obj.show()

    return path



def run_traj_opti():
    path = get_practice_path()
    path = reduce_path(path)
    path = expand_path(path)
    path = expand_path(path)

    path = calcTrajOpti(path)

    # path = convert_list_to_path(path)
    # path.show()


#external call
def optimise_trajectory(path):
    # path = reduce_path(path)
    # path = expand_path(path)
    # path = expand_path(path)

    # path = calcTrajOpti(path)

    return path

def add_velocity(path_obj, car=None):
    max_v = 5
    if car is not None:
        max_v = car.max_v

    new_path = Path()
    if type(path_obj) != type(new_path):
        path_obj = convert_list_to_path(path_obj)

    path = path_obj.route
    for i, wp in enumerate(path):
        if i == 0:
            last_wp = wp
            continue
        gradient = f.get_gradient(last_wp.x, wp.x)
        theta = np.arctan(gradient) - np.pi/2  # gradient to next point
        # if theta < 0: # negative theta
        #     theta = 2 * np.pi + theta
        last_wp.theta = theta
        last_wp.v = max_v * (np.pi - abs(last_wp.theta)) / np.pi

        last_wp = wp

    path[len(path)-1].theta = 0 # set the last point
    path[len(path)-1].v = max_v

    return path_obj





if __name__ == "__main__":
    run_traj_opti()
