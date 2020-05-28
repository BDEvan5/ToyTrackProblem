import numpy as np 

import LibFunctions as f 
from StateStructs import WayPoint, Path
from TrackMapInterface import show_track_path



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

        # delta_grads = []
        # for j in range(look_ahead - 1):
        #     dm = abs(grads[-1] - grads[j])
        #     delta_grads.append(dm)

        for j in range(look_ahead -2):
            ddm = abs((grads[j] - grads[-1])) / (abs(grads[-1]) + 0.0001) # div 0
            if ddm > tol: # the grad is within tolerance
                continue
            index = i + j + 1
            if index in skip_pts: # no repeats
                continue
            # cur_max_index = max(skip_pts) 
            # dis = f.get_distance(pts[0], path[index])
            # if dis > 15: # the distance to nearest point isn't too big.
            #     continue
            skip_pts.append(j + i + 1)        

    for i in range(len(path)):
        if i in skip_pts:
            continue
        new_path.append(path[i])

    print(f"Number of skipped pts: {len(skip_pts)}")

    return new_path


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
    # new_path.remove(new_path[1])
    # new_path.remove(new_path[-2])

    return new_path

def add_velocity(path):
    v = 5 # const v
    for i, wp in enumerate(path.route):
        if i == 0:
            last_wp = wp
            continue
        gradient = f.get_gradient(last_wp.x, wp.x)
        last_wp.theta = np.arctan(gradient) - np.pi/2
        last_wp.v = v

        last_wp = wp 

    path.route[-1].v = v 
    path.route[-1].theta = 0 

    return path

    


    return path

def convert_list_to_path(path):
    new_path = Path()
    for pt in path:
        new_path.add_way_point(pt)

    return new_path


# external total call
def process_path(path):
    path = reduce_path_diag(path)
    # path = reduce_diagons(path)
    path = expand_path(path)

    path_obj = convert_list_to_path(path)

    return path_obj, path

