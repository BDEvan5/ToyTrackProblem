import LibFunctions as f 
import numpy as np 
from PathPlanner import get_practice_path, convert_list_to_path, A_StarFinderMod, A_StarTrackWrapper
from StateStructs import Path
from TrackMapInterface import load_map, show_track_path

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from casadi import *

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

    new_path = reduce_diagons(new_path)

    return new_path

def reduce_diagons(path):
    new_path = []
    skip_pts = []
    tol = 0.2
    look_ahead = 5 

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




def smart_casadi_opti(track, path):
    # symbols
    N = len(path)
    seed = np.asarray(path)
    delta_t = 1
    
    x, y = MX.sym('x', N), MX.sym('y', N) 
    theta = MX.sym('theta', N)
    v, w = MX.sym('v', N), MX.sym('w', N) 

    lut = get_look_up_table(track)
    nlp = get_feed_dict(x, y, theta, v, w, seed, delta_t, lut, N)

    S = nlpsol('vert', 'ipopt', nlp, {'ipopt':{'print_level':5}})

    x0 = initSolution(seed)
    lbx, ubx, lbg, ubg = get_bound_cons(seed, N)

    r = S(x0=x0, lbg=lbg, ubg=ubg, ubx=ubx, lbx=lbx)
    x_opt = r['x']

    print(f"X_opt: {x_opt}")

    x_new = np.array(x_opt[0:N])
    y_new = np.array(x_opt[N:2*N])
    th_new = np.array(x_opt[2*N:N*3])
    v_new = np.array(x_opt[3*N:N*4])
    w_new = np.array(x_opt[4*N:N*5])

    path = np.concatenate([x_new, y_new], axis=1)

    return path

# helpers for opt  
def get_heat_map(track):
    # heat_map = preprocess_heat_map(track, False)
    track_map = track.get_heat_map()
    # track_map = add_obstacle_map_boundary(track_map)
    # show_heat_map(track_map)
    track_map = np.asarray(track_map, dtype=np.float32)

    return track_map

def initSolution(seed):
    max_v = 5
    vs = []
    ths = []
    ws = []
    for i, wp in enumerate(seed):
        if i == 0:
            last_wp = wp
            continue
        gradient = f.get_gradient(last_wp, wp)
        theta = np.arctan(gradient) - np.pi/2  # gradient to next point
        v = max_v * (np.pi - abs(theta)) / np.pi

        ths.append(theta)
        vs.append(v)
    ths.append(0)
    vs.append(max_v)

    dt = 1
    for i in range(len(ths)-1):
        w = (ths[i + 1] - ths[i])/dt 
        ws.append(w)
    ws.append(0)

    xs = seed[:, 0]
    ys = seed[:, 1]

    ret = vertcat(xs, ys, ths, vs, ws)

    print(ret)

    return ret

def get_look_up_table(track):
    track_map = get_heat_map(track)

    xgrid = np.linspace(0, 100, 100)
    ygrid = np.linspace(0, 100, 100)
    lut = interpolant('lut', 'bspline', [xgrid, ygrid], track_map.flatten())

    return lut

def get_feed_dict(x, y, theta, v, w, seed, delta_t, lut, N):
    
    nlp = {\
       'x':vertcat(x,y,theta,v,w),
       'f': 5*(sumsqr(v) + sumsqr(w)) + (sumsqr(x-seed[:,0]) + sumsqr(y-seed[:,1])),
       'g':vertcat(\
                   x[1:] - (x[:-1] + delta_t*v[:-1]*cos(theta[:-1])),
                   y[1:] - (y[:-1] + delta_t*v[:-1]*sin(theta[:-1])),
                   theta[1:] - (theta[:-1] + delta_t*w[:-1]),
                   x[0]-seed[0, 0], y[0]-seed[0, 1],
                   x[N-1]-seed[-1, 0], y[N-1]-seed[-1, 1],
                   lut(horzcat(x,y).T).T * 1e8
                  )\
    }

    return nlp 

def get_bound_cons(seed, N):
    box = 60
    lbx_pre = [i-box for i in seed[:, 0]] + [i-box for i in seed[:, 1]] 
    ubx_pre = [i+box for i in seed[:, 0]] + [i+box for i in seed[:, 1]]

    lbg = [0] * (N-1)*3 + [0]*4 + [0]*N
    ubg = ([0]*(N-1)*3 + [0]*4 + [0]*N)

    # lbx = [0]*N + [0]*N + [-np.inf]*N + [0]*N + [-1]*N
    # ubx = [100]*N + [100]*N + [np.inf]*N + [5]*N + [1]*N

    lbx = lbx_pre + [-np.inf]*N + [0]*N + [-1]*N
    ubx = ubx_pre + [np.inf]*N + [5]*N + [1]*N

    return lbx, ubx, lbg, ubg


# tester
def run_opti():
    track = load_map("myTrack3") #todo: save name in map so that other things can shaddow name

    # get a path for the map if saved
    path_file = "DataRecords/maze_path_list.npy"
    try:
        path = np.load(path_file)
    except:
        path = A_StarFinderMod(track, 1)
        np.save(path_file, path)

        show_track_path(track, path)


    path = reduce_path_diag(path)
    path = expand_path(path)
    show_track_path(track, path)

    path = smart_casadi_opti(track, path)

    show_track_path(track, path)



if __name__ == "__main__":
    run_opti()