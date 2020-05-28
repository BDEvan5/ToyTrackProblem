import LibFunctions as f 
import numpy as np 
from Models import TrackData
from PathPlanner import get_practice_path, convert_list_to_path, A_StarFinderMod, A_StarTrackWrapper
from StateStructs import Path
from TrackMapInterface import load_map, show_track_path
from Interface import show_path

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize as so
from scipy import interpolate as ip 
import cvxpy
from casadi import *
# import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


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

    new_path = reduce_diagons(new_path)

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

def show_point_heat_map(track):
    track_map = preprocess_heat_map(track, False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = ys = range(100)
    X, Y = np.meshgrid(xs, ys)

    ax.scatter(X, Y, track_map, marker='x')
    plt.show()

def get_heat_map_eqn(track_map):
    # aim is to get a polynomial function for the heat map
    x = range(100)
    y = range(100)
    # X, Y = np.meshgrid(x, y)
    new_map = np.reshape(track_map, (10000, 1))
    deg = np.asarray(10)

    pre_vals = [10, 10]
    pre_vals = np.reshape(pre_vals, (1, -1))

    poly = PolynomialFeatures(2)
    d = []
    for i in x:
        for j in y:
            d.append([i, j])
    # print(d)
    trans = poly.fit_transform(d)
    _pre = poly.fit_transform(pre_vals)

    clf = LinearRegression()
    clf.fit(trans, new_map)

    prediction = clf.predict(_pre)

    # print(f" Trans: {trans}")
    print(f"Pre: {_pre}")
    print(f" Predict: {prediction}")
    c = clf.coef_[0, :]
    print(f"Res: {c}")

    p = pre_vals[0, :]
    res = claculate_value(pre_vals[0, :], clf)

    print(f"Res 2: {res}") 

    return clf

def claculate_value(p, clf):
    c = clf.coef_[0, :]
    i = clf.intercept_

    res = c[0] + c[1] * p[0] + c[2] * p[1] + c[3] * p[0]**2 + c[4] * p[0]*p[1] + c[5] * p[1]**2
    res += i 

    return res

def plot_interp():
    track = load_map()
    traj_map = preprocess_heat_map(track, False)

    clf = get_heat_map_eqn(traj_map)

    new_map = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            new_map[i, j] = claculate_value([i, j], clf)

    show_heat_map(new_map)

    # x = np.linspace(0, 100, 1000)
    # y = np.linspace(0, 100, 1000)
    # X, Y = np.meshgrid(x, y)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # new_map = []
    # for i in x:
    #     for j in y:
    #         d = claculate_value([i, j], clf)
    #         new_map.append(d)

    # ax.scatter(X, Y, new_map, marker='x')
    # plt.show()
                    
 
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
    traj_map = preprocess_heat_map(track, False)
    # traj_map = preprocess_heat_map(track, True)
 
    def trajectory_objective(traj):
        A = 20
        traj = np.reshape(traj, (-1, 2))
        for pt in traj:
            if pt[0] >= 100 or pt[1] >= 100:
                return 20 * pt[0] * pt[1]

    
        distances = np.array([f.get_distance(traj[i], traj[i+1]) for i in range(len(traj)-1)])
        cost_distance = np.sum(distances)

        obstacle_costs = np.array([traj_map[int(pt[0]-1), int(pt[1])] ** 0.5 for pt in traj])
        obs_cost = np.sum(obstacle_costs) * A
        # obs_cost += traj_map[int(traj[-1][0]-1), int(traj[-1][1]) * 50

        cost = obs_cost
        # cost += cost_distance
        print(f"Cost: {cost} -> dis: {cost_distance} -> Obs: {obs_cost}")
        return cost

    def get_bounds():
        start1 = tuple((path[0][0], path[0][0]))
        end1 = tuple((path[-1][0], path[-1][0]))
        start2 = tuple((path[0][1], path[0][1]))
        end2 = tuple((path[-1][1], path[-1][1]))
        bnd = [(0, 100)]*((len(path)-2)*2)
        bnd.insert(0, start1)
        bnd.insert(1, start2)
        bnd.insert(-1, end2)
        bnd.insert(-1, end1)

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

def cvxpy_optimisation(path, track):
    traj_map = preprocess_heat_map(track, False)
    # traj_map = preprocess_heat_map(track, True)

    x = cvxpy.Variable((len(path), 2))

    cost = 0.0

    for i in range(1, len(path) - 1): # exclude start and end pts
        cost += cvxpy.sqrt(cvxpy.abs(cvxpy.square(x[i]) - cvxpy.square(x[i])))


    constraints = []
    constraints += [x[0] == path[0]]
    constraints += [x[-1] == path[-1]]
    constraints += [x[:,:] >= 0]
    constraints += [x[:, :] <= 0]
    constraints += [traj_map[int(x[:, 0].value), int(x[:, 1].value)] <= 0.5]

    problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    problem.solve()


    print(f"X values: {x.value}")


class LookUpCallback(Callback):
    def __init__(self, name, track, opts={}):
        Callback.__init__(self)
        self.construct(name, opts)
        track_map = track.get_heat_map()
        self.track_map = np.asarray(track_map, dtype=np.float32)

        # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1

        # Initialize the object
    def init(self):
        print('initializing lookup callback object')

        # Evaluate numerically
    def eval(self, arg):
        x, y = horzsplit(arg[0])
        ret = 0
        for i, j in zip(x, y):
            if self.track_map[int(i-1), int(j)]:
                return 1
        return 0 # no collissions

class ObjectiveCallback(Callback):
    def __init__(self, name, opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

        # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1

        # Initialize the object
    def init(self):
        print('initializing Objective Function callback object')

        # Evaluate numerically
    def eval(self, arg):
        x, y = horzsplit(arg[0])
        print(f"Arg: {arg}")
        cost = 0
        for i in range(len(x)-1):
            pt1 = [x[i], y[i]]
            pt2 = [x[i+1], y[i+1]]
            cost += f.get_distance(pt1, pt2) 

        return cost


def casadi_optimisation(path, track):
    # heat_map = preprocess_heat_map(track, False)
    track_map = track.get_heat_map()
    track_map = np.asarray(track_map, dtype=np.float32)

    path_length = len(path)

    x = MX.sym('x', path_length)
    y = MX.sym('y', path_length)

    # lut = LookUpCallback(name='lut', track=track)
    objective = ObjectiveCallback(name='objective')

    xgrid = np.linspace(0, 100, 100)
    ygrid = np.linspace(0, 100, 100)

    lut = interpolant('lut', 'bspline', [xgrid, ygrid], track_map.flatten())

    nlp = {\
        'x':vertcat(x, y),
        'f':sumsqr(v),
        # 'f':horzcat(objective(horzcat(x, y))), # can be a dedicated seperate function too.
        'g':vertcat(lut(horzcat(x, y).T).T,)
        }

    print(nlp)

    
    S = nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':3}})
    print(S)
    x0 = vertcat(path[0, :], path[1, :])

    r = S(x0=x0, lbg=0, ubg=100)

    x_opt = r['x']
    print('x_opt: ', x_opt)
    x_opt = r['y']
    print('y_opt: ', x_opt)
    x_opt = r['th']
    print('theta_opt: ', x_opt)

    return path

def add_obstacle_map_boundary(track_map):
    track_map = np.asarray(track_map, dtype=np.float32)

    x_len = track_map.shape[0]
    y_len = track_map.shape[1]
    for _ in range(2): # add a buffer of 2 in all directions
        for i in range(1, x_len-1):
            for j in range(1, y_len-1):
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


def casadi_optimisation_ruan(path, track):
    # heat_map = preprocess_heat_map(track, False)
    track_map = track.get_heat_map()
    # track_map = add_obstacle_map_boundary(track_map)
    # show_heat_map(track_map)
    track_map = np.asarray(track_map, dtype=np.float32)

    N = len(path)
    seed = np.asarray(path)
    # show_track_path(track, seed)
    A = 0
    B = 1
    delta_t = 1

    x = MX.sym('x', N)
    y = MX.sym('y', N)
    theta = MX.sym('theta', N)
    v = MX.sym('v', N)
    w = MX.sym('w', N)

    xgrid = np.linspace(0, 100, 100)
    ygrid = np.linspace(0, 100, 100)
    lut = interpolant('lut', 'bspline', [xgrid, ygrid], track_map.flatten())

    start = seed[0, :]
    end = seed[-1, :]

    nlp = {\
       'x':vertcat(x,y,theta,v,w),
    #    'f': A*(sumsqr(v) + sumsqr(w)) + B*(sumsqr(x-seed[:,0]) + sumsqr(y-seed[:,1])),
    #    'f': -sumsqr(v) + sumsqr(w) ,#+ B*(sumsqr(x-seed[:,0]) + sumsqr(y-seed[:,1])),
    'f': sum1(sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)), 
       'g':vertcat(\
                   x[1:] - (x[:-1] + delta_t*v[:-1]*cos(theta[:-1])),
                   y[1:] - (y[:-1] + delta_t*v[:-1]*sin(theta[:-1])),
                   theta[1:] - (theta[:-1] + delta_t*w[:-1]),
                   x[0]-start[0], y[0]-start[1],
                   x[N-1]-end[0], y[N-1]-end[1],
                   lut(horzcat(x,y).T).T * 1e8
                  )\
    }

    print(nlp)
    S = nlpsol('vert', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    # print(S)

    x0 = initSolution(seed)


    box = 60
    lbx_pre = [i-box for i in seed[:, 0]] + [i-box for i in seed[:, 1]] 
    ubx_pre = [i+box for i in seed[:, 0]] + [i+box for i in seed[:, 1]]

    lbg = [0] * (N-1)*3 + [0]*4 + [0]*N
    ubg = ([0]*(N-1)*3 + [0]*4 + [0]*N)
    # lbx = [0]*N + [0]*N + [-np.inf]*N + [0]*N + [-1]*N
    # ubx = [100]*N + [100]*N + [np.inf]*N + [5]*N + [1]*N

    lbx = lbx_pre + [-np.inf]*N + [0]*N + [-1]*N
    ubx = ubx_pre + [np.inf]*N + [5]*N + [1]*N
    r = S(x0=x0, lbg=lbg, ubg=ubg, ubx=ubx, lbx=lbx)
    x_opt = r['x']
    # print(S.stats())
    print(f"X_opt: {x_opt}")

    x_new = np.array(x_opt[0:N])
    y_new = np.array(x_opt[N:2*N])
    th_new = np.array(x_opt[2*N:N*3])
    v_new = np.array(x_opt[3*N:N*4])
    w_new = np.array(x_opt[4*N:N*5])

    # print(f"LUT res")
    # print(lut(horzcat(x_new, y_new).T))

    path = np.concatenate([x_new, y_new], axis=1)
    # print(path)

    return path

def casadi_optimisation_track_maze():
    track = TrackData()
    track.simple_maze()
    track_map = track.get_heat_map()
    track_map = np.asarray(track_map, dtype=np.float32)

    show_heat_map(track_map)

    path = get_practice_path()

    N = len(path)
    seed = np.asarray(path)
    # show_track_path(track, seed)
    A = 0
    B = 1
    delta_t = 1

    x = MX.sym('x', N)
    y = MX.sym('y', N)
    # theta = MX.sym('theta', N)
    # v = MX.sym('v', N)
    # w = MX.sym('w', N)

    xgrid = np.linspace(0, 100, 100)
    ygrid = np.linspace(0, 100, 100)
    lut = interpolant('lut', 'bspline', [xgrid, ygrid], track_map.flatten())

    start = seed[0, :]
    end = seed[-1, :]

    nlp = {\
       'x':vertcat(x,y),
    #    'f': A*(sumsqr(v) + sumsqr(w)) + B*(sumsqr(x-seed[:,0]) + sumsqr(y-seed[:,1])),
    #    'f': -sumsqr(v) + sumsqr(w) ,#+ B*(sumsqr(x-seed[:,0]) + sumsqr(y-seed[:,1])),
    'f': sum1(sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)), 
       'g':vertcat(\
                #    x[1:] - (x[:-1] + delta_t*v[:-1]*cos(theta[:-1])),
                #    y[1:] - (y[:-1] + delta_t*v[:-1]*sin(theta[:-1])),
                #    theta[1:] - (theta[:-1] + delta_t*w[:-1]),
                   x[0]-start[0], y[0]-start[1],
                   x[N-1]-end[0], y[N-1]-end[1],
                   lut(horzcat(x,y).T).T * 1e8
                  )\
    }

    print(nlp)
    S = nlpsol('vert', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    # print(S)

    # x0 = initSolution(seed)
    xs = seed[:, 0]
    ys = seed[:, 1]

    x0 = vertcat(xs, ys)

    box = 60
    lbx_pre = [i-box for i in seed[:, 0]] + [i-box for i in seed[:, 1]] 
    ubx_pre = [i+box for i in seed[:, 0]] + [i+box for i in seed[:, 1]]

    # lbg = [0] * (N-1)*3 + [0]*4 + [0]*N
    # ubg = ([0]*(N-1)*3 + [0]*4 + [0]*N)
    lbg =  [0]*4 + [0]*N
    ubg = [0]*4 + [0]*N
    # lbx = [0]*N + [0]*N + [-np.inf]*N + [0]*N + [-1]*N
    # ubx = [100]*N + [100]*N + [np.inf]*N + [5]*N + [1]*N

    lbx = lbx_pre # + [-np.inf]*N + [0]*N + [-1]*N
    ubx = ubx_pre #+ [np.inf]*N + [5]*N + [1]*N
    r = S(x0=x0, lbg=lbg, ubg=ubg, ubx=ubx, lbx=lbx)
    x_opt = r['x']
    # print(S.stats())
    print(f"X_opt: {x_opt}")

    x_new = np.array(x_opt[0:N])
    y_new = np.array(x_opt[N:2*N])
    # th_new = np.array(x_opt[2*N:N*3])
    # v_new = np.array(x_opt[3*N:N*4])
    # w_new = np.array(x_opt[4*N:N*5])

    # print(f"LUT res")
    # print(lut(horzcat(x_new, y_new).T))

    path = np.concatenate([x_new, y_new], axis=1)
    # print(path)

    show_path(track, convert_list_to_path(path))
    return path

def casadi_optimisation_ruan_simple(path, track):
    # heat_map = preprocess_heat_map(track, False)
    track_map = track.get_heat_map()
    track_map = np.asarray(track_map, dtype=np.float32)

    N = len(path)
    seed = np.asarray(path)
    # show_track_path(track, seed)
    A = 0
    B = 1
    C = 0.001
    D = 1e8
    delta_t = 1
    # print(seed)

    x = MX.sym('x', N)
    y = MX.sym('y', N)
    # theta = MX.sym('theta', N)
    # v = MX.sym('v', N)
    # w = MX.sym('w', N)

    xgrid = np.linspace(0, 100, 100)
    ygrid = np.linspace(0, 100, 100)

    lut = interpolant('lut', 'bspline', [xgrid, ygrid], track_map.flatten())
    # objective = ObjectiveCallback('Object')


    ans = lut(horzcat(50, 50).T)
    print(f"Answer: {ans}")

    start = seed[0, :]
    end = seed[-1, :]

    nlp = {\
       'x':vertcat(x,y),
    #    'f': B*(sumsqr(x-seed[:,0]) + sumsqr(y-seed[:,1])),
       'f':sq,
       'g':vertcat(\
                #    x[1:] - (x[:-1] + delta_t*v[:-1]*cos(theta[:-1])),
                #    y[1:] - (y[:-1] + delta_t*v[:-1]*sin(theta[:-1])),
                #    theta[1:] - (theta[:-1] + delta_t*w[:-1]),
                   x[0]-start[0], y[0]-start[1],
                   x[N-1]-end[0], y[N-1]-end[1],
                   lut(horzcat(x,y).T).T 
                  )\
    }

    print(nlp)
    S = nlpsol('vert', 'ipopt', nlp, {'ipopt':{'print_level':3}})
    # print(S)

    xs = seed[:, 0]
    ys = seed[:, 1]

    x0 = vertcat(xs, ys)

    box = 15
    lbx_pre = [i-box for i in xs] + [i-box for i in ys] 
    ubx_pre = [i+box for i in xs] + [i+box for i in ys]

    lbg =  [0]*4 + [0]*N
    ubg = [0]*4 + [0]*N
    # lbx = [0]*N + [0]*N + [-np.inf]*N + [0]*N + [-1]*N
    # ubx = [100]*N + [100]*N + [np.inf]*N + [5]*N + [1]*N

    lbx = lbx_pre 
    ubx = ubx_pre 
    r = S(x0=x0, lbg=lbg, ubg=ubg, ubx=ubx, lbx=lbx)
    x_opt = r['x']
    # print(S.stats())
    print(f"X_opt: {x_opt}")

    x_new = np.array(x_opt[0:N])
    y_new = np.array(x_opt[N:2*N])
    # th_new = np.array(x_opt[2*N:N*3])
    # v_new = np.array(x_opt[3*N:N*4])
    # w_new = np.array(x_opt[4*N:N*5])

    print(f"LUT res")
    print(lut(horzcat(x_new, y_new).T))

    path = np.concatenate([x_new, y_new], axis=1)
    print(path)

    return path

def casadi_optimisation_startend(path, track):
    # heat_map = preprocess_heat_map(track, False)
    track_map = track.get_heat_map()
    # track_map = add_obstacle_map_boundary(track_map)
    # show_heat_map(track_map)
    track_map = np.asarray(track_map, dtype=np.float32)

    N = len(path)
    seed = np.asarray(path)
    # show_track_path(track, seed)

    delta_t = 1

    x = MX.sym('x', N)
    y = MX.sym('y', N)
    theta = MX.sym('theta', N)
    v = MX.sym('v', N)
    w = MX.sym('w', N)

    xgrid = np.linspace(0, 100, 100)
    ygrid = np.linspace(0, 100, 100)
    lut = interpolant('lut', 'bspline', [xgrid, ygrid], track_map.flatten())

    start = seed[0, :]
    end = seed[-1, :]

    nlp = {\
       'x':vertcat(x,y,theta,v,w),
    #    'f': A*(sumsqr(v) + sumsqr(w)) + B*(sumsqr(x-seed[:,0]) + sumsqr(y-seed[:,1])),
    # 'f': sum1(sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)), 
    'f': 5*(sumsqr(v) + sumsqr(w)) + (sumsqr(x-seed[:,0]) + sumsqr(y-seed[:,1])),
       'g':vertcat(\
                   x[1:] - (x[:-1] + delta_t*v[:-1]*cos(theta[:-1])),
                   y[1:] - (y[:-1] + delta_t*v[:-1]*sin(theta[:-1])),
                   theta[1:] - (theta[:-1] + delta_t*w[:-1]),
                   x[0]-start[0], y[0]-start[1],
                   x[N-1]-end[0], y[N-1]-end[1],
                   lut(horzcat(x,y).T).T * 1e8
                  )\
    }

    print(nlp)
    S = nlpsol('vert', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    # print(S)

    x0 = initSolution(seed)

    box = 60
    lbx_pre = [i-box for i in seed[:, 0]] + [i-box for i in seed[:, 1]] 
    ubx_pre = [i+box for i in seed[:, 0]] + [i+box for i in seed[:, 1]]

    lbg = [0] * (N-1)*3 + [0]*4 + [0]*N
    ubg = ([0]*(N-1)*3 + [0]*4 + [0]*N)

    # lbx = [0]*N + [0]*N + [-np.inf]*N + [0]*N + [-1]*N
    # ubx = [100]*N + [100]*N + [np.inf]*N + [5]*N + [1]*N

    lbx = lbx_pre + [-np.inf]*N + [0]*N + [-1]*N
    ubx = ubx_pre + [np.inf]*N + [5]*N + [1]*N
    r = S(x0=x0, lbg=lbg, ubg=ubg, ubx=ubx, lbx=lbx)
    x_opt = r['x']
    # print(S.stats())
    print(f"X_opt: {x_opt}")

    x_new = np.array(x_opt[0:N])
    y_new = np.array(x_opt[N:2*N])
    th_new = np.array(x_opt[2*N:N*3])
    v_new = np.array(x_opt[3*N:N*4])
    w_new = np.array(x_opt[4*N:N*5])

    path = np.concatenate([x_new, y_new], axis=1)
    # print(path)

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

def test_start_end_opti():
    track = load_map("myTrack3")

    path_file = "DataRecords/maze_path_list.npy"
    try:
        path = np.load(path_file)
    except:
        path = A_StarFinderMod(track, 1)
        np.save(path_file, path)

        show_track_path(track, path)


    path = reduce_path_diag(path)
    path = expand_path(path)
    # show_track_path(track, path)

    path = casadi_optimisation_startend(path, track)

    show_track_path(track, path)


def test_cvxpy_opti(load_opti_path=True, load_path=True):
    track = load_map("myTrack2")
    # show_point_heat_map(track)

    # path_file = "DataRecords/path_list.npy"
    # path = np.load(path_file)

    # path_finder = A_StarPathFinderTrack(track)
    path = A_StarTrackWrapper(track, 1)
    # path = path_finder.run_search(2)

    # show_track_path(track, path)


    path = reduce_path_diag(path)
    # show_track_path(track, path)

    # track_map = preprocess_heat_map(track)
    # get_heat_map_eqn(track_map)

    # path = cvxpy_optimisation(path, track)
    # path = casadi_optimisation(path, track)
    path = casadi_optimisation_ruan(path, track)
    # path = casadi_optimisation_ruan_simple(path, track)
    # path = convert_list_to_path(path)
    show_track_path(track, path)
    # path_obj = add_velocity(path)


def test_heat_map():
    track = load_map()
    traj_map = preprocess_heat_map(track, False)

    get_heat_map_eqn(traj_map)




if __name__ == "__main__":
    # run_traj_opti()
    # test_cvxpy_opti(False, True)
    # casadi_optimisation_track_maze()
    test_start_end_opti()
    # test_heat_map()
    # plot_interp()
