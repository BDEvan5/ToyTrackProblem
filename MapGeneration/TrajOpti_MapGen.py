import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt
import math
import csv

import LibFunctions as lib 


def load_track(filename='Maps/RaceTrack1000_abscissa.csv', show=True):
    track = []
    with open(filename, 'r') as csvfile:
        csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
    
        for lines in csvFile:  
            track.append(lines)

    track = np.array(track)

    print(f"Track Loaded")

    return track


def interp_track(track, N):
    seg_lengths = np.sqrt(np.sum(np.power(np.diff(track[:, :2], axis=0), 2), axis=1))
    dists_cum = np.cumsum(seg_lengths)
    dists_cum = np.insert(dists_cum, 0, 0.0)
    length = sum(seg_lengths)

    ds = length / N

    no_points_interp = math.ceil(dists_cum[-1] / (ds)) + 1
    dists_interp = np.linspace(0.0, dists_cum[-1], no_points_interp)

    interp_track = np.zeros((no_points_interp, 2))
    interp_track[:, 0] = np.interp(dists_interp, dists_cum, track[:, 0])
    interp_track[:, 1] = np.interp(dists_interp, dists_cum, track[:, 1])

    return interp_track

def get_nvec(x0, x2):
    th = lib.get_bearing(x0, x2)
    new_th = lib.add_angles_complex(th, np.pi/2)
    nvec = lib.theta_to_xy(new_th)

    return nvec

def create_nvecs(track):
    N = len(track)

    nvecs = []
    nvecs.append(get_nvec(track[0, 0:2], track[1, 0:2]))
    for i in range(1, N-1):
        nvec = get_nvec(track[i-1, 0:2], track[i+1, 0:2])
        nvecs.append(nvec)
    nvecs.append(get_nvec(track[-2, 0:2], track[-1, 0:2]))

    return_track = np.concatenate([track, nvecs], axis=-1)

    return return_track

def plot_race_line(track, nset=None, wait=False):
    c_line = track[:, 0:2]
    l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
    r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

    plt.figure(1)
    plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
    plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1)
    plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1)

    if nset is not None:
        deviation = np.array([track[:, 2] * nset[:, 0], track[:, 3] * nset[:, 0]]).T
        r_line = track[:, 0:2] + deviation
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=3)

    plt.pause(0.0001)
    if wait:
        plt.show()

def set_widths(track, width=5):
    N = len(track)

    ls, rs = [width], [width]
    for i in range(N-2):
        # dw = (lib.get_bearing(track[i, 0:2], track[i+1, 0:2]) - lib.get_bearing(track[i+1, 0:2], track[i+2, 0:2])) / (np.pi/2)
        # wl = width + dw
        # wr = width - dw

        # ls.append(wl)
        # rs.append(wr)

        ls.append(width)
        rs.append(width)

    ls.append(width)
    rs.append(width)

    ls = np.array(ls)
    rs = np.array(rs)

    new_track = np.concatenate([track, ls[:, None], rs[:, None]], axis=-1)

    return new_track

def save_map(track, name):
    filename = 'Maps/' + name

    with open(filename, 'w') as csvfile:
        csvFile = csv.writer(csvfile)
        csvFile.writerows(track)



def run_map_gen():
    N = 100
    path = load_track()
    path = interp_track(path, N)

    track = create_nvecs(path)
    track = set_widths(track, 5)

    plot_race_line(track, wait=False)

    save_map(track, "TrackMap1000.csv")

    return track



def get_min_curve_pts():
    track = run_map_gen()
    n_set = MinCurvatureTrajectory(track)

    deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
    r_line = track[:, 0:2] + deviation

    return r_line



if __name__ == "__main__":
    # run_map_gen()
    get_min_curve_pts()
    # MinCurvatureTrajectory()


