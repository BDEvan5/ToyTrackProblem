import numpy as np 
from matplotlib import pyplot as plt
import yaml
import csv

import LibFunctions as lib
from TrajectoryPlanner import MinCurvatureTrajectory


class TrackMap:
    def __init__(self, csv_map="TrackMap1000"):
        self.name = csv_map

        self.track = None
        self.track_pts = None
        self.nvecs = None
        self.ws = None
        self.N = None

        self.start = None
        self.end = None

        self.obs_map = None
        self.scan_map = None
        self.obs_res = 0.1

        self.load_map()
        self.set_up_scan_map()

    def load_map(self):
        track = []
        filename = 'Maps/' + self.name + ".csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded")

        self.track = track
        self.N = len(track)
        self.track_pts = track[:, 0:2]
        self.nvecs = track[:, 2: 4]
        self.ws = track[:, 4:6]

        self.start = self.track_pts[0] - 0.1
        self.end = self.track_pts[-1]

        self.random_obs(0)

    def get_min_curve_path(self):
        path_name = 'Maps/' + self.name + "_path.npy"
        try:
            # raise Exception
            path = np.load(path_name)
            print(f"Path loaded from file: min curve")
        except:
            track = self.track
            n_set = MinCurvatureTrajectory(track, self.obs_map)
            deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
            path = track[:, 0:2] + deviation

            np.save(path_name, path)
            print(f"Path saved: min curve")

        return path

    def find_nearest_point(self, x):
        distances = []
        for i in range(self.N):
            d = lib.get_distance(self.track_pts[i], x)
            distances.append(d)
        nearest_idx = np.argmin(np.array(distances))

        return nearest_idx

    def _check_location(self, x):
        idx = self.find_nearest_point(x)
        dis = lib.get_distance(self.track_pts[idx], x)
        if dis > self.ws[idx, 0] * 1.5:
            return True
        return False

    def random_obs(self, n=10):
        resolution = 100
        self.obs_map = np.zeros((resolution, resolution))
        obs_size = [3, 4]
        rands = np.random.randint(1, self.N-1, n)
        obs_locs = []
        for i in range(n):
            # obs_locs.append(lib.get_rand_ints(40, 25))
            pt = self.track_pts[rands[i]][:, None]
            obs_locs.append(pt[:, 0])

        for obs in obs_locs:
            for i in range(0, obs_size[0]):
                for j in range(0, obs_size[1]):
                    x = min(int(round(i + obs[0]/ self.obs_res)), 99)
                    y = min(int(round(j + obs[1]/ self.obs_res)), 99)
                    self.obs_map[x, y] = 1

    def set_up_scan_map(self):
        try:
            # raise Exception
            self.scan_map = np.load("Maps/scan_map.npy")
        except:
            resolution = 100
            self.scan_map = np.zeros((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    ii = i*self.obs_res
                    jj = j*self.obs_res
                    if self._check_location([ii, jj]):
                        self.scan_map[i, j] = 1
            np.save("Maps/scan_map", self.scan_map)

            print("Scan map ready")
        # plt.imshow(self.scan_map.T)
        # plt.show()

    def get_show_map(self):
        ret_map  = np.clip(self.obs_map + self.scan_map, 0 , 1)
        return ret_map

    def check_scan_location(self, x_in):
        y = int(min(x_in[1] / self.obs_res, 99))
        x = int(min(x_in[0] / self.obs_res, 99))
        if self.scan_map[x, y]:
            return True
        if self.obs_map[x, y]:
            return True
        return False

    def reset_map(self):
        self.random_obs(10)
