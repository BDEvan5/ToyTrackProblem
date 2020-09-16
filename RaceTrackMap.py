import numpy as np 
from matplotlib import pyplot as plt
import yaml
import csv

import LibFunctions as lib


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

        self.start = self.track_pts[0]
        self.end = self.track_pts[-1]

        self.random_obs()

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
        if dis > self.ws[idx, 0]:
            return True
        return False

    def random_obs(self, n=10):
        self.obs_map = np.zeros((100, 100))
        obs_size = [2, 3]
        rands = np.random.randint(1, self.N-1, n)
        obs_locs = []
        for i in range(n):
            # obs_locs.append(lib.get_rand_ints(40, 25))
            pt = self.track_pts[rands[i]][:, None]
            obs_locs.append(pt[:, 0])

        for obs in obs_locs:
            for i in range(obs_size[0]):
                for j in range(obs_size[1]):
                    x = int(round(i + obs[0]))
                    y = int(round(j + obs[1]))
                    self.obs_map[x, y] = 1

    def set_up_scan_map(self):
        try:
            self.scan_map = np.load("Maps/scan_map.npy")
        except:
            resolution = 100
            self.scan_map = np.zeros((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    if self._check_location([i, j]):
                        self.scan_map[i, j] = 1
            np.save("Maps/scan_map", self.scan_map)

            print("Scan map ready")

    def check_scan_location(self, x):
        if self.scan_map[int(x[0]), int(x[1])]:
            return True
        if self.obs_map[int(x[0]), int(x[1])]:
            return True
        return False

    def reset_map(self):
        self.random_obs(10)
