import numpy as np 
from matplotlib import pyplot as plt
import yaml
import csv

import LibFunctions as lib


class TrackMap:
    def __init__(self, csv_map="TrackMap1000.csv"):
        self.name = csv_map

        self.track = None
        self.track_pts = None
        self.nvecs = None
        self.ws = None
        self.N = None

        self.start = None
        self.end = None

        self.load_map()

    def load_map(self):
        track = []
        filename = 'Maps/' + self.name
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

