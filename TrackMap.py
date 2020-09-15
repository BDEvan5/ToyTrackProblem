import numpy as np 
from matplotlib import pyplot as plt
import yaml
import csv

import LibFunctions as lib


class TrackMap:
    def __init__(self, csv_map="RaceTrack1000_abscissa.csv"):
        self.csv_name = csv_map

        self.track_pts = None
        self.nvecs = None
        self.ws = None
        self.N = None


    def load_map(self):
        track = []
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded")

        self.track_pts = track[:, 0:2]
        self.nvecs = track[:, 2: 4]
        self.ws = track[:, 4:6]

        

