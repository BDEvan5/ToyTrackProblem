import numpy as np 
import matplotlib.pyplot as plt 
import csv

import LibFunctions as lib 
from RaceTrackMaps import RaceMap
from PathFinder import PathFinder


def create_track():
    track_name = 'RaceTrack1000'
    env_map = RaceMap('RaceTrack1000')
    fcn = env_map.obs_free_hm._check_line
    path_finder = PathFinder(fcn, env_map.start, env_map.end)

    path = path_finder.run_search(5)
    # env_map.race_course.show_map(path=path, show=True)

    width = 5
    widths = np.ones_like(path) * width
    track = np.concatenate([path, widths], axis=-1)

    filename = 'Maps/' + track_name + '_abscissa.csv'

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['xi', 'yi', 'nl', 'nr'])
        csvwriter.writerows(track)


if __name__ == "__main__":
    create_track()
