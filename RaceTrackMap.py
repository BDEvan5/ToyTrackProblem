import numpy as np 
from matplotlib import pyplot as plt
import yaml
import csv

import LibFunctions as lib
from TrajectoryPlanner import MinCurvatureTrajectory
from PathFinder import PathFinderStarA


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

        self.load_map_csv()
        self.set_up_scan_map()
        lengths = [lib.get_distance(self.track_pts[i], self.track_pts[i+1]) for i in range(self.N-1)]
        lengths.insert(0, 0)
        self.cum_lengs = np.cumsum(lengths)

        self.wpts = None # used for the target
        self.target = None

    def load_map_csv(self):
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
        distances = [lib.get_distance(x, self.track_pts[i]) for i in range(self.N)]

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
        if x_in[0] < 0 or x_in[1] < 0:
            return True

        y = int(max(min(x_in[1] / self.obs_res, 99), 0))
        x = int(max(min(x_in[0] / self.obs_res, 99), 0))
        if self.scan_map[x, y]:
            return True
        if self.obs_map[x, y]:
            return True
        return False

    def reset_map(self):
        self.random_obs(10)

    def get_s_progress(self, x):
        idx = self.find_nearest_point(x)

        if idx == 0:
            return lib.get_distance(x, self.track_pts[0])

        if idx == self.N-1:
            s = self.cum_lengs[-2] + lib.get_distance(x, self.track_pts[-2])
            return s

        p_d = lib.get_distance(x, self.track_pts[idx-1])
        n_d = lib.get_distance(x, self.track_pts[idx+1])

        if p_d < n_d:
            s = self.cum_lengs[idx-1] + p_d
        else:
            s = self.cum_lengs[idx] + lib.get_distance(self.track_pts[idx], x)


        return s

    def set_wpts(self, wpts):
        self.wpts = wpts

    def find_target(self, obs):
        distances = [lib.get_distance(obs[0:2], self.wpts[i]) for i in range(len(self.wpts))]
        ind = np.argmin(distances)
        N = len(self.wpts)

        look_ahead = 3
        pind = ind + look_ahead
        if pind >= N-look_ahead:
            pind = 1

        # front_dis = lib.get_distance(self.wpts[ind], self.wpts[ind+1])
        # back_dis = lib.get_distance(self.wpts[ind], self.wpts[ind-1])


        # if front_dis < back_dis * 0.5:
        #     pind = ind + 1
        # else:
        #     pind = ind

        target = self.wpts[pind]
        self.target = target

        return target, pind


class MapConverter:
    def __init__(self, map_name):
        self.name = map_name
        self.yaml_file = None
        self.scan_map = None

        self.resolution = None
        self.width = None
        self.height = None
        self.origin = None

    def load_map_pgm(self):
        map_name = 'maps/' + self.name 
        self.read_yaml_file(map_name + '.yaml')

        map_file_name = self.yaml_file['image']
        pgm_name = 'maps/' + map_file_name

        self.read_pgm_map(pgm_name)

    def read_pgm_map(self, pgm_name):
        with open(pgm_name) as f:
            lines = f.readlines()

        # This ignores commented lines
        for l in list(lines):
            if l[0] == '#':
                lines.remove(l)
        # here,it makes sure it is ASCII format (P2)
        assert lines[0].strip() == 'P2' 

        # Converts data to a list of integers
        data = []
        for line in lines[1:]:
            data.extend([int(c) for c in line.split()])

        data = (np.array(data[3:]),(data[1],data[0]),data[2])
        data = np.reshape(data[0],data[1])

        self.scan_map = data

    def read_yaml_file(self, file_name, print_out=False):
        with open(file_name) as file:
            documents = yaml.full_load(file)

            yaml_file = documents.items()
            if print_out:
                for item, doc in yaml_file:
                    print(item, ":", doc)

        self.yaml_file = dict(yaml_file)

        self.origin = self.yaml_file['origin']
        self.resolution = self.yaml_file['resolution']

    def show_map(self):
        plt.figure(1)
        plt.imshow(self.scan_map)

        plt.show()

    def find_a_path(self):
        #TODO: keep working on finding a path here.
        pass 
        # path_finder = PathFinderStarA(self.check_o, )




def test_map_converter():
    name = 'columbia'
    myConv = MapConverter(name)
    myConv.load_map_pgm()
    myConv.show_map()

if __name__ == "__main__":
    test_map_converter()
