import numpy as np 
from scipy import ndimage
from matplotlib import pyplot as plt
import yaml
import csv

import LibFunctions as lib
from TrajectoryPlanner import MinCurvatureTrajectory, ObsAvoidTraj, ShortestTraj


class MapBase:
    """
    This is a base map class for both the race maps and the forest maps.

    Methods
    ------
    Load a map from the relevant yaml, npy and csv files
        yaml files must have: resolution (from array to meters), start location
        CSV files are 6xN with xs, ys, nvecx, nvecy, wn, wp
    """
    def __init__(self, map_name):
        self.name = map_name

        self.scan_map = None
        self.obs_map = None

        self.track = None
        self.track_pts = None
        self.nvecs = None
        self.ws = None
        self.N = None

        self.start = None
        self.wpts = []

        self.height = None
        self.width = None
        self.resolution = None

        self.read_yaml_file()
        self.load_map_csv()

    def read_yaml_file(self, print_out=False):
        file_name = 'maps/' + self.name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

            yaml_file = documents.items()
            if print_out:
                for item, doc in yaml_file:
                    print(item, ":", doc)

        self.yaml_file = dict(yaml_file)

        self.resolution = self.yaml_file['resolution']
        self.start = self.yaml_file['start']

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

        self.scan_map = np.load(f'Maps/{self.name}.npy')

        self.width = self.scan_map.shape[1]
        self.height = self.scan_map.shape[0]

    def convert_position(self, pt):
        x = pt[0] / self.resolution
        y =  pt[1] / self.resolution

        return x, y

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.convert_position(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)
        
    def convert_int_position(self, pt):
        x = int(round(np.clip(pt[0] / self.resolution, 0, self.width-2)))
        y = int(round(np.clip(pt[1] / self.resolution, 0, self.height-2)))

        return x, y

    def check_scan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True
        x, y = self.convert_int_position(x_in)
        if x > self.width or y > self.height:
            return True

        if self.scan_map[y, x]:
            return True
        if self.obs_map[y, x]:
            return True
        return False

    def render_map(self, figure_n=4, wait=False):
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.width])
        plt.ylim([self.height, 0])

        track = self.track
        c_line = track[:, 0:2]
        l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
        r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

        cx, cy = self.convert_positions(c_line)
        plt.plot(cx, cy, linewidth=2)
        lx, ly = self.convert_positions(l_line)
        plt.plot(lx, ly, linewidth=1)
        rx, ry = self.convert_positions(r_line)
        plt.plot(rx, ry, linewidth=1)

        if self.wpts is not None:
            xs, ys = [], []
            for pt in self.wpts:
                x, y = self.convert_position(pt)
                # plt.plot(x, y, '+', markersize=14)
                xs.append(x)
                ys.append(y)
            plt.plot(xs, ys, '--', linewidth=2)

        if self.obs_map is None:
            plt.imshow(self.scan_map)
        else:
            plt.imshow(self.obs_map + self.scan_map)

        plt.gca().set_aspect('equal', 'datalim')

        plt.pause(0.0001)
        if wait:
            plt.show()


class SimMap(MapBase):
    def __init__(self, map_name):
        MapBase.__init__(self, map_name)

        self.obs_map = np.zeros_like(self.scan_map)
        self.end = self.start
        self.eld = None

        self.set_true_widths()
        track = self.track
        n_set = ShortestTraj(track, self.check_scan_location)

        track = self.track
        deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
        self.track[:, 0:2] += deviation
        self.set_true_widths()

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

        self.wpts = path

        return path

    def set_euclidian(self):
        m = np.ones_like(self.obs_map) - self.obs_map
        self.eld = ndimage.distance_transform_edt(m)

        # plt.figure(1)
        # plt.imshow(self.eld)
        # plt.show()

    def check_eld_location(self, x_in):
        if self.check_scan_location(x_in):
            return 0
        else:
            return self.eld[int(x_in[0]), int(x_in[1])]

    def get_optimal_path(self):

        # self.set_euclidian()
        # n_set = ObsAvoidTraj(track, self.check_eld_location)

        # self.render_map(figure_n=1, wait=False)
        
        self.render_map(figure_n=1, wait=False)
        
        track = self.track
        n_set = ObsAvoidTraj(self.track, self.check_scan_location)
        deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T

        self.wpts = track[:, 0:2] + deviation

        return self.wpts

    def get_reference_path(self):
        path_name = 'Maps/' + self.name + "_ref_path.npy"
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

        self.wpts = path

        return self.wpts

    def random_obs(self, n=10):
        self.obs_map = np.zeros_like(self.obs_map)

        obs_size = [self.width/600, self.height/600]
        # obs_size = [0.3, 0.3]
        # obs_size = [1, 1]
        x, y = self.convert_int_position(obs_size)
        obs_size = [x, y]
    
        rands = np.random.randint(1, self.N-1, n)
        obs_locs = []
        for i in range(n):
            pt = self.track_pts[rands[i]][:, None]
            obs_locs.append(pt[:, 0])

        for obs in obs_locs:
            for i in range(0, obs_size[0]):
                for j in range(0, obs_size[1]):
                    x, y = self.convert_int_position([obs[0], obs[1]])
                    self.obs_map[y+j, x+i] = 1

        return obs_locs

    def reset_map(self):
        o =  self.random_obs(10)
        return o

    def set_true_widths(self):
        nvecs = self.track[:, 2:4]
        tx = self.track[:, 0]
        ty = self.track[:, 1]

        stp_sze = 0.1
        sf = 0.95 # safety factor
        nws, pws = [], []
        for i in range(self.N):
            pt = [tx[i], ty[i]]
            nvec = nvecs[i]

            j = stp_sze
            s_pt = lib.add_locations(pt, nvec, j)
            while not self.check_scan_location(s_pt):
                j += stp_sze
                s_pt = lib.add_locations(pt, nvec, j)
            pws.append(j*sf)

            j = stp_sze
            s_pt = lib.sub_locations(pt, nvec, j)
            while not self.check_scan_location(s_pt):
                j += stp_sze
                s_pt = lib.sub_locations(pt, nvec, j)
            nws.append(j*sf)

        nws, pws = np.array(nws), np.array(pws)

        self.track[:, 4] = nws
        self.track[:, 5] = pws

        # new_track = np.concatenate([self.track[0:4], nws[:, None], pws[:, None]], axis=-1)

        # self.track = new_track


class ForestMap(MapBase):
    def __init__(self, map_name="forest"):
        MapBase.__init__(self, map_name)

        self.obs_map = np.zeros_like(self.scan_map)
        self.end = [3, 23]

    def get_optimal_path(self):
        track = self.track
        n_set = ObsAvoidTraj(track, self.check_scan_location)
        deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
        self.wpts = track[:, 0:2] + deviation

        return self.wpts

    def get_reference_path(self):
        self.wpts = self.track_pts

        return self.wpts

    def get_obs_free_path(self):
        pass
        # set up the optimisation to get this
        
    def random_obs(self, n=10):
        self.obs_map = np.zeros_like(self.obs_map)

        obs_size = [1.5, 1]
        xlim = (6 - obs_size[0]) / 2

        x, y = self.convert_int_position(obs_size)
        obs_size = [x, y]

        tys = np.linspace(4, 20, n)
        txs = np.random.normal(xlim, 1, size=n)
        txs = np.clip(txs, 0, 4)
        obs_locs = np.array([txs, tys]).T

        for obs in obs_locs:
            for i in range(0, obs_size[0]):
                for j in range(0, obs_size[1]):
                    x, y = self.convert_int_position([obs[0], obs[1]])
                    x = np.clip(x+i, 0, self.width-1)
                    y = np.clip(y+j, 0, self.height-1)
                    self.obs_map[y, x] = 1

        return obs_locs

    def reset_map(self):
        o = self.random_obs(6)
        return o

    def render_map(self, figure_n=1, wait=False):
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.width])
        plt.ylim([self.height, 0])

        if self.wpts is not None:
            xs, ys = [], []
            for pt in self.wpts:
                x, y = self.convert_position(pt)
                # plt.plot(x, y, '+', markersize=14)
                xs.append(x)
                ys.append(y)
            plt.plot(xs, ys, '--', color='g', linewidth=2)

        if self.obs_map is None:
            plt.imshow(self.scan_map)
        else:
            plt.imshow(self.obs_map + self.scan_map)

        # plt.gca().set_aspect('equal', 'datalim')
        x, y = self.convert_position(self.end)
        plt.plot(x, y, '*', markersize=14)

        plt.pause(0.0001)
        if wait:
            plt.show()



def test_sim_map_obs():
    name = 'race_track'
    env_map = SimMap(name)
    env_map.reset_map()

    wpts = env_map.get_optimal_path()
    env_map.render_map(wait=True)



if __name__ == "__main__":
    # test_map_converter()

    # forest_gen()
    test_sim_map_obs()