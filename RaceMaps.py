import numpy as np 
from matplotlib import pyplot as plt
import yaml

import LibFunctions as lib

class GeneralMap:
    def __init__(self, race_map):
        self.race_map = race_map

        self.map_height = race_map.shape[0]
        self.map_width = race_map.shape[1]

    def _check_location(self, x):
        if x[0] < 1 or x[0] > self.map_width:
            return True
        if x[1] < 1 or x[1] > self.map_height:
            return True 

        if self.race_map[int(round(x[0])), int(round(x[1]))]:
            return True

        return False

    def _check_line(self, start, end):
        n_checks = 5
        dif = lib.sub_locations(end, start)
        diff = [dif[0] / (n_checks), dif[1] / n_checks]
        for i in range(5):
            search_val = lib.add_locations(start, diff, i + 1)
            if self._check_location(search_val):
                return True
        return False

    def show_map(self, path=None):
        fig = plt.figure(7)

        plt.imshow(self.race_map.T, origin='lower')
        # plt.plot(self.start[0], self.start[1], '*', markersize=20)
        # plt.plot(self.end[0], self.end[1], '*', markersize=20)

        if path is not None:
            xs, ys = [], []
            for pt in path:
                xs.append(pt[0])
                ys.append(pt[1])
            
            plt.plot(xs, ys)
            plt.plot(xs, ys, 'x', markersize=16)

        plt.show()
        # plt.pause(0.001)


class EnvironmentMap:
    def __init__(self, name='TestTrack1010'):
        self.name = name 
        self.yaml_name = 'DataRecords/' + self.name + '.yaml'

        self.obs_free_hm_name = 'DataRecords/' + self.name + '_heatmap_obs_free.npy'
        self.hm_name = 'DataRecords/' + self.name + '_heatmap.npy'

        self.race_course = None
        self.obs_hm = None
        self.obs_free_hm = None

    def load_maps(self):
        yaml_editor = MapYamlEditor(self.name)
        yaml_editor.load_yaml_file()
        yaml_editor.print_contents()

        map_name = 'DataRecords/' + yaml_editor.map_name + '.npy'
        map_array = np.load(map_name)
        self.race_course = GeneralMap(map_array.T)

        obs_free_hm = self.create_hm(self.obs_free_hm_name)
        self.obs_free_hm = GeneralMap(obs_free_hm)

        self.add_obs(yaml_editor.obs_locs, yaml_editor.obs_size)
        obs_hm = self.create_hm(self.hm_name)
        self.obs_hm = GeneralMap(obs_hm)


    def add_obs(self, obs_locs, obs_size):
        for obs in obs_locs:
            for i in range(obs_size[0]):
                for j in range(obs_size[1]):
                    x = i + obs[0]
                    y = j + obs[1]
                    self.race_course.race_map[x, y] = 2

    def create_hm(self, hm_name, n_units=2):
        try:
            # raise Exception
            return np.load(hm_name)
            # print(f"Heatmap loaded")
        except:
            hm = self._set_up_heat_map()
            np.save(hm_name, hm)
            print(f"Heatmap saved")
            return hm

    def _set_up_heat_map(self):
        print(f"Starting heatmap production")
        track_map = self.race_course.race_map
        for i in range(2): 
            new_map = np.zeros_like(track_map)
            print(f"Map run: {i}")
            for i in range(1, 98 - 2):
                for j in range(1, 98 - 2):
                    left = track_map[i-1, j]
                    right = track_map[i+1, j]
                    up = track_map[i, j+1]
                    down = track_map[i, j-1]

                    # logical directions, not according to actual map orientation
                    left_up = track_map[i-1, j+1] *3
                    left_down = track_map[i-1, j-1]*3
                    right_up = track_map[i+1, j+1]*3
                    right_down = track_map[i+1, j-1]*3

                    centre = track_map[i, j]

                    obs_sum = sum((centre, left, right, up, down, left_up, left_down, right_up, right_down))
                    if obs_sum > 0:
                        new_map[i, j] = 1

            track_map = new_map
        new_map[:, 98:] = np.ones_like(new_map[:, 98:])
        return new_map
 

class MapYamlEditor:
    def __init__(self, map_name='TestTrack1000'):
        self.dict = {}

        self.map_name = map_name
        self.start = None
        self.end = None
        self.obs_locs = []
        self.obs_size = None
        self.height = None
        self.width = None

        self.set_dict_values()
        self.create_dictionary()

    def set_dict_values(self):
        self.start = [53, 5]
        self.end = [90, 20]

        self.obs_locs = [[20, 20], [60, 60]]

        self.obs_size = [8, 14]
        self.height = 100
        self.width = 100

    def create_dictionary(self):
        self.dict['map_name'] = self.map_name
        self.dict['start'] = self.start
        self.dict['end'] = self.end

        obs_dict = {}
        for i in range(len(self.obs_locs)):
            obs_dict[i] = self.obs_locs[i]

        self.dict['obstacles'] = obs_dict
        self.dict['obs_size'] = self.obs_size
        self.dict['width'] = self.width
        self.dict['height'] = self.height

    def save_dict_yaml(self):
        yaml_name = 'DataRecords/' + self.map_name + '.yaml'
        with open(yaml_name, 'w') as yaml_file:
            yaml.dump(self.dict, yaml_file)

    def load_yaml_file(self):
        yaml_name = 'DataRecords/' + self.map_name + '.yaml'
        with open(yaml_name) as yaml_file:
            self.dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

    def print_contents(self):
        print(f"Dictionary of: {self.map_name}")
        print(self.dict)



class MapSetUp:
    def __init__(self):
        self.obs_locs = []

    def map_1000(self, add_obs=True):
        self.name = 'TestTrack1000'

        self.start = [10, 90]
        self.end = [90, 25]

        if add_obs:
            self.obs_locs = [[15, 50], [28, 25], [44, 28], [70, 74], [88, 40]]
        self.set_up_map()
        
    def map_1010(self):
        self.name = 'TestTrack1010'

        self.start = [53, 5]
        self.end = [90, 20]

        self.obs_locs = [[20, 20], [60, 60]]
        self.set_up_map()

    def map_1020(self):
        self.name = 'TestTrack1020'

        self.start = [70, 18]
        self.end = [75, 80]

        self.obs_locs = [[58, 20], [25, 36], [28, 56], [45, 30], [37, 68], [60, 68]]
        self.set_up_map()

    def map_1030_strip(self):
        self.name = 'TestTrack1030'

        self.start = [10, 45]
        self.end = [90, 45]

        self.obs_locs = [[30, 40], [50, 35], [70, 45]]
        self.set_up_map()
        
    def set_up_map(self):
        self.hm_name = 'DataRecords/' + self.name + '_heatmap.npy'
        self.path_name = "DataRecords/" + self.name + "_path.npy" # move to setup call


        self.create_race_map()
        self._place_obs()
        self.create_hm()


class TestMap(MapSetUp):
    def __init__(self):
        MapSetUp.__init__(self)
        self.name = None
        self.map_dim = 100

        self.race_map = None
        self.heat_map = None
        self.start = None
        self.end = None

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]
        self.hm_name = None

    def create_race_map(self):
        race_map_name = 'DataRecords/' + self.name + '.npy'
        array = np.load(race_map_name)
        new_map = np.zeros((self.map_dim, self.map_dim))
        block_range = self.map_dim / array.shape[0]
        for i in range(self.map_dim):
            for j in range(self.map_dim):
                new_map[i, j] = array[int(i/block_range), int(j/block_range)]

        self.race_map = new_map.T

    def create_hm(self):
        try:
            # raise Exception
            self.heat_map = np.load(self.hm_name)
            # print(f"Heatmap loaded")
        except:
            self._set_up_heat_map()
            np.save(self.hm_name, self.heat_map)
            print(f"Heatmap saved")

        # self.show_hm()

    def _place_obs(self):
        obs_locs = self.obs_locs
        obs_size = [6, 10]
        for obs in obs_locs:
            for i in range(obs_size[0]):
                for j in range(obs_size[1]):
                    x = i + obs[0]
                    y = j + obs[1]
                    # if not s
                    self.race_map[x, y] = 2

    def _check_location(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if self.y_bound[0] > x[1] or x[1] > self.y_bound[1]:
            return True 

        if self.race_map[int(round(x[0])), int(round(x[1]))]:
            return True

        return False

    def _check_line(self, start, end):
        n_checks = 5
        dif = lib.sub_locations(end, start)
        diff = [dif[0] / (n_checks), dif[1] / n_checks]
        for i in range(5):
            search_val = lib.add_locations(start, diff, i + 1)
            if self._check_location(search_val):
                return True
        return False

    def show_map(self, path=None):
        fig = plt.figure(7)

        plt.imshow(self.race_map.T, origin='lower')
        plt.plot(self.start[0], self.start[1], '*', markersize=20)
        plt.plot(self.end[0], self.end[1], '*', markersize=20)

        if path is not None:
            xs, ys = [], []
            for pt in path:
                xs.append(pt[0])
                ys.append(pt[1])
            
            plt.plot(xs, ys)
            plt.plot(xs, ys, 'x', markersize=16)

        plt.show()
        # plt.pause(0.001)

    def show_hm(self, path=None):
        plt.imshow(self.heat_map.T, origin='lower')
        plt.plot(self.start[0], self.start[1], '*', markersize=20)
        plt.plot(self.end[0], self.end[1], '*', markersize=20)


        if path is not None:
            xs, ys = [], []
            for pt in path:
                xs.append(pt[0])
                ys.append(pt[1])
            
            plt.plot(xs, ys)

        plt.show()
        # plt.pause(0.001)

    def _set_up_heat_map(self):
        print(f"Starting heatmap production")
        track_map = self.race_map
        for i in range(2): 
            new_map = np.zeros_like(track_map)
            print(f"Map run: {i}")
            for i in range(1, self.map_dim - 2):
                for j in range(1, self.map_dim - 2):
                    left = track_map[i-1, j]
                    right = track_map[i+1, j]
                    up = track_map[i, j+1]
                    down = track_map[i, j-1]

                    # logical directions, not according to actual map orientation
                    left_up = track_map[i-1, j+1] *3
                    left_down = track_map[i-1, j-1]*3
                    right_up = track_map[i+1, j+1]*3
                    right_down = track_map[i+1, j-1]*3

                    centre = track_map[i, j]

                    obs_sum = sum((centre, left, right, up, down, left_up, left_down, right_up, right_down))
                    if obs_sum > 0:
                        new_map[i, j] = 1

            track_map = new_map
        new_map[:, 98:] = np.ones_like(new_map[:, 98:])
        self.heat_map = new_map
        

        # fig = plt.figure(1)
        # plt.imshow(self.heat_map.T, origin='lower')
        # fig = plt.figure(2)
        # plt.imshow(self.race_map.T, origin='lower')
        # plt.show()

    def _path_finder_collision(self, x):
        if self.x_bound[0] >= x[0] or x[0] > self.x_bound[1]:
            return True
        if self.y_bound[0] >= x[1] or x[1] > self.y_bound[1]:
            return True 

        if self.heat_map[int(x[0]), int(x[1])]:
            return True

        return False

    def _check_line_path(self, start, end):
        n_checks = 5
        dif = lib.sub_locations(end, start)
        diff = [dif[0] / (n_checks), dif[1] / n_checks]
        for i in range(5):
            search_val = lib.add_locations(start, diff, i + 1)
            if self._path_finder_collision(search_val):
                return True
        return False
    
    def generate_random_start(self):
        self.start = lib.get_rands()
        while self.race_map._check_location(self.start):
            self.start = lib.get_rands()

        self.end = lib.get_rands()
        while self.race_map._check_location(self.end) or \
            lib.get_distance(self.start, self.end) < 30:
            self.end = lib.get_rands()
        self.end = lib.get_rands(80, 10)


#Test and setup fcns
def create_yaml_file():
    yaml_maker = MapYamlEditor('TestTrack1010')
    yaml_maker.save_dict_yaml()
    yaml_maker.load_yaml_file()
    yaml_maker.print_contents()

def test_environment():
    my_environ = EnvironmentMap('TestTrack1000')
    my_environ.load_maps()
    my_environ.race_course.show_map()
    my_environ.obs_free_hm.show_map()
    my_environ.obs_hm.show_map()


if __name__ == "__main__":
    # create_yaml_file()    
    test_environment()

    


