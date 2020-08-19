import numpy as np
import LibFunctions as f
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class TrackMapData:
    def __init__(self, name, resolution=1, scaling_factor=10):
        # replacing the old trackdata
        self.fs = int(scaling_factor) # scaling factor from map size to display dize
        self.res = int(resolution) # how many map blocks to drawn blocks
        self.map_size = np.array([100, 100]) # how big the map is, 100, 100
        self.display_size = self.map_size * self.fs # tkinter pxls
        self.n_blocks = int(self.map_size[0]/self.res) # how many blocks per dimension

        #zeros to ones
        # self.track_map = np.ones((self.n_blocks, self.n_blocks), dtype=np.bool)
        self.track_map = np.zeros((self.n_blocks, self.n_blocks), dtype=np.bool)
        self.track_hm = None
        self.obs_map = np.zeros_like(self.track_map) # stores heatmap of obstacles
        
        self.start_x1 = None 
        self.start_x2 = None
        self.start_location = None
        self.start_line = []
        self.way_pts = []
        self.path_end_location = None
        self.path_start_location = None

        self.name = name


        # boundariy?? not needed
        self.obstacles = []

    def set_map_parameters(self, fs, res, display_size, n_blocks, map_size):
        self.fs = fs
        self.res = res 
        self.display_size = display_size
        self.n_blocks = n_blocks
        self.map_size = map_size

    def add_location(self, x_start, x_end):
        self.start = x_start
        self.end = x_end

    def add_random_obstacle(self):
        o = Obstacle([3, 3])
        self.obstacles.append(o)

    def reset_obstacles(self):
        # this is called to respawn the obstacles
        obs_map = np.zeros_like(self.track_map) # create new map
        n_obs = len(self.obstacles)
        rands = np.random.randint(1, self.n_blocks-1, (n_obs, 2))

        for i, obs in enumerate(self.obstacles):
            obs_map = obs.get_new_map(obs_map, rands[i]) # possibly move fcn to self

        self.obs_map = obs_map

    def random_start_end(self):
        rands = np.random.rand(4) * 100
        self.path_start_location = rands[0:2]
        self.path_end_location = rands[2:4]  

        while self.check_hm_collision(self.path_start_location):
            self.path_start_location = np.random.rand(2) * 100
        while self.check_hm_collision(self.path_end_location) or \
            f.get_distance(self.path_end_location, self.path_start_location) < 15:
            self.path_end_location = np.random.rand(2) * 100

        return self.path_start_location, self.path_end_location


    def set_start_line(self):
        if self.start_x1 == None or self.start_x2 == None:
            self.start_line.clear()
        else:
            for i in range(self.start_x1[0], self.start_x2[0]):
                self.start_line.append([i, self.start_x1[1]])
            y = self.start_x1[1] 
            x = int((self.start_x1[0] + self.start_x2[0])/2)
            self.start_location = [x, y]

    def check_collision(self, x, hidden_obs=False):
        i = int(x[0])
        j = int(x[1])
        if i >= 100 or i < 0: 
            return True
        if j >= 100 or j < 0:
            return True
        if self.track_map[i, j]:
            return True
        if hidden_obs and self.obs_map[i, j]:
            return True
        return False
    
    def check_hm_collision(self, x): # used for safe path finding
        if self.track_hm is None:
            self.track_hm = process_heat_map(self)
            print("Processed Heat Map")
        i = int(x[0])
        j = int(x[1])
        if i >= 95 or i < 1: 
            return True
        if j >= 95 or j < 1:
            return True
        if self.track_hm[i, j]:
            return True
        return False

    def check_hm_line_collision(self, x1, x2):
        n_pts = 15
        m = f.get_gradient(x1, x2)
        x_search = np.linspace(0, x2[0] - x1[0], n_pts)
        for i in range(n_pts):
            pt_add = [x_search[i], m * x_search[i]]
            pt = f.add_locations(pt_add, x1)
            if self.check_hm_collision(pt):
                return True
        return False
    



    def get_location_value(self, x):
        block_size = self.fs * self.res

        x_ret = int(np.floor(x[0] / block_size))
        y_ret = int(np.floor(x[1] / block_size))

        return [x_ret, y_ret]

    def check_line_collision(self, x1, x2, hidden_obs=False):
        n_pts = 15
        m = f.get_gradient(x1, x2)
        x_search = np.linspace(0, x2[0] - x1[0], n_pts)
        for i in range(n_pts):
            pt_add = [x_search[i], m * x_search[i]]
            pt = f.add_locations(pt_add, x1)
            if self.check_collision(pt, hidden_obs):
                return True
        return False

    def check_past_start(self, x1, x2):
        if max(x1[0], x2[0]) > self.start_x2[0] or min(x1[0], x2[0]) < self.start_x1[0]:
            return False # wrong x value
        y = self.start_x1[1] + self.res * self.fs * 0.5 # same y val - middle line
        if x1[1] > y and x2[1] < y:
            return True
        elif x1[1] > y and x2[1] < y:
            return True
        return False

    def check_done(self, x):
        if f.get_distance(x, self.path_end_location) < 10:
            return True
        return False

    def get_heat_map(self):
        return self.track_map


def process_heat_map(track, show=True):
    show = False
    track_map = track.get_heat_map()
    track_map = np.asarray(track_map, dtype=np.float32)
    new_map = np.zeros_like(track_map)

    for _ in range(3): # blocks up to 5 away will start to have a gradient
        new_map = np.zeros_like(track_map)
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

                centre = track_map[i, j]

                obs_sum = sum((centre, left, right, up, down, left_up, left_down, right_up, right_down))
                if obs_sum > 0:
                    new_map[i, j] = 1
                # new_map[i, j] = max(obs_sum / 16, track_map[i, j])
        track_map = new_map

    if show:
        # show_heat_map(track_map)
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



class Obstacle:
    def __init__(self, size=[0, 0]):
        self.size = size
        self.layer_size_x = size[0] // 2
        self.layer_size_y = size[1] // 2

    def get_new_map(self, obs_map, rand_xy):
        rand_x = rand_xy[0]
        rand_y = rand_xy[1]

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                x = i - self.layer_size_x + rand_x
                y = j - self.layer_size_y + rand_y

                try:
                    obs_map[x, y] = 1
                except:
                    print(f"Error setting obs at {x}:{y}")

        return obs_map



