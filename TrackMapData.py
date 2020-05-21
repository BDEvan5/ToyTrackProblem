import numpy as np
import LibFunctions as f



class TrackMapData:
    def __init__(self, track_map):
        # replacing the old trackdata
        self.track_map = track_map
        self.obs_map = np.zeros_like(self.track_map) # stores heatmap of obstacles
        
        self.start_x1 = None 
        self.start_x2 = None
        self.start_line = []

        self.fs = None # scaling factor from map size to display dize
        self.res = None # how many map blocks to drawn blocks
        self.display_size = None # tkinter pxls
        self.n_blocks = None # how many blocks per dimension
        self.map_size = None # how big the map is, 100, 100

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

    def set_start_line(self):
        if self.start_x1 == None or self.start_x2 == None:
            self.start_line.clear()
        else:
            for i in range(self.start_x1[0], self.start_x2[0]):
                self.start_line.append([i, self.start_x1[1]])


    def check_collision(self, x, hidden_obs=False):
        i = x[0]
        j = x[1]
        if self.track_map[i, j]:
            return True
        if hidden_obs and self.obs_map[i, j]:
            return True
        return False

    def check_line_collision(self, x1, x2, hidden_obs=False):
        n_pts = 15
        m = f.get_gradient(x1, x2)
        x_search = np.linspace(x1[0], x1[0], n_pts)
        for i in range(n_pts):
            pt_add = [x_search[i], * x_search[i]]
            pt = f.add_locations(pt_add, x1)
            if self.check_line_collision(pt, hidden_obs):
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



