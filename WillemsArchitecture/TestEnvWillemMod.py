import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt

from PathFinder import PathFinder, modify_path


class CarModelDQN:
    def __init__(self, n_ranges=10):
        self.car_x = [0, 0]
        self.theta = 0
        self.velocity = 0
        self.steering = 0

        self.max_velocity = 20
        self.dth_action = 0.25 # amount of rad to swerve with each action 

        self.n_ranges = n_ranges
        self.ranges = np.zeros(self.n_ranges)
        self.range_angles = np.zeros(self.n_ranges)
        dth = np.pi/(self.n_ranges-1)
        for i in range(self.n_ranges):
            self.range_angles[i] = i * dth - np.pi/2

    def _x_step(self, action):
        dt = 1

        theta = self.theta + action[0]
        x_i = np.sin(theta)* self.velocity * dt + self.car_x[0]
        y_i = np.cos(theta)* self.velocity * dt + self.car_x[1]
        
        return [x_i, y_i]

    def update_state(self, action):
        a, d_dot = self.get_integrals(action) 

        dt = 1

        self.theta = self.theta + action[0] 
        x_i = np.sin(self.theta)* self.velocity * dt + self.car_x[0]
        y_i = np.cos(self.theta)* self.velocity * dt + self.car_x[1]
        
        self.car_x = [x_i, y_i]

        self.velocity = self.velocity + a * dt
           
    def get_integrals(self, action):
        d_ref = action[0] 
        v_ref = (action[1])* self.max_velocity 

        k_v = 1 # proportional control
        k_d = 1

        e_v = v_ref - self.velocity
        a = e_v * k_v

        e_d = d_ref - self.steering
        d_dot = k_d * e_d

        return a, d_dot


class MapSetUp:
    def __init__(self):
        self.obs_locs = []

    def map_1000(self, add_obs=True):
        self.name = 'TestTrack1000'

        self.start = [10, 90]
        self.end = [90, 10]

        if add_obs:
            self.obs_locs = [[15, 50], [32, 24], [50, 45], [70, 74], [88, 40]]
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

    def set_up_map(self):
        self.hm_name = 'DataRecords/' + self.name + '_heatmap.npy'
        self.path_name = "DataRecords/" + self.name + "_path.npy" # move to setup call


        self.create_race_map()
        # self._place_obs()
        self.run_path_finder()


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

        if self.race_map[int(x[0]), int(x[1])]:
        # if self.race_map[int(x[1]), int(x[0])]:
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
        self.heat_map = new_map

        # fig = plt.figure(1)
        # plt.imshow(self.heat_map.T, origin='lower')
        # fig = plt.figure(2)
        # plt.imshow(self.race_map.T, origin='lower')
        # plt.show()

    def _path_finder_collision(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if self.y_bound[0] > x[1] or x[1] > self.y_bound[1]:
            return True 

        if self.heat_map[int(x[0]), int(x[1])]:
            return True

        return False


class TestEnvDQN(TestMap, CarModelDQN):
    def __init__(self):
        self.steps = 0
        self.memory = []
        self.speed_memory = []
        self.eps = 0
        
        self.n_ranges = 10 
        self.state_space = 2 + self.n_ranges
        self.dth_action = 0.4

        TestMap.__init__(self)
        CarModelDQN.__init__(self, self.n_ranges)

        self.wpts = None
        self.pind = None
        self.path_name = None
        self.target = None

        self.lp_th = None
        self.lp_sp = None

        self.step_size = 1
        self.n_searches = 30

    def run_path_finder(self):
        # self.show_map()
        # self.show_hm()
        try:
            # raise Exception
            self.wpts = np.load(self.path_name)
            # print(f"Path Loaded")
        except:
            path_finder = PathFinder(self._path_finder_collision, self.start, self.end)
            path = path_finder.run_search(2)
            path = modify_path(path)
            self.wpts = path
            np.save(self.path_name, self.wpts)
            print("Path Generated")

        self.wpts = np.append(self.wpts, self.end)
        self.wpts = np.reshape(self.wpts, (-1, 2))
        # print(f"Wpts including end")

        new_pts = []
        for wpt in self.wpts:
            if not self._check_location(wpt):
                new_pts.append(wpt)
            else:
                pass
                # print(f"Wpt removed: {wpt}")
        self.wpts = np.asarray(new_pts)    

        # self.show_map(self.wpts)
      
    def reset(self):
        self.eps += 1
        self.steps = 0
        self.memory = []

        self.theta = np.pi / 2 - np.arctan(lib.get_gradient(self.start, self.end))
        if self.end[0] < self.start[0]:
            self.theta += np.pi
        
        self.last_distance = lib.get_distance(self.start, self.end)
        self.car_x = self.start
        self.velocity = 0
        self.steering = 0
        self._update_ranges()
        self.pind = 1 # first wpt

        return self._get_state_obs()

    def step(self, action):
        self.memory.append(self.car_x)
        self.steps += 1

        th_mod = (action[0] - 2) * self.dth_action
        self.action = [self.lp_th + th_mod, self.lp_sp]

        new_x = self._x_step(self.action)
        crash = self._check_location(new_x)
        if not crash:
            self.update_state(self.action)

        self.calculate_reward(crash, action)
        r = self.reward
        obs = self._get_state_obs()

        return obs, r, crash, None

    def calculate_reward(self, crash, action):
        if crash:
            self.reward = -1
            return 
        
        alpha = 0.5
        self.reward = 1 - alpha * abs(action[0])
   
    def render(self):
        x, y = [], []
        for step in self.memory:
            x.append(step[0])
            y.append(step[1])

        fig = plt.figure(4)
        plt.clf()  
        plt.imshow(self.race_map.T, origin='lower')
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.plot(x, y)
        plt.plot(self.start[0], self.start[1], '*', markersize=20)
        plt.plot(self.end[0], self.end[1], '*', markersize=20)

        for pt in self.wpts:
            plt.plot(pt[0], pt[1], 'x', markersize=10)
        
        plt.pause(0.001)
        # fig.savefig(f"Renders/Rendering_{self.eps}")

    def _get_state_obs(self):
        self._update_ranges()

        self._set_target()
        if self._check_location(self.target):
            self.pind += 1
            self._set_target()

        self.set_lp_action()

        lp_sp = self.lp_sp / self.max_velocity
        lp_th = self.lp_th / np.pi

        self._update_ranges()

        obs = np.concatenate([[lp_th], [lp_sp], self.ranges])

        return obs

    def set_lp_action(self):
        th_target = lib.get_bearing(self.car_x, self.target) 
        self.lp_th = th_target - self.theta

        # called in the reset
        # rel_target = lib.sub_locations(self.end, self.car_x)
        # transformed_target = lib.transform_coords(rel_target, self.theta)
        # normalised_target = lib.normalise_coords(transformed_target)
        # self.lp_th = np.arctan(lib.get_gradient([0, 0], normalised_target))

        # speed between 1 and 0.4
        # self.lp_sp = (1 - abs(self.lp_th)/(np.pi/2) * 0.6) * self.max_velocity
        self.lp_sp = 1

    def _set_target(self):
        dis_last_pt = lib.get_distance(self.wpts[self.pind-1], self.car_x)
        dis_next_pt = lib.get_distance(self.car_x, self.wpts[self.pind+1])
        max_update_dis = 20 # must be this close to got to next pt
        while dis_next_pt < dis_last_pt and \
            self.pind < len(self.wpts)-2 and  \
            dis_next_pt < max_update_dis:
            self.pind += 1

            dis_last_pt = lib.get_distance(self.wpts[self.pind-1], self.car_x)
            dis_next_pt = lib.get_distance(self.car_x, self.wpts[self.pind+1])
        # use pind and pind +1 to get a 5 unit distance vector 
        next_pt = self.wpts[self.pind]
        next_next_pt = self.wpts[self.pind+1]

        target = lib.add_locations(next_pt, next_next_pt)
        target = [target[0]/2, target[1]/2]

        self.target = target

    def _update_ranges(self):
        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            for j in range(self.n_searches): # number of search points
                fs = self.step_size * j
                dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
                search_val = lib.add_locations(self.car_x, dx)
                if self._check_location(search_val):
                    break             
            self.ranges[i] = (j) / self.n_searches # gives a scaled val to 1 

    def box_render(self):
        box = int(self.map_dim / 5)
        car_x = int(self.car_x[0])
        car_y = int(self.car_x[1])
        x_min = max(0, car_x-box)
        y_min = max(0, car_y-box)
        x_max = min(self.map_dim, x_min+2*box)
        y_max = min(self.map_dim, y_min+2*box)
        plot_map = self.race_map[x_min:x_max, y_min:y_max]
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        car_pos = [car_x - x_mid + box, car_y - y_mid + box]

        fig = plt.figure(5)
        plt.clf()  
        plt.imshow(plot_map.T, origin='lower')
        plt.xlim(0, box * 2)
        plt.ylim(0, box * 2)

        plt.plot(car_pos[0], car_pos[1], '+', markersize=20)
        targ = lib.add_locations(self.target, car_pos)
        plt.plot(targ[0], targ[1], 'x', markersize=14)

        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            fs = self.ranges[i] * self.step_size * self.n_searches
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations(car_pos, dx)
            x = [car_pos[0], range_val[0]]
            y = [car_pos[1], range_val[1]]
            plt.plot(x, y)

        
        plt.pause(0.01)
        # plt.show()

    def full_render(self):
        car_pos = self.car_x

        fig = plt.figure(6)
        plt.clf()  
        plt.imshow(self.race_map.T, origin='lower')
        plt.xlim(0, self.map_dim)
        plt.ylim(0, self.map_dim)

        plt.plot(car_pos[0], car_pos[1], '+', markersize=20)
        plt.plot(self.target[0], self.target[1], 'x', markersize=14)

        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            fs = self.ranges[i] * self.step_size * self.n_searches
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations(car_pos, dx)
            x = [car_pos[0], range_val[0]]
            y = [car_pos[1], range_val[1]]
            plt.plot(x, y)

        
        plt.pause(0.01)

    def render_snapshot(self):
        fig = plt.figure(8)
        plt.clf()  
        plt.imshow(self.race_map.T, origin='lower')
        plt.xlim(0, self.map_dim)
        plt.ylim(0, self.map_dim)

        xs, ys = [], []
        for x in self.memory:
            xs.append(x[0])
            ys.append(x[1])
        plt.plot(xs, ys, '+', markersize=12)
        plt.plot(xs, ys, linewidth=3)

        xs, ys = [], []
        for pt in self.wpts:
            xs.append(pt[0])
            ys.append(pt[1])
        
        # plt.plot(xs, ys)
        plt.plot(xs, ys, 'x', markersize=20)

        s = f"Steps: {self.steps}"
        plt.text(100, 80, s)
        s = f"Average speed: {np.mean(self.speed_memory)}"
        plt.text(100, 70, s)
            
        plt.pause(0.001)



