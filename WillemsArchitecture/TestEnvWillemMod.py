import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt

from PathFinder import PathFinder, modify_path


class CarModelDQN:
    def __init__(self):
        self.n_ranges = 5
        self.state_space = self.n_ranges + 2
        self.action_space = 5
        self.center_act = 2

        self.car_x = [0, 0]
        self.theta = 0
        self.velocity = 0
        self.steering = 0

        self.lp_th = None
        self.lp_sp = None

        self.max_velocity = 4
        self.dth_action = np.pi/5 # amount of rad to swerve with each action 

        self.ranges = np.zeros(self.n_ranges)
        self.range_angles = np.zeros(self.n_ranges)
        dth = np.pi/(self.n_ranges-1)
        for i in range(self.n_ranges):
            self.range_angles[i] = i * dth - np.pi/2

    def _x_step(self, action):
        a, d_dot = self.get_integrals(action) 
        dt = 1

        theta = self.theta + action[0]
        velocity = self.velocity + a * dt
        x_i = np.sin(theta)* velocity * dt + self.car_x[0]
        y_i = np.cos(theta)* velocity * dt + self.car_x[1]
        
        return [x_i, y_i]

    def update_state(self, action):
        a, d_dot = self.get_integrals(action) 

        dt = 1

        self.theta = self.theta + action[0] 
        self.velocity = self.velocity + a * dt

        x_i = np.sin(self.theta)* self.velocity * dt + self.car_x[0]
        y_i = np.cos(self.theta)* self.velocity * dt + self.car_x[1]
        
        self.car_x = [x_i, y_i]
        
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

    def set_lp_action(self, target):
        th_target = lib.get_bearing(self.car_x, target) 
        self.lp_th = th_target - self.theta

        # speed between 1 and 0.4
        # self.lp_sp = (1 - abs(self.lp_th)/(np.pi/2) * 0.6) * self.max_velocity
        self.lp_sp = 1

    def _get_state_obs(self, target):
        self._update_ranges()
        self.set_lp_action(target)

        lp_sp = self.lp_sp 
        lp_th = self.lp_th #/ np.pi

        xy = lib.theta_to_xy(lp_th)

        obs = np.concatenate([xy, self.ranges])

        return obs

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

    def set_up_map(self):
        self.hm_name = 'DataRecords/' + self.name + '_heatmap.npy'
        self.path_name = "DataRecords/" + self.name + "_path.npy" # move to setup call


        self.create_race_map()
        self._place_obs()
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


class TestEnvDQN(TestMap, CarModelDQN):
    def __init__(self):
        self.steps = 0
        self.memory = []
        self.speed_memory = []
        self.eps = 0
        
        CarModelDQN.__init__(self)
        TestMap.__init__(self)

        self.wpts = None
        self.pind = None
        self.path_name = None
        self.target = None

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
            path_finder = PathFinder(self._check_line_path, self.start, self.end)
            path = path_finder.run_search(2)
            # self.show_map(path)
            path = modify_path(path)
            self.wpts = path
            np.save(self.path_name, self.wpts)
            print("Path Generated")
            # self.show_map(path)

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

        # self.theta = np.pi / 2 - np.arctan(lib.get_gradient(self.start, self.end))
        # if self.end[0] < self.start[0]:
        #     self.theta += np.pi
        self.theta = lib.get_bearing(self.start, self.wpts[1])
        
        self.last_distance = lib.get_distance(self.start, self.end)
        self.car_x = self.start
        self.velocity = 0
        self.steering = 0
        self._update_ranges()
        self.pind = 1 # first wpt

        self.memory.append(self.car_x)
        self._set_target()
        return self._get_state_obs(self.target)

    def step(self, action):
        self.steps += 1

        th_mod = (action[0] - self.center_act) * self.dth_action
        self.action = [self.lp_th + th_mod, self.lp_sp]

        new_x = self._x_step(self.action)
        crash = self._check_line(new_x, self.car_x) or (self.steps > 200)
        if not crash:
            self.update_state(self.action)

        self.calculate_reward(crash, action)
        r = self.reward
        self._set_target()
        obs = self._get_state_obs(self.target)
        done = self.check_done() or crash

        self.speed_memory.append(self.velocity)
        self.memory.append(self.car_x)


        return obs, r, done, None

    def calculate_reward(self, crash, action):
        if crash:
            self.reward = -1
            return 
        
        alpha = 0.1
        self.reward = 1 - alpha * abs(action[0] - self.center_act)

    def check_done(self):
        if lib.get_distance(self.car_x, self.end) < 5:
            return True
        return False
   
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



    def _set_target(self):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], self.car_x)
        if dis_cur_target < 10 and self.pind < len(self.wpts)-2: # how close to say you were there
            self.pind += 1
        
        target = self.wpts[self.pind]

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

        
        plt.pause(0.1)
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



