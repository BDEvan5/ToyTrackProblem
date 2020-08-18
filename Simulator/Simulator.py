import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image

import LibFunctions as lib


class CarModel:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.steering = 0

        self.wheelbase = 0.33
        self.mass = 3.74
        self.len_cg_rear = 0.17
        self.I_z = 0.047
        self.mu = 0.523
        self.height_cg = 0.074
        self.cs_f = 4.718
        self.cs_r = 5.45

        self.max_d_dot = 3.2
        self.max_steer = 0.4
        self.max_a = 7.5
        self.max_decel = -8.5
        self.max_v = 7.5


    def update_kinematic_state(self, a, d_dot, dt):
        self.x = self.x + self.velocity * np.sin(self.theta) * dt
        self.y = self.y + self.velocity * np.cos(self.theta) * dt
        theta_dot = self.velocity / self.wheelbase * np.tan(self.steering)
        self.theta = self.theta + theta_dot * dt

        a = np.clip(a, self.max_decel, self.max_a)
        d_dot = np.clip(d_dot, -self.max_d_dot, self.max_d_dot)

        self.steering = self.steering + d_dot * dt
        self.velocity = self.velocity + a * dt

        self.steering = np.clip(self.steering, -self.max_steer, self.max_steer)
        self.velocity = np.clip(self.velocity, -self.max_v, self.max_v)

        if self.theta > np.pi:
            self.theta = self.theta - 2*np.pi
        if self.theta < -np.pi:
            self.theta += 2*np.pi

    def get_car_state(self):
        state = []
        state.append(self.x)
        state.append(self.y)
        state.append(self.theta)
        state.append(self.velocity)
        state.append(self.steering)

        return state



        

class ScanSimulator:
    def __init__(self, number_of_beams=10, fov=np.pi, std_noise=0.01):
        self.number_of_beams = number_of_beams
        self.fov = fov 
        self.std_noise = std_noise

        self.dth = self.fov / (self.number_of_beams -1)
        self.scan_output = np.zeros(number_of_beams)

        self.step_size = 1
        self.n_searches = 30

        self.race_map = None
        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

    def get_scan(self, x, y, theta):
        for i in range(self.number_of_beams):
            scan_theta = theta + self.dth * i - self.fov/2
            self.scan_output[i] = self.trace_ray(x, y, scan_theta)

        return self.scan_output

    def trace_ray(self, x, y, theta):
        for j in range(self.n_searches): # number of search points
            fs = self.step_size * j
            dx =  [np.sin(theta) * fs, np.cos(theta) * fs]
            search_val = lib.add_locations([x, y], dx)
            if self._check_location(search_val):
                break       

        return j / self.n_searches      


    def set_map(self, check_fcn):
        self._check_location = check_fcn


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
        # self.run_path_finder()


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
    
    def run_path_finder(self):
        try:
            # raise Exception
            self.wpts = np.load(self.path_name)
            # print(f"Path Loaded")
        except:
            path_finder = PathFinder(self._check_line_path, self.start, self.end)
            path = path_finder.run_search(2)
            path = modify_path(path)
            self.wpts = path
            np.save(self.path_name, self.wpts)
            print("Path Generated")

        self.wpts = np.append(self.wpts, self.end)
        self.wpts = np.reshape(self.wpts, (-1, 2))

        new_pts = []
        for wpt in self.wpts:
            if not self._check_location(wpt):
                new_pts.append(wpt)
            else:
                pass
        self.wpts = np.asarray(new_pts)    

        # self.show_map(self.wpts)



class F110Env:
    def __init__(self):
        self.timestep = 0.01

        self.race_map = TestMap()
        self.race_map.map_1000(True)

        self.car = CarModel()
        self.scan_sim = ScanSimulator(10)
        self.scan_sim.set_map(self.race_map._check_location)

        self.done = False
        self.reward = 0
        self.action = np.zeros((2, 1))


    def step(self, action):
        self.action = action
        acceleration = action[0]
        steer_dot = action[1]

        self.car.update_kinematic_state(acceleration, steer_dot, self.timestep)
        self.check_done()
        self.update_reward()

        obs = self.get_state_obs()
        done = self.done
        reward = self.reward

        return obs, reward, done, None


    def reset(self, poses=None):
        self.done = False

        if poses is not None:
            self.x = poses['x']
            self.y = poses['y']
            self.theta = poses['theta']
            self.steering = 0
            self.velocity = 0
        else:
            # self.generate_random_start()
            self.start = [10, 90]
            self.end = [90, 10]
            self.car.x = self.start[0]
            self.car.y = self.start[1]
            self.car.velocity = 0
            self.car.steering = 0
            self.car.theta = 3

        

        return self.get_state_obs()


    def get_state_obs(self):
        car_state = self.car.get_car_state()
        scan = self.scan_sim.get_scan(self.car.x, self.car.y, self.car.theta)

        state = np.concatenate([car_state, scan])

        return state

    def check_done(self):
        if self.race_map._check_location([self.car.x, self.car.y]):
            self.done = True

    def update_reward(self):
        self.reward = 1
    
    def generate_random_start(self):
        self.start = lib.get_rands()
        while self.race_map._check_location(self.start):
            self.start = lib.get_rands()
        self.car.x = self.start[0]
        self.car.y = self.start[1]

        self.end = lib.get_rands()
        while self.race_map._check_location(self.end) or \
            lib.get_distance(self.start, self.end) < 30:
            self.end = lib.get_rands()
        self.end = lib.get_rands(80, 10)

        rand_theta_mod = + np.random.random() * np.pi/2 - np.pi/4
        self.car.theta = lib.get_bearing(self.start, self.end) #+ rand_theta_mod

        self.steering = 0
        self.velocity = 0

    def render(self, wait=False):
        car_x = int(self.car.x)
        car_y = int(self.car.y)
        fig = plt.figure(4)
        plt.clf()  
        plt.imshow(self.race_map.race_map.T, origin='lower')
        plt.xlim(0, self.race_map.map_dim)
        plt.ylim(-10, self.race_map.map_dim)
        plt.plot(self.start[0], self.start[1], '*', markersize=12)

        plt.plot(self.end[0], self.end[1], '*', markersize=12)
        plt.plot(self.car.x, self.car.y, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - np.pi/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            x = [car_x, range_val[0]]
            y = [car_y, range_val[1]]
            plt.plot(x, y)

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(100, 80, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(100, 70, s) 
        s = f"Done: {self.done}"
        plt.text(100, 65, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(100, 60, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(100, 55, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(100, 50, s) 
        s = f"Delta: [{self.car.steering:.2f}]"
        plt.text(100, 45, s) 

        plt.pause(0.001)
        if wait:
            plt.show()


def CorridorAction(obs):
    # all this does is go in the direction of the biggest range finder
    ranges = obs[5:]
    max_range = np.argmax(ranges)
    dth = np.pi / 9
    heading_ref = dth * max_range - np.pi/2
    d_heading = heading_ref - obs[4] # d_delta

    kp_delta = 1
    d_dot = d_heading * kp_delta

    a = 0
    if obs[3] < 6:
        a = 8

    return [a, d_dot]



def sim_driver():
    env = F110Env()

    done, state, score = False, env.reset(None), 0.0
    while not done:
        action = CorridorAction(state)
        s_p, r, done, _ = env.step(action)
        score += r
        state = s_p

        # env.render(True)
        env.render(False)

    print(f"Score: {score}")

if __name__ == "__main__":
    sim_driver()
