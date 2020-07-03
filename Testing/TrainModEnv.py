import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt

from PathFinder import PathFinder

name00 = 'DataRecords/TrainTrack1000.npy'
name10 = 'DataRecords/TrainTrack1010.npy'
name20 = 'DataRecords/TrainTrack1020.npy'


class CarModel:
    def __init__(self, n_ranges=10):
        self.n_ranges = n_ranges
        self.car_x = [0, 0]
        self.theta = 0
        self.ranges = np.zeros(self.n_ranges)
        self.range_angles = np.zeros(self.n_ranges)
        dth = np.pi/(self.n_ranges-1)
        for i in range(self.n_ranges):
            self.range_angles[i] = i * dth - np.pi/2

        # parameters
        self.action_space = 10
        self.action_scale = 8

    def _x_step_discrete(self, action):
        # actions in range [0, n_acts) are a fan in front of vehicle
        # no backwards
        fs = self.action_scale
        dth = np.pi / (self.action_space-1)
        angle = -np.pi/2 + action * dth 
        angle += self.theta # for the vehicle offset
        dx = [np.sin(angle)*fs, np.cos(angle)*fs] 
        
        new_x = lib.add_locations(dx, self.car_x)
        
        new_grad = lib.get_gradient(new_x, self.car_x)
        new_theta = np.pi / 2 - np.arctan(new_grad)
        if dx[0] < 0:
            new_theta += np.pi
        if new_theta >= 2*np.pi:
            new_theta = new_theta - 2*np.pi

        return new_x, new_theta
    
    def random_action(self):
        return np.random.randint(0, self.action_space-1)

class TrainMap:
    def __init__(self, name='DataRecords/TrainTrack1020.npy'):
        self.name = name
        self.map_dim = 100

        self.race_map = None
        self.start = None
        self.end = None

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

        self.create_race_map()

    def create_race_map(self):
        new_map = np.load(self.name)
        t_map = new_map
        self.race_map = t_map

    def _check_location(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if self.y_bound[0] > x[1] or x[1] > self.y_bound[1]:
            return True 

        if self.race_map[int(x[0]), int(x[1])]:
            return True

        return False

    def show_map(self):
        plt.imshow(self.race_map, origin='lower')
        plt.plot(self.start[0], self.start[1], '*', markersize=20)
        plt.plot(self.end[0], self.end[1], '*', markersize=20)
        for pt in self.wpts:
            plt.plot(pt[0], pt[1], 'x', markersize=10)
        plt.show()
        # plt.pause(0.001)

    def set_start_end(self):
        rands = np.random.rand(4) * 100
        self.start = rands[0:2]
        self.end = rands[2:4]  

        while self.race_map[int(self.start[0]), int(self.start[1])]:
            self.start = np.random.rand(2) * 100
        while self.race_map[int(self.end[0]), int(self.end[1])] or \
            lib.get_distance(self.start, self.end) < 15:
            self.end = np.random.rand(2) * 100

    def _set_up_heat_map(self):
        track_map = self.race_map
        for _ in range(15): # blocks up to 5 away will start to have a gradient
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


class TrainModEnv(TrainMap, CarModel):
    def __init__(self, name):
        self.steps = 0
        self.memory = []
        self.eps = 0
        
        self.n_ranges = 10 
        self.state_space = 2 + self.n_ranges

        TrainMap.__init__(self, name)
        CarModel.__init__(self, self.n_ranges)
        self.set_start_end()
        self.wpts = []
        self.pind = 1

    def run_path_finder(self):
        # path_finder = PathFinder(self._path_finder_collision, self.start, self.end)
        # path_finder = PathFinder(self.return_false, self.start, self.end)
        # path = path_finder.run_search(10)
        wpt_n = 6
        wpts = []
        dx = lib.sub_locations(self.end, self.start)
        dxx = dx[0] / wpt_n
        dyy = dx[1] / wpt_n
        for i in range(wpt_n):
            pt = lib.add_locations(self.start, [dxx, dyy], i)
            wpts.append(pt)

        self.wpts = wpts
        self.target = None

        # self.show_map()

    def return_false(self, x):
        return False
        
    def reset(self):
        self.eps += 1
        self.steps = 0
        self.memory.clear()

        self.set_start_end()
        self._set_up_heat_map()
        self.run_path_finder()
        self.pind = 1

        self.theta = np.pi / 2 - np.arctan(lib.get_gradient(self.start, self.end))
        if self.end[0] < self.start[0]:
            self.theta += np.pi
        
        self.last_distance = lib.get_distance(self.start, self.end)
        self.car_x = self.start
        self._update_ranges()

        return self._get_state_obs()

    def step(self, action):
        self.memory.append(self.car_x)
        self.steps += 1

        new_x, new_theta = self._x_step_discrete(action)
        crash = self._check_location(new_x) 
        if not crash:
            self.car_x = new_x
            self.theta = new_theta
        reward, done = self._get_reward(crash)
        obs = self._get_state_obs()

        reward2 = int(not crash)
        reward = 0.01 * reward
        return obs, reward, done, reward2 

    def _get_reward(self, crash):
        if crash:
            r_crash = -100
            return r_crash, True

        beta = 0.5 # scale to 
        r_done = 100
        step_penalty = 5
        max_steps = 1000

        cur_distance = lib.get_distance(self.car_x, self.end)
        if cur_distance < 1 + self.action_scale:
            return r_done, True
        d_dis = self.last_distance - cur_distance
        reward = beta * (d_dis**2 * d_dis/abs(d_dis)) # - step_penalty
        self.last_distance = cur_distance
        done = True if self.steps > max_steps else False
        return reward, done

    def render(self):
        x, y = [], []
        for step in self.memory:
            x.append(step[0])
            y.append(step[1])

        fig = plt.figure(4)
        plt.clf()  
        plt.imshow(self.race_map.T, origin='lower')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.plot(x, y)
        plt.plot(self.start[0], self.start[1], '*', markersize=20)
        plt.plot(self.end[0], self.end[1], '*', markersize=20)
        for pt in self.wpts:
            plt.plot(pt[0], pt[1], 'x', markersize=10)
        
        plt.pause(0.001)
        # fig.savefig(f"Renders/Rendering_{self.eps}")

    def _get_state_obs(self):
        self._update_ranges()
        target = self._get_target()

        rel_target = lib.sub_locations(target, self.car_x)
        self.target = rel_target
        transformed_target = lib.transform_coords(rel_target, self.theta)
        normalised_target = lib.normalise_coords(transformed_target)
        obs = np.concatenate([normalised_target, self.ranges])

        return obs

    def _get_target(self):
        dis_last_pt = lib.get_distance(self.wpts[self.pind-1], self.car_x)
        dis_next_pt = lib.get_distance(self.car_x, self.wpts[self.pind+1])
        while dis_next_pt < dis_last_pt and self.pind < len(self.wpts)-2:
            self.pind += 1

            dis_last_pt = lib.get_distance(self.wpts[self.pind-1], self.car_x)
            dis_next_pt = lib.get_distance(self.car_x, self.wpts[self.pind+1])
        # use pind and pind +1 to get a 5 unit distance vector 
        next_pt = self.wpts[self.pind]
        next_next_pt = self.wpts[self.pind+1]

        target = lib.add_locations(next_pt, next_next_pt)
        target = [target[0]/2, target[1]/2]

        return target

    def _update_ranges(self):
        step_size = 3
        n_searches = 15
        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            for j in range(n_searches): # number of search points
                fs = step_size * j
                dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
                search_val = lib.add_locations(self.car_x, dx)
                if self._check_location(search_val):
                    break             
            self.ranges[i] = (j-1) / n_searches # gives a scaled val to 1 

    def box_render(self):
        car_x = int(self.car_x[0])
        car_y = int(self.car_x[1])

        fig = plt.figure(5)
        plt.clf()  
        plt.imshow(self.race_map.T, origin='lower')
        plt.xlim(0, 100)
        plt.ylim(0, 100)

        plt.plot(car_x, car_y, '+', markersize=16)

        plt.plot(self.start[0], self.start[1], '*', markersize=12)
        plt.plot(self.end[0], self.end[1], '*', markersize=12)

        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            fs = self.ranges[i] * 15 * 3
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations(self.car_x, dx)
            x = [car_x, range_val[0]]
            y = [car_y, range_val[1]]
            plt.plot(x, y)

        for pt in self.wpts:
            plt.plot(pt[0], pt[1], 'x', markersize=10)
        
        plt.pause(0.2)

    def full_render(self):
        car_pos = self.car_x

        fig = plt.figure(5)
        plt.clf()  
        plt.imshow(self.race_map.T, origin='lower')
        plt.xlim(0, 100)
        plt.ylim(0, 100)

        plt.plot(car_pos[0], car_pos[1], '+', markersize=20)
        targ = lib.add_locations(car_pos, self.target)
        plt.plot(targ[0], targ[1], 'x', markersize=14)

        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            fs = self.ranges[i] * 8 * 20
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations(car_pos, dx)
            x = [car_pos[0], range_val[0]]
            y = [car_pos[1], range_val[1]]
            plt.plot(x, y)

        for pt in self.wpts:
            plt.plot(pt[0], pt[1], 'x', markersize=10)

        plt.pause(0.01)
        # plt.show()



def test_TrainEnv():
    env = TrainEnv(name20)
    env.show_map()

    env.reset()
    print_n = 1000
    for i in range(100000):
        s, d, r, _ = env.step(np.random.randint(0, env.action_space-1))
        if d:
            env.reset()
        if i % print_n == 1:
            print(f"Running test: {i}")

if __name__ == "__main__":
    test_TrainEnv()
