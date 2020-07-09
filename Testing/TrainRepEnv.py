import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt

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
    def __init__(self, name):
        self.name = name
        self.map_dim = 100

        self.race_map = None
        self.start = None
        self.end = None

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

        self.create_race_map()

    def create_race_map(self):
        track_name = 'DataRecords/' + self.name + '.npy'
        new_map = np.load(track_name)
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

class TrainRepEnv(TrainMap, CarModel):
    def __init__(self, name):
        self.steps = 0
        self.memory = []
        self.eps = 0
        
        self.n_ranges = 10 
        self.state_space = 2 + self.n_ranges

        TrainMap.__init__(self, name)
        CarModel.__init__(self, self.n_ranges)
        self.set_start_end()
      
    def reset(self):
        self.eps += 1
        self.steps = 0
        self.memory.clear()

        self.set_start_end()

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
        reward = reward * 0.01 # scale to -1, 1
        obs = self._get_state_obs()

        return obs, reward, done, None

    def _get_reward(self, crash):
        if crash:
            r_crash = -50
            return r_crash, True

        beta = 0.8 # scale to 
        r_done = 100
        # step_penalty = 5
        max_steps = 1000

        cur_distance = lib.get_distance(self.car_x, self.end)
        if cur_distance < 1 + self.action_scale:
            return r_done, True
        d_dis = self.last_distance - cur_distance
        reward = 0
        if abs(d_dis) > 0.01:
            reward = beta * (d_dis**2 * d_dis/abs(d_dis)) # - step_penalty
            # reward = beta * d_dis# - step_penalty
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
        
        plt.pause(0.001)
        # fig.savefig(f"Renders/Rendering_{self.eps}")

    def _get_state_obs(self):
        self._update_ranges()
        rel_target = lib.sub_locations(self.end, self.car_x)
        transformed_target = lib.transform_coords(rel_target, self.theta)
        normalised_target = lib.normalise_coords(transformed_target)
        obs = np.concatenate([normalised_target, self.ranges])

        return obs

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

        
        plt.pause(0.2)


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
