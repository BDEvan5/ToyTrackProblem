import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt




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
        self.action_scale = 20

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

        return new_x, new_theta
    




class TestMap:
    def __init__(self, name='DataRecords/TrainTrack1000.npy'):
        self.name = name
        self.map_dim = 1000

        self.race_map = None
        self.start = [36, 80]
        self.end = [948, 700] 

        self.x_bound = [1, 999]
        self.y_bound = [1, 999]

        self.create_race_map()
        # self.race_map = np.flip(self.race_map)

    def create_race_map(self):
        array = np.load(self.name)
        new_map = np.zeros((self.map_dim, self.map_dim))
        block_range = self.map_dim / array.shape[0]
        for i in range(self.map_dim):
            for j in range(self.map_dim):
                new_map[i, j] = array[int(i/block_range), int(j/block_range)]

        self.race_map = new_map

    def _check_location(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if self.y_bound[0] > x[1] or x[1] > self.y_bound[1]:
            return True 

        # if self.race_map[int(x[0]), int(x[1])]:
        if self.race_map[int(x[1]), int(x[0])]:
            return True

        return False

    def show_map(self):
        plt.imshow(self.race_map, origin='lower')
        plt.plot(self.start[0], self.start[1], '*', markersize=20)
        plt.plot(self.end[0], self.end[1], '*', markersize=20)
        # plt.show()
        plt.pause(0.001)


class TestEnv(TestMap, CarModel):
    def __init__(self, name):
        self.steps = 0
        self.memory = []
        self.eps = 0
        
        self.n_ranges = 10 
        self.state_space = 2 + self.n_ranges

        TestMap.__init__(self, name)
        CarModel.__init__(self, self.n_ranges)
      
    def reset(self):
        self.eps += 1
        self.steps = 0
        self.memory = []

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
        if self._check_location(new_x):
            obs = self._get_state_obs()
            r_crash = -100
            return obs, r_crash, True, None
        self.car_x = new_x 
        self.theta = new_theta
        obs = self._get_state_obs()
        reward, done = self._get_reward()
        return obs, reward, done, None

    def _get_reward(self):
        beta = 0.6
        r_done = 100
        step_penalty = 5
        max_steps = 1000

        cur_distance = lib.get_distance(self.car_x, self.end)
        if cur_distance < 1 + self.action_scale:
            return r_done, True
        d_dis = self.last_distance - cur_distance
        reward = beta * (d_dis**2 * d_dis/abs(d_dis)) - step_penalty
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
        plt.imshow(self.race_map, origin='lower')
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.plot(x, y)
        plt.plot(self.start[0], self.start[1], '*', markersize=20)
        plt.plot(self.end[0], self.end[1], '*', markersize=20)
        
        plt.pause(0.001)
        # fig.savefig(f"Renders/Rendering_{self.eps}")

    def _get_state_obs(self):
        self._update_ranges()
        rel_target = lib.sub_locations(self.end, self.car_x)
        transformed_target = lib.transform_coords(rel_target, self.theta)
        obs = np.concatenate([transformed_target, self.ranges])

        return obs

    def _update_ranges(self):
        step_size = 20
        n_searches = 8
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
        box = 100
        car_x = int(self.car_x[0])
        car_y = int(self.car_x[1])
        x_min = max(0, car_x-box)
        y_min = max(0, car_y-box)
        x_max = min(1000, x_min+2*box)
        y_max = min(1000, y_min+2*box)
        # plot_map = self.race_map[x_min:x_max, y_min:y_max]
        plot_map = self.race_map[y_min:y_max, x_min:x_max]
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        car_pos = [car_x - x_mid + box, car_y - y_mid + box]

        fig = plt.figure(5)
        plt.clf()  
        plt.imshow(plot_map, origin='lower')
        plt.xlim(0, 200)
        plt.ylim(0, 200)

        plt.plot(car_pos[0], car_pos[1], '+', markersize=16)

        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            fs = self.ranges[i] * 8 * 20
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations(car_pos, dx)
            x = [car_pos[0], range_val[0]]
            y = [car_pos[1], range_val[1]]
            plt.plot(x, y)

        
        plt.pause(0.01)
        # plt.show()


if __name__ == "__main__":
    name00 = 'DataRecords/TrainTrack1000.npy'
    name10 = 'DataRecords/TrainTrack1010.npy'

    # test_map = TestMap(name00)
    # test_map.show_map()

    env = TestEnv(name10)
    env.reset()
    env.step(1)
    env.show_map()
