import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt

class BaseEnv:
    def __init__(self):
        # parameters
        self.n_ranges = 10
        self.action_space = 10
        self.state_dim = 2 + self.n_ranges
        self.max_action = 1 # what the net gives
        self.action_dim = 2
        self.action_scale = 5 # internal implementation

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

        # state info
        self.car_x = [0, 0]
        self.theta = 0
        self.ranges = np.zeros(self.n_ranges)

        # internals
        self.o_grid = np.zeros((101, 101))
        self.range_angles = np.zeros(self.n_ranges)
        self.last_distance = None
        self.start = None
        self.end = None
        self.steps = 0
        self.eps = 0
        self.memory = []

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

        return new_x, new_theta

    def _get_reward(self):
        beta = 0.6
        r_done = 100
        step_penalty = 5
        max_steps = 100

        cur_distance = lib.get_distance(self.car_x, self.end)
        if cur_distance < 1 + self.action_scale:
            return r_done, True
        d_dis = self.last_distance - cur_distance
        reward = beta * (d_dis**2 * d_dis/abs(d_dis)) - step_penalty
        self.last_distance = cur_distance
        done = True if self.steps > max_steps else False
        return reward, done

    def _get_state_obs(self):
        self._update_ranges()
        rel_target = lib.sub_locations(self.end, self.car_x)
        transformed_target = lib.transform_coords(rel_target, self.theta)
        obs = np.concatenate([transformed_target, self.ranges])

        return obs

    def _update_ranges(self):
        step_size = 8
        n_searches = 6
        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            for j in range(n_searches): # number of search points
                fs = step_size * j
                dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
                search_val = lib.add_locations(self.car_x, dx)
                if self._check_location(search_val):
                    break             
            self.ranges[i] = (j-1) / n_searches # gives a scaled val to 1 
        
    def render(self):
        x, y = [], []
        for step in self.memory:
            x.append(step[0])
            y.append(step[1])

        fig = plt.figure(4)
        plt.clf()  
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.plot(x, y)
        plt.plot(self.start[0], self.start[1], '*', markersize=20)
        plt.plot(self.end[0], self.end[1], '*', markersize=20)

        ax = fig.gca()
        for o in self.obstacles:
            circle = plt.Circle(o.location, o.size, color='r')
            ax.add_artist(circle)
        
        plt.pause(0.001)
        # fig.savefig(f"Renders/Rendering_{self.eps}")

    def _check_location(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if self.y_bound[0] > x[1] or x[1] > self.y_bound[1]:
            return True 

        if self.o_grid[int(x[0]), int(x[1])]:
            return True
        return False
    

class TrainEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.obstacles = []
        self.set_o_map()

    def reset(self):
        self.eps += 1
        self.steps = 0
        self.memory = []
        self.set_start_end()

        self.theta = np.pi / 2 - np.arctan(lib.get_gradient(self.start, self.end))
        if self.end[0] < self.start[0]:
            self.theta += np.pi
        
        self.last_distance = lib.get_distance(self.start, self.end)
        self.car_x = self.start

        return self._get_state_obs()

    def set_start_end(self):
        rands = np.random.rand(4) * 100
        self.start = rands[0:2]
        self.end = rands[2:4]  

        while self.o_grid[int(self.start[0]), int(self.start[1])]:
            self.start = np.random.rand(2) * 100
        while self.o_grid[int(self.end[0]), int(self.end[1])] or \
            lib.get_distance(self.start, self.end) < 15:
            self.end = np.random.rand(2) * 100

    def random_discrete_action(self):
        rand_act = np.random.randint(0, self.action_space)
        return rand_act
    
    def set_o_map(self):
        self.obstacles.append(Obstacle(8, [25, 25]))
        self.obstacles.append(Obstacle(8, [25, 50]))
        self.obstacles.append(Obstacle(8, [25, 75]))
        self.obstacles.append(Obstacle(8, [75, 25]))
        self.obstacles.append(Obstacle(8, [75, 50]))
        self.obstacles.append(Obstacle(8, [75, 75]))

        for i in range(100):
            for j in range(100):
                for o in self.obstacles:
                    if o.check_collision([i, j]):
                        self.o_grid[i, j] = 1

        print(f"O Grid Complete")


class TestEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.eps += 1
        self.steps = 0
        self.memory = []



class Obstacle:
    def __init__(self, size, location=[0, 0]):
        self.size = size
        self.location = location

    def set_random_location(self):
        # range_bound = [10, 90]
        rands = np.random.rand(2) * 80 
        location = lib.add_locations(rands, [10, 10])
        self.location = location

    def check_collision(self, x):
        dis = lib.get_distance(self.location, x)
        if dis < self.size:
            return True
        return False




