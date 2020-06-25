import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt


class MakeEnv:
    def __init__(self):
        self.car_x = [0, 0]
        self.theta = 0
        self.last_distance = None
        self.start = None
        self.end = None
        self.obstacles = []

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]
        self.steps = 0
        self.eps = 0
        self.memory = []
        self.action_memory = []
        
        #parameters
        self.max_action = 1 # what the net gives
        self.state_dim = 2
        self.action_dim = 2
        self.action_scale = 5 # internal implementation

    def reset(self):
        self.eps += 1
        self.steps = 0
        self.memory = []
        self.action_memory = []
        self.reset_obstacles()
        rands = np.random.rand(4) * 100
        self.start = rands[0:2]
        self.end = rands[2:4]  

        while self._check_bounds(self.start):
            self.start = np.random.rand(2) * 100
        while self._check_bounds(self.end) or \
            lib.get_distance(self.start, self.end) < 15:
            self.end = np.random.rand(2) * 100

        self.theta = np.pi / 2 - np.arctan(lib.get_gradient(self.start, self.end))
        rel_target = lib.sub_locations(self.end, self.start)
        if rel_target[0] < 0:
            self.theta += np.pi

        self.last_distance = lib.get_distance(self.start, self.end)
        self.car_x = self.start

        return self._get_state_obs()

    def step(self, action):
        self.memory.append(self.car_x)
        self.steps += 1
        new_x = self._new_x(action)
        if self._check_bounds(new_x):
            obs = self._get_state_obs()
            r_crash = -100
            return obs, r_crash, True, None
        self.car_x = new_x 
        obs = self._get_state_obs()
        reward, done = self._get_reward()
        return obs, reward, done, None

    def step_continuous(self, action):
        self.memory.append(self.car_x)
        self.action_memory.append(action)
        self.steps += 1
        # new_x = lib.add_locations(self.car_x, action)
        new_x, new_theta = self.take_x_step(action)
        if self._check_bounds(new_x):
            obs = self._get_state_obs()
            r_crash = -100
            return obs, r_crash, True, None
        self.car_x = new_x 
        self.theta = new_theta
        obs = self._get_state_obs()
        reward, done = self._get_reward()
        return obs, reward, done, None

    def take_x_step(self, action):
        norm = 1
        transformed_action = lib.transform_coords(action, -self.theta)
        r = transformed_action[0] / transformed_action[1]
        y_sign = transformed_action[1] / abs(transformed_action[1]) # for the sign
        y = np.sqrt(norm**2/(1+r**2)) * y_sign
        x = r * y 

        scaled_action = [x*self.action_scale, y*self.action_scale]
        new_x = lib.add_locations(self.car_x, scaled_action)

        new_grad = lib.get_gradient(new_x, self.car_x)
        new_theta = np.pi / 2 - np.arctan(new_grad)

        return new_x, new_theta

    def random_action(self):
        rand = np.random.rand(2)
        a = (rand * 2 - [1, 1]) # this shifts the interval [0, 1) to
        act = a * self.max_action * 2 # [-1, 1]
        return act

    def _get_state_obs(self):
        rel_target = lib.sub_locations(self.end, self.car_x)
        transformed_target = lib.transform_coords(rel_target, self.theta)
        # obs = np.array(transformed_target) 

        return transformed_target

    def _get_reward(self):
        beta = 2
        r_done = 100

        cur_distance = lib.get_distance(self.car_x, self.end)
        if cur_distance < 1 + self.action_scale:
            return r_done, True
        d_dis = self.last_distance - cur_distance
        reward = beta * (d_dis**2 * d_dis/abs(d_dis)) - 5
        self.last_distance = cur_distance
        done = self._check_steps()
        return reward, done

    def _check_steps(self):
        if self.steps > 100:
            return True
        return False

    def _check_bounds(self, x):
        if self._check_obstacles(x):
            return True
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if self.y_bound[0] > x[1] or x[1] > self.y_bound[1]:
            return True 
        return False

    def _new_x(self, action):
        # action is 0, 1, 2, 3
        scale = 2
        if action == 0:
            dx = [0, 1] * scale
        elif action == 1:
            dx = [0, -1] * scale
        elif action == 2:
            dx = [1, 0] * scale
        elif action == 3:
            dx = [-1, 0] * scale
        x = lib.add_locations(dx, self.car_x)
        
        return x

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
        fig.savefig(f"Renders/Rendering_{self.eps}")

    def render_actions(self):
        xs, ys = [], []
        for pt in self.action_memory:
            xs.append(pt[0])
            ys.append(pt[1])
        
        fig = plt.figure(5)
        plt.clf()  
        plt.ylim(-1, 1)
        plt.plot(xs)
        plt.plot(ys)
        plt.pause(0.001)

    def add_obstacles(self, n=1):
        for i in range(n):
            o = Obstacle(10)
            self.obstacles.append(o)

    def _check_obstacles(self, x):
        for o in self.obstacles:
            if o.check_collision(x):
                return True

        return False

    def reset_obstacles(self):
        for o in self.obstacles:
            o.set_random_location()



class Obstacle:
    def __init__(self, size):
        self.size = size
        self.location = None

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

