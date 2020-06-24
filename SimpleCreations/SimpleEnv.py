import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt


class MakeEnv:
    def __init__(self):
        self.car_x = [0, 0]
        self.last_distance = None
        self.start = None
        self.end = None

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]
        self.steps = 0
        self.memory = []
        
        #parameters
        self.max_action = 1
        self.state_dim = 2
        self.action_dim = 2

    def reset(self):
        self.steps = 0
        self.memory = []
        rands = np.random.rand(4) * 100
        self.start = rands[0:2]
        self.end = rands[2:4]  

        while self._check_bounds(self.start):
            self.start = np.random.rand(2) * 100
        while self._check_bounds(self.end) or \
            lib.get_distance(self.start, self.end) < 15:
            self.end = np.random.rand(2) * 100

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
        self.steps += 1
        # new_x = lib.add_locations(self.car_x, action)
        new_x = self.take_x_step(action)
        if self._check_bounds(new_x):
            obs = self._get_state_obs()
            r_crash = -100
            return obs, r_crash, True, None
        self.car_x = new_x 
        obs = self._get_state_obs()
        reward, done = self._get_reward()
        return obs, reward, done, None

    def take_x_step(self, action):
        # action comes in rand [-1, 1] for two dimensions
        # I want to normalise the action so r == 1 and then take step in that direction
        # doing this, the action becomes purely a position vector
        norm = 1
        r = action[0] / action[1]
        y = np.sqrt(norm**2/(1+r**2)) * action[1] / abs(action[1]) # for the sign
        x = r * y 

        new_x = lib.add_locations(self.car_x, [x*5, y*5])
        return new_x

    def random_action(self):
        a = np.random.rand(2) * self.max_action
        return a

    def _get_state_obs(self):
        # scale = 100
        # distance = lib.get_distance(self.end, self.car_x)
        # theta = np.tan(lib.get_gradient(self.end, self.car_x)**-1) + np.pi 
        rel_target = lib.sub_locations(self.end, self.car_x)
        obs = np.array(rel_target) 
        # obs = [self.end[0], self.end[1], self.car_x[0], self.car_x[1]]
        # obs = np.array(obs)
        
        # current obs is in terms of an x, y target
        return obs

    def _get_reward(self):
        beta = 2
        r_done = 100

        cur_distance = lib.get_distance(self.car_x, self.end)
        if cur_distance < 5:
            return r_done, True
        reward = beta * (self.last_distance - cur_distance)
        self.last_distance = cur_distance
        done = self._check_steps()
        return reward, done

    def _check_steps(self):
        if self.steps > 100:
            return True
        return False

    def _check_bounds(self, x):
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
        plt.plot(self.start[0], self.start[1], '*')
        plt.plot(self.end[0], self.end[1], '*')
        plt.pause(0.001)
        
