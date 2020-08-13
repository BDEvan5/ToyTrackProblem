import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt
import sys
import collections
import random
import torch

from TestEnvWillemMod import CarModelDQN


class TrainEnvWillem(CarModelDQN):
    def __init__(self):
        self.map_dim = 100
        self.n_ranges = 10
        self.state_space = self.n_ranges + 2
        self.action_space = 9
        
        CarModelDQN.__init__(self, self.n_ranges)

        self.start_theta = 0
        self.start_velocity = 0
        self.th_start_end = None
        self.reward = np.zeros((2, 1))
        self.action = None
        self.done = None
        self.modified_action = None

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]
        self.step_size = int(1)
        self.n_searches = 30
        
        self.end = None
        self.start = None
        self.race_map = None

    def reset(self):
        self.race_map = np.zeros((100, 100))
        self._locate_obstacles()

        self.start = [np.random.random() * 60 + 20 , np.random.random() * 60 + 20]
        while self._check_location(self.start):
            self.start = [np.random.random() * 60 + 20 , np.random.random() * 60 + 20]
        self.car_x = self.start

        self.end = [np.random.random() * 60 + 20 , np.random.random() * 60 + 20]
        while self._check_location(self.end) or \
            lib.get_distance(self.start, self.end) < 20 or \
                lib.get_distance(self.start, self.end) > 60:
            self.end = [np.random.random() * 60 + 20 , np.random.random() * 60 + 20]

        grad = lib.get_gradient(self.start, self.end)
        dx = self.end[0] - self.start[0]
        th_start_end = np.arctan(grad)
        if th_start_end > 0:
            if dx > 0:
                th_start_end = np.pi / 2 - th_start_end
            else:
                th_start_end = -np.pi/2 - th_start_end
        else:
            if dx > 0:
                th_start_end = np.pi / 2 - th_start_end
            else:
                th_start_end = - np.pi/2 - th_start_end
        self.th_start_end = th_start_end
        self.start_theta = th_start_end + np.random.random() * np.pi/2 - np.pi/4
        self.theta = self.start_theta
        # self.theta = th_start_end
        # self.start_theta = self.theta


        # self.start_velocity = np.random.random() * self.max_velocity
        # self.velocity = self.start_velocity
        self.velocity = self.max_velocity

        self.steering = 0

        # text
        self.reward = np.zeros((2, 1))
        self.action = None
        self.pp_action = None

        

        return self._get_state_obs()

    def step(self, action):
        self.action = action # mod action
        th_mod = (action[0] - 2) * self.dth_action
        # x2 so that it can "look ahead further"
        modified_action = [self.lp_th + th_mod, self.lp_sp]

        self.pp_action = [self.lp_th *180/np.pi, self.lp_sp]
        self.modified_action = modified_action[0] * 180 / np.pi

        new_x = self._x_step(modified_action)
        crash = self._check_location(new_x)
        self.done = crash
        if not crash:
            self.update_state(modified_action)

        self.calculate_reward(crash, action)
        r = self.reward
        obs = self._get_state_obs()

        return obs, r, crash, None

    def calculate_reward(self, crash, action):
        if crash:
            self.reward = -1
            return 
        # self.reward = 1.0
        
        alpha = 0.1
        self.reward = 1 - alpha * abs(action[0] - 2)
      
    def _locate_obstacles(self):
        n_obs = 2
        xs = np.random.randint(30, 70, (n_obs, 1))
        ys = np.random.randint(20, 80, (n_obs, 1))
        obs_locs = np.concatenate([xs, ys], axis=1)
        obs_size = [10, 14]

        for obs in obs_locs:
            for i in range(obs_size[0]):
                for j in range(obs_size[1]):
                    x = i + obs[0]
                    y = j + obs[1]
                    self.race_map[x, y] = 1


        # wall boundaries
        obs_size = [20, 60]
        obs_locs = [[15, 0], [65, 40]]

        for obs in obs_locs:
            for i in range(obs_size[0]):
                for j in range(obs_size[1]):
                    x = i + obs[0]
                    y = j + obs[1]
                    self.race_map[x, y] = 1

    def _update_ranges(self):
        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            for j in range(self.n_searches): # number of search points
                fs = self.step_size * j
                dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
                search_val = lib.add_locations(self.car_x, dx)
                if self._check_location(search_val):
                    break             
            self.ranges[i] = (j) / (self.n_searches) # gives a scaled val to 1 

    def _get_state_obs(self):
        self.set_lp_action(self.end)
        self._update_ranges()

        lp_sp = self.lp_sp / self.max_velocity
        lp_th = self.lp_th / np.pi

        obs = np.concatenate([[lp_th], [lp_sp], self.ranges])

        return obs
        
    def _check_location(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if x[1] > self.y_bound[1]:
            return True 
        if x[1] < 0:
            # return False # represents the area below the zero line. Done for ranges
            return True

        if self.race_map[int(x[0]), int(x[1])]:
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

    def random_action(self):
        action = [np.random.randint(0, self.action_space-1)]
        return action

    def render(self, wait=False):
        car_x = int(self.car_x[0])
        car_y = int(self.car_x[1])
        fig = plt.figure(4)
        plt.clf()  
        plt.imshow(self.race_map.T, origin='lower')
        plt.xlim(0, self.map_dim)
        plt.ylim(-10, self.map_dim)
        plt.plot(self.start[0], self.start[1], '*', markersize=12)
        x_start_v = [self.start[0], self.start[0] + 15*np.sin(self.start_theta)]
        y_start_v = [self.start[1], self.start[1] + 15*np.cos(self.start_theta)]
        plt.plot(x_start_v, y_start_v, linewidth=2)
        plt.plot(self.end[0], self.end[1], '*', markersize=12)
        plt.plot(self.car_x[0], self.car_x[1], '+', markersize=16)

        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            fs = self.ranges[i] * self.n_searches * self.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations(self.car_x, dx)
            x = [car_x, range_val[0]]
            y = [car_y, range_val[1]]
            plt.plot(x, y)

        s = f"Action: {self.action}"
        plt.text(100, 70, s) 
        s = f"PP Action: [{self.pp_action[0]:.2f}, {self.pp_action[1]:.2f}]"
        plt.text(100, 60, s) 
        s = f"Mod action: {self.modified_action:.2f}"
        plt.text(100, 40, s) 
        s = f"Done: {self.done}"
        plt.text(100, 50, s) 
        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(100, 80, s)

        plt.pause(0.001)
        if wait:
            plt.show()

