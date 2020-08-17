import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt
import sys
import collections
import random
import torch

from TestEnvPureRep import CarModelDQN

pi2 = np.pi / 2

class TrainEnvPureRep(CarModelDQN):
    def __init__(self):
        self.map_dim = 100
        
        CarModelDQN.__init__(self)

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

        self.memory = []
        self.steps = 0

    def reset(self):
        self.memory.clear()
        self.steps = 0
        self.race_map = np.zeros((100, 100))
        self._locate_obstacles()
        
        self.start = lib.get_rands()
        while self._check_location(self.start):
            self.start = lib.get_rands()
        self.car_x = self.start

        self.end = lib.get_rands()
        while self._check_location(self.end) or \
            lib.get_distance(self.start, self.end) < 30:
            self.end = lib.get_rands()
        self.end = lib.get_rands(80, 10)


        th_start_end = lib.get_bearing(self.start, self.end)
        self.th_start_end = th_start_end
        self.start_theta = th_start_end + np.random.random() * np.pi/2 - np.pi/4
        self.theta = self.start_theta
        # self.theta = th_start_end
        # self.start_theta = self.theta

        self.memory.append(self.car_x)


        # self.start_velocity = np.random.random() * self.max_velocity
        # self.velocity = self.start_velocity
        self.velocity = self.max_velocity

        self.steering = 0

        # text
        self.reward = None
        self.action = None
        self.pp_action = np.zeros((2))

        
        return self._get_state_obs(self.end)

    

    def step(self, action):
        self.steps += 1
        self.action = action # mod action

        new_x = self._x_step(action)
        crash = self._check_line(new_x, self.car_x)
        self.done = crash
        if not crash:
            self.update_state(action)

        self.calculate_reward(crash, action)
        r = self.reward
        obs = self._get_state_obs(self.end)

        if lib.get_distance(self.car_x, self.end) < 10:
            self.done = True

        self.memory.append(self.car_x)

        return obs, r, self.done, None

    def calculate_reward(self, crash, action):
        if crash:
            self.reward = -1
            return 
        
        lp_a = self.lp_th * self.action_space / np.pi
        action_dif = action[0] - lp_a
        self.reward = 1 - action_dif * 0.2
      
    def _locate_obstacles(self):
        n_obs = 2
        # xs = np.random.randint(30, 62, (n_obs, 1))
        # ys = np.ones_like(xs) * 22
        # ys = np.random.randint(20, 80, (n_obs, 1))
        # obs_locs = np.concatenate([xs, ys], axis=1)
        # obs_locs = [[45, 40], [25, 68], [35, 20], [53, 70], [70, 10]]
        # obs_locs = [[35, 22], [55, 24]]
        # obs_locs = [[int(self.start[0]) - 3, 24]]

        obs_locs = [[40, 75], [60, 50], [50, 30], [10, 80], [20, 10]]
        obs_size = [8, 14]

        for obs in obs_locs:
            for i in range(obs_size[0]):
                for j in range(obs_size[1]):
                    x = i + obs[0]
                    y = j + obs[1]
                    self.race_map[x, y] = 2

        obs_locs = [[20, 60], [60, 10], [20, 40], [70, 80], [70, 30]]
        obs_size = [14, 5]

        for obs in obs_locs:
            for i in range(obs_size[0]):
                for j in range(obs_size[1]):
                    x = i + obs[0]
                    y = j + obs[1]
                    self.race_map[x, y] = 2
        # 
        # obs_locs = [[46, 23]]
        # obs_size = [8, 10]

        # for obs in obs_locs:
        #     for i in range(obs_size[0]):
        #         for j in range(obs_size[1]):
        #             x = i + obs[0]
        #             y = j + obs[1]
        #             self.race_map[x, y] = 2

             
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
        # s = f"PP Action: [{self.pp_action[0]:.2f}, {self.pp_action[1]:.2f}]"
        # plt.text(100, 60, s) 
        # s = f"Mod action: {self.modified_action:.2f}"
        # plt.text(100, 40, s) 
        s = f"Done: {self.done}"
        plt.text(100, 50, s) 
        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(100, 80, s)

        plt.pause(0.001)
        if wait:
            plt.show()

    def render_snapshot(self):
        fig = plt.figure(9)
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

        plt.plot(self.start[0], self.start[1], '*', markersize=12)
        x_start_v = [self.start[0], self.start[0] + 15*np.sin(self.start_theta)]
        y_start_v = [self.start[1], self.start[1] + 15*np.cos(self.start_theta)]
        plt.plot(x_start_v, y_start_v, linewidth=2)
        plt.plot(self.end[0], self.end[1], '*', markersize=12)
        plt.plot(self.car_x[0], self.car_x[1], '+', markersize=16)
        

        s = f"Steps: {self.steps}"
        plt.text(100, 80, s)
        s = f"Action: {self.action}"
        plt.text(100, 70, s) 
        # s = f"PP Action: [{self.pp_action[0]:.2f}, {self.pp_action[1]:.2f}]"
        # plt.text(100, 60, s) 
        # s = f"Mod action: {self.modified_action:.2f}"
        # plt.text(100, 40, s) 
        s = f"Done: {self.done}"
        plt.text(100, 50, s) 
        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(100, 80, s)

        plt.pause(0.001)
