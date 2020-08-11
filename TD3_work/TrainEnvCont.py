import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt
import sys
import collections
import random
import torch


class RewardFunctions:
    def __init__(self):
        self._get_reward = None

    def pure_mod(self):
        self._get_reward = self._get_mod_reward

    def switch(self):
        self._get_reward = self._get_switch_reward

    def pure_rep(self):
        self._get_reward = self._get_rep_reward

    def _get_mod_reward(self, crash, action):
        if crash:
            return -1, True

        pp_action = self._get_optimal_direction()
        action_dif = abs(action - pp_action)
        if action_dif == 0:
            r = 1
        else:
            r = 0.8 - action_dif * 0.1

        r = np.clip(r, 0.2, 1)

        return r, False

    def _get_rep_reward(self, crash, action):
        if crash:
            return -1, True

        action_angle = action[0] * 90 # degrees for debugg
        self.action = action_angle

        self.pp_action = self._get_optimal_direction() * 90 / (np.pi / 2)
        pp_action = pp_action_deg / 90 # range [-1, 1]

        d_action_angle = abs(pp_action - action[0])
        if d_action_angle < 0.01:
            reward = 1
        else:
            reward = - 1.5 * d_action_angle + 0.9

        reward = np.clip(reward, -0.9, 1)
        self.reward = reward

        return reward, False

    def _get_switch_reward(self, crash, action):
        if crash:
            return -1, True
        return 1, False

    def _get_optimal_direction(self):
        move_angle = self.th_start_end - self.start_theta 

        return move_angle

    def _get_optimal_velocity(self, optimal_heading):
        # heading in range [-1, 1]
        vel = 1 - abs(optimal_heading) 

        return vel


class TrainEnvCont():
    def __init__(self):
        self.map_dim = 100
        self.n_ranges = 10
        self.state_space = self.n_ranges + 2
        self.action_space = 10
        self.action_scale = self.map_dim / 20

        self.action_dim = 1

        self.car_x = None
        self.theta = None
        self.th_start_end = None
        self.start_theta = None

        self.reward = None
        self.pp_action_deg = 0
        self.action = [0, 0]

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

        self.step_size = int(1)
        self.n_searches = 30
        
        self.target = None
        self.end = None
        self.start = [50, 0]
        self.race_map = np.zeros((100, 100))

        self.ranges = np.zeros(self.n_ranges)
        self.range_angles = np.zeros(self.n_ranges)
        dth = np.pi/(self.n_ranges-1)
        for i in range(self.n_ranges):
            self.range_angles[i] = i * dth - np.pi/2

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

        # text
        self.reward = None
        self.action = None
        self.pp_action = None

        return self._get_state_obs()

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
        self._update_ranges()
        rel_target = lib.sub_locations(self.end, self.car_x)
        transformed_target = lib.transform_coords(rel_target, self.theta)
        normalised_target = lib.normalise_coords(transformed_target)
        normalised_target = np.asarray(normalised_target)
        normalised_target = np.reshape(normalised_target, (2))
        obs = np.concatenate([normalised_target, self.ranges])

        return obs

    def step(self, action):
        new_x, new_theta = self._x_step_discrete(action[0])
        crash = self._check_line(self.car_x, new_x)
        if not crash:
            self.car_x = new_x
            self.theta = new_theta
        r, done = self._get_reward(crash, action)

        obs = self._get_state_obs()

        return obs, r, done, None

    def super_step(self):
        direction = self._get_optimal_direction()
        velocity = self._get_optimal_velocity(direction)

        return [direction, velocity]

    def _x_step_discrete(self, action):
        action_angle = action[0] * np.pi / 2 # range [-1, 1] to [-pi/2, pi/2]
        # actions in range [0, n_acts) are a fan in front of vehicle
        # no backwards
        fs = self.action_scale
        action_angle += self.theta # for the vehicle offset
        dx = [np.sin(action_angle)*fs, np.cos(action_angle)*fs] 
        
        new_x = lib.add_locations(dx, self.car_x)
        
        new_grad = lib.get_gradient(new_x, self.car_x)
        new_theta = np.pi / 2 - np.arctan(new_grad)
        if dx[0] < 0:
            new_theta += np.pi
        if new_theta >= 2*np.pi:
            new_theta = new_theta - 2*np.pi

        return new_x, new_theta

    def _check_location(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if x[1] > self.y_bound[1]:
            return True 
        if x[1] < 0:
            return False # represents the area below the zero line. Done for ranges

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
        return [np.random.random() * 2 - 1] #range [-1, 1]

    def render(self):
        car_x = int(self.car_x[0])
        car_y = int(self.car_x[1])
        fig = plt.figure(4)
        plt.clf()  
        plt.imshow(self.race_map.T, origin='lower')
        plt.xlim(0, self.map_dim)
        plt.ylim(-10, self.map_dim)
        plt.plot(self.start[0], self.start[1], '*', markersize=12)
        plt.plot(self.end[0], self.end[1], '*', markersize=12)
        plt.plot(self.car_x[0], self.car_x[1], '+', markersize=16)
        x_start_v = [self.start[0], self.start[0] + 15*np.sin(self.start_theta)]
        y_start_v = [self.start[1], self.start[1] + 15*np.cos(self.start_theta)]
        plt.plot(x_start_v, y_start_v, linewidth=2)

        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            fs = self.ranges[i] * self.n_searches * self.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations(self.car_x, dx)
            x = [car_x, range_val[0]]
            y = [car_y, range_val[1]]
            plt.plot(x, y)

        s = f"Reward: [{self.reward:.2f}]" 
        plt.text(100, 80, s)
        s = f"Action: {self.action:.2f}"
        plt.text(100, 70, s) 
        s = f"PP Action: {self.pp_action:.2f}"
        plt.text(100, 60, s) 
        
        plt.pause(0.001)

    def _get_optimal_direction(self):
        move_angle = self.th_start_end - self.start_theta 

        return move_angle

    def _get_optimal_velocity(self, optimal_heading):
        # heading in range [-1, 1]
        vel = 1 - abs(optimal_heading) 

        return vel
