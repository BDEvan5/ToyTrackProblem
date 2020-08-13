import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt
import sys
import collections
import random
import torch

from TestEnvDQN import CarModelDQN
from CommonTestUtilsDQN import PurePursuit


class RewardFunctions:
    def __init__(self):
        self._get_reward = None
        self.pp_action = None

    def pure_mod(self):
        self._get_reward = self._get_mod_reward

    def switch(self):
        self._get_reward = self._get_switch_reward

    def pure_rep(self):
        self._get_reward = self._get_rep_reward

    def _get_mod_reward(self, crash, action):
        if crash:
            return -1, True

        pp_action = self._get_pp_action()
        action_dif = abs(action - pp_action)
        if action_dif == 0:
            r = 1
        else:
            r = 0.8 - action_dif * 0.1

        r = np.clip(r, 0.2, 1)

        return r, False

    def _get_rep_reward(self, crash, action):
        if crash:
            reward = np.array([-1, -1])[:, None]
            return reward, True

        pp_action = self._get_pp_action()
        self.pp_action = pp_action

        angle = action[0]
        velocity = action[1]

        d_action = abs(pp_action - angle)
        if d_action == 0:
            angle_reward = 1
        else:
            angle_reward = - 0.3 * d_action + 0.9

        angle_reward = np.clip(angle_reward, -0.9, 1)

        pp_velocity = self._get_pp_velocity(pp_action)
        velocity_reward = 1 - 3 * abs(velocity * 0.1 - pp_velocity)
        velocity_reward = np.clip(velocity_reward, -0.9, 1)

        reward = np.array([angle_reward, velocity_reward])[:, None]

        return reward, False

    def _get_switch_reward(self, crash, action):
        if crash:
            return -1, True
        return 1, False

    def _get_pp_action(self):
        move_angle = self.th_start_end - self.start_theta + np.pi / 2
        pp_action = round(move_angle / self.dth)

        return pp_action

    def _get_pp_velocity(self, pp_action):
        if pp_action == 4 or pp_action == 5:
            return 0.9
        if pp_action == 3 or pp_action == 6:
            return 0.8
        if pp_action == 2 or pp_action == 7:
            return 0.6
        if pp_action == 1 or pp_action == 8:
            return 0.5
        if pp_action == 0 or pp_action == 9:
            return 0.4



class TrainEnvDQN(RewardFunctions, CarModelDQN):
    def __init__(self):
        self.map_dim = 100
        self.n_ranges = 10
        self.state_space = self.n_ranges + 4
        self.action_space = 10
        self.action_scale = self.map_dim / 20
        self.dth = np.pi / (self.action_space - 1)

        RewardFunctions.__init__(self)
        CarModelDQN.__init__(self, self.n_ranges)

        self.start_theta = 0
        self.start_velocity = 0
        self.th_start_end = None

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]
        self.reward = np.zeros((2, 1))
        self.action = None

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

        self.start_velocity = np.random.random() * self.max_velocity
        self.velocity = self.start_velocity

        self.steering = 0

        # text
        self.reward = np.zeros((2, 1))
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
        rel_v = self.velocity / self.max_velocity
        rel_th = self.theta / np.pi

        self._update_ranges()
        rel_target = lib.sub_locations(self.end, self.car_x)
        transformed_target = lib.transform_coords(rel_target, self.theta)
        normalised_target = lib.normalise_coords(transformed_target)
        obs = np.concatenate([normalised_target, [rel_v], [rel_th], self.ranges])

        return obs

    def step(self, action):
        self.action = action
        new_x, new_theta, new_v, new_w = self._x_step_discrete(action)
        crash = self._check_crash(new_x, new_v, action[0])
        if not crash:
            self.car_x = new_x
            self.theta = new_theta
            self.velocity = new_v
            self.steering = new_w
        r, done = self._get_reward(crash, action)
        self.reward = r

        obs = self._get_state_obs()

        return obs, r, done, None

    def _check_crash(self, new_x, new_v, angle_action):
        new_v = new_v / self.max_velocity
        if self._check_line(self.car_x, new_x):
            return True
        if (angle_action == 3 or angle_action == 6) and new_v >= 0.9:
            return True
        if (angle_action == 2 or angle_action == 7) and new_v >= 0.7:
            return True
        if (angle_action == 1 or angle_action == 8) and new_v >= 0.6:
            return True
        if (angle_action == 0 or angle_action == 9) and new_v >= 0.5:
            return True

        return False
        
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
        action = [np.random.randint(0, self.action_space-1), np.random.randint(0, self.action_space-1)]
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

        s = f"Reward: [{self.reward[0, 0]:.1f}, {self.reward[1, 0]:.1f}]" 
        plt.text(100, 80, s)
        s = f"Action: {self.action}"
        plt.text(100, 70, s) 
        s = f"PP Action: {self.pp_action}"
        plt.text(100, 60, s) 

        plt.pause(0.001)
        if wait:
            plt.show()

