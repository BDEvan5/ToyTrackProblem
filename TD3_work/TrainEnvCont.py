import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt
import sys
import collections
import random
import torch

from TestEnvCont import CarModelCont
from PathFinder import PathFinder

class TrainEnvCont(CarModelCont):
    def __init__(self):
        self.map_dim = 100
        self.n_ranges = 10
        self.state_space = self.n_ranges + 2
        self.action_space = 10
        self.action_scale = self.map_dim / 20

        self.action_dim = 1

        CarModelCont.__init__(self, self.n_ranges)

        self.th_start_end = None
        self.start_theta = None
        self.start_velocity = 0

        self.reward = None
        self.pp_action_deg = 0
        self.action = [0, 0]

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

        self.step_size = int(1)
        self.n_searches = 30
        
        self.target = None
        self.end = None
        self.start = None
        self.path = []
        self.race_map = np.zeros((100, 100))

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

        # self.steering = 0

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

        rel_v = self.velocity / self.max_velocity
        rel_th = self.theta / np.pi

        rel_target = lib.sub_locations(self.end, self.car_x)
        transformed_target = lib.transform_coords(rel_target, self.theta)
        normalised_target = lib.normalise_coords(transformed_target)
        normalised_target = np.asarray(normalised_target)
        normalised_target = np.reshape(normalised_target, (2))
        obs = np.concatenate([normalised_target, [rel_v], [rel_th], self.ranges])

        return obs

    def step(self, action):
        new_x, new_theta, new_v = self._x_step(action)
        crash = self._check_line(self.car_x, new_x)
        if not crash:
            self.car_x = new_x
            self.theta = new_theta
            self.velocity = new_v
        r, done = self._get_reward(crash, action)

        obs = self._get_state_obs()

        return obs, r, done, None

    def super_step(self):
        direction = self._get_optimal_direction()
        velocity = self._get_optimal_velocity(direction)

        return [direction, velocity]

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
        
        # WB = 0.5
        # delta = np.arctan(move_angle * WB / self.velocity)

        return move_angle

    def _get_optimal_velocity(self, optimal_heading):
        # heading in range [-1, 1]
        vel = 1 - abs(optimal_heading) 

        return vel

    def plan_optimal_path(self):
        # self.show_path()
        path_finder = PathFinder(self._check_location, self.start, self.end)
        self.path = path_finder.run_search(5)

        self.show_path()

        self.optimise_path()

    def optimise_path(self):
        Q = self.path

        alpha_init = 0.1
        alpha = alpha_init
        minReached = False
        noCollision = self.compute_path_collision(Q)

        while not (noCollision and minReached):
            P = self.compute_optimal_step(Q)
            if alpha == 1 or np.linalg.norm(P) < 1e-2:
                minReached = True
            # Q_next = Q + alpha * P
            Q_next = self.get_q_next(Q, alpha, P)
            if not self.compute_path_collision(Q_next):
                noCollision = False
                if alpha != 1:
                    print(f"Adding constraint")
                    self.compute_collision_constraint(Q, Q_next)
                    # compute collision constraint
                    # find new constarint
                    # add new constraint
                    alpha = 1
                else:
                    alpha = alpha_init
            else:
                Q = Q_next
                noCollision = True

        self.path = Q

    def compute_optimal_step(self, Q):
        d_cs = []
        for k in range(len(Q)-2):
            # d_c = (Q[k+1] - Q[k]) - (Q[k+2] - Q[k+1])
            d_1 = lib.sub_locations(Q[k+1], Q[k])
            d_2 = lib.sub_locations(Q[k+2], Q[k+1])
            d_c = lib.sub_locations(d_1, d_2)
            d_cs.append(d_c)

        return d_cs
            
    def get_q_next(self, Q, alpha, P):
        Q_next = []
        Q_next.append(Q[0]) # start doesn't change
        for k in range(len(Q) - 2):
            # q_k = Q[k+1] + alpha * P[k]
            q_k = lib.add_locations(Q[k+1], P[k], alpha)
            Q_next.append(q_k)
        Q_next.append(Q[-1]) # unchanged end point

        return Q_next

    def compute_path_collision(self, Q):
        for i in range(len(Q) - 1):
            if self._check_line(Q[i], Q[i+1]):
                return True
        return False

    # def compute_collision_constraint(self, Q, Q_next):
        #leave this out for the moment
        # it is used to get x_free and x_col

    def find_new_constraint(self, x_free, x_coll, p, alpha):
        # x is used for q
        solved = False
        while not solved:
            alpha = 0.5 * alpha
            x = self.get_q_next(x_free, alpha, p)
            if not self.compute_path_collision(x):
                x_free = x
            else:
                x_coll = x
            


    def show_path(self):
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

        # write code here to show the path, for pt i self.path
        for pt in self.path:
            plt.plot(pt[0], pt[1], 'x', markersize=10)

        plt.show()





def path_optimiser_driver():
    env = TrainEnvCont()
    s = env.reset()
    env.plan_optimal_path()
    env.show_path()



if __name__ == "__main__":
    path_optimiser_driver()
