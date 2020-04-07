import numpy as np 
from copy import deepcopy

class ClassicalAgent:
    def __init__(self, config, buffer, env):
        self.config = config
        self.buffer = buffer
        self.env = env

    def run_sim(self):
        next_obs, done, ep_reward = self.env.reset(), False, 0
        while not done:
            obs = deepcopy(next_obs)
            action = self.choose_action(obs)
            next_obs, reward, done = self.env.step(action)
            ep_reward += reward
            self.buffer.save_step((obs, action, reward, next_obs, done))
        return ep_reward

    def choose_action(self, state):
        # write function to choose logical action 
        # v = state[0]
        # th = state[1]
        # wp_x = state[2]
        # wp_y = state[3]
        # print(state)
        r = []
        for i in range(self.config.ranges_n):
            # range values
            r.append(state[0][4 + i])

        r_left = min([r[0] + r[1]])
        r_right = min([r[3] + r[4]])
        r_mid = (r[2])

        min_r = min([r_left, r_right])
        action = 1 # default to straight
        # if r_mid < 3 *min_r: # obstacle ahead
        #     if r_left > r_right:
        #         action = 0 # go left
        #     else:
        #         action = 2 #go right
        # if r_left < 2* r_right: 
        #     action = 2
        # if r_right < 2*r_left:
        #     action = 0

        return action




