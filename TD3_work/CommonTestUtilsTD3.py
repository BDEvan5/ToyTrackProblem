import numpy as np 
import torch
from collections import deque
import random
from TestEnvCont import TestEnvCont
from matplotlib import pyplot as plt



class ReplayBufferTD3(object):
    def __init__(self, max_size=100000):     
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.filename = "DataRecords/buffer"

    def add(self, data):        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind: 
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)


"""Single Evals"""
def single_evaluationCont(agent, show_snap=True, show_render=False):
    env = TestEnvCont()
    env.map_1000(True)
    # env.map_1010()
    # env.map_1020()

    score, done, state = 0, False, env.reset()
    while not done:
        a = agent.act(state)
        s_prime, _, done, _ = env.step(a)
        state = s_prime
        score += 1 # counts number of steps
        if show_render:
            # env.box_render()
            # env.full_render()
            pass
    if show_snap:
        env.render_snapshot()
        if show_render:
            plt.show()
        
    print(f"SingleRun --> Score: {score} --> Steps: {env.steps}")
    return score


class CorridorAgent:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.name = "Corridor"

    def act(self, obs):
        # all this does is go in the direction of the biggest range finder
        ranges = obs[2:]
        action = np.argmax(ranges)

        return action

    def load(self):
        pass



class PurePursuit:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.name = "PurePursuit"

    def load(self):
        pass

    def act(self, obs):
        try:
            grad = obs[1] / obs[0] # y/x
        except:
            grad = 10000
        angle = np.arctan(grad)
        if angle > 0:
            angle = np.pi - angle
        else:
            angle = - angle
        dth = np.pi / (self.act_space - 1)
        action = round(angle / dth)

        return action

        
class TestOptimalSolution:
    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.act_space = action_space

    def act(self, obs):
        pp_action = self._get_pp_action(obs)
        pp_velocity = self._get_pp_velocity(pp_action) * 10

        return [pp_action, pp_velocity]
        
    def _get_pp_action(self, obs):
        try:
            grad = obs[1] / obs[0] # y/x
        except:
            grad = 10000
        angle = np.arctan(grad)

        if angle > 0:
            angle = np.pi - angle
        else:
            angle = - angle
        dth = np.pi / (self.act_space - 1)
        pp_action = round(angle / dth)

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
