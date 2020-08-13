import numpy as np 
import torch
from collections import deque
import random

from TestEnvWillemMod import TestEnvDQN
from matplotlib import pyplot as plt


MEMORY_SIZE = 100000

class ReplayBufferDQN():
    def __init__(self):
        self.buffer = deque(maxlen=MEMORY_SIZE)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def memory_sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # todo: move the tensor bit to the agent file, just return lists for the moment.
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)



"""Single Evals"""
def single_evaluation(agent, show_snap=True, show_render=False):
    env = TestEnvDQN()
    env.map_1000(True)
    # env.map_1010()
    # env.map_1020()

    score, done, state = 0, False, env.reset()
    while not done:
        a = agent.act(state)
        # a = [2]
        s_prime, _, done, _ = env.step(a)
        state = s_prime
        score += 1 # counts number of steps
        if show_render:
            # env.box_render()
            # env.full_render()
            pass
    if show_snap:
        env.render_snapshot()
        # if show_render:
        #     plt.show()
        
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
