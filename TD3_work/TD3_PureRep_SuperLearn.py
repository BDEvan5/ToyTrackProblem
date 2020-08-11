import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import sys
import pickle
import matplotlib.pyplot as plt 

import LibFunctions as lib 
from TrainEnvCont import TrainEnvCont
from CommonTestUtilsTD3 import single_evaluationCont


# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2


class SuperLearnBufferTD3(object):
    def __init__(self, max_size=100000):     
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, right_actions = [], []

        for i in ind: 
            s, a_r = self.storage[i]
            states.append(np.array(s, copy=False))
            right_actions.append(np.array(a_r, copy=False))

        return np.array(states), np.array(right_actions)

        
class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action=1):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        y = F.relu(self.l2(x))
        z = self.l3(y)
        a = self.max_action * torch.tanh(z) 
        return a



class SuperTrainRep(object):
    def __init__(self, state_dim, action_dim, agent_name):
        self.model = None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_name = agent_name

    def train(self, replay_buffer):
        s, a_r = replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(s)
        right_actions = torch.FloatTensor(a_r)

        actions = self.model(states)

        actor_loss = F.mse_loss(actions, right_actions)

        self.model.optimizer.zero_grad()
        actor_loss.backward()
        self.model.optimizer.step()

        return actor_loss

    def save(self, directory='./td3_saves'):
        torch.save(self.model, '%s/%s_model.pth' % (directory, self.agent_name))

    def load(self, directory='./td3_saves'):
        self.model = torch.load('%s/%s_model.pth' % (directory, self.agent_name))

    def create_agent(self):
        self.model = Actor(self.state_dim, self.action_dim, 1)

    def try_load(self, load=True):
        if load:
            try:
                self.load()
            except:
                print(f"Unable to load model")
                pass
        else:
            self.create_agent()
            print(f"Not loading - restarting training")

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.model(state).data.numpy().flatten()

        return action


def build_data_set(observation_steps=1000):
    buffer = SuperLearnBufferTD3()

    env = TrainEnvCont()

    for i in range(observation_steps):
        s = env.reset()
        a_r = env.super_step()
        buffer.add((s, a_r))

        print("\rPopulating Buffer {}/{}.".format(i, observation_steps), end="")
        sys.stdout.flush()
    print(" ")

    return buffer

def RunSuperLearn(agent_name, load=True):
    buffer = build_data_set(100)

    state_dim = 14
    action_dim = 2
    agent = SuperTrainRep(state_dim, action_dim, agent_name)
    agent.try_load(load)

    training_loops = 1000
    show_n = 100
    for i in range(training_loops):
        agent.train(buffer)

        if i % show_n == 1:
            agent.save()
            single_evaluationCont(agent, True)

    agent.save()
    single_evaluationCont(agent, True)


if __name__ == "__main__":
    agent_name = "testing_super"

    RunSuperLearn(agent_name, False)
    # RunSuperLearn(agent_name, True)


