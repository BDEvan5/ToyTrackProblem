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
    def __init__(self, state_dim, action_dim, max_action):
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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TrainRepTD3(object):
    def __init__(self, state_dim, action_dim, max_action, agent_name):
        self.agent_name = agent_name
        self.actor = None
        self.actor_target = None

        self.critic = None
        self.critic_target = None

        self.max_action = max_action
        self.act_dim = action_dim
        self.state_dim = state_dim
        self.last_action = None

        self.train_counter = 0

    def act(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.max_action, self.max_action)

    def train(self, replay_buffer, iterations=2):
        # iterations = 1 # number of times to train per step
        for it in range(iterations):
            # Sample replay buffer 
            s, a_r = replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(s)
            right_action = torch.FloatTensor(a_r)

            # self.update_critic(state, right_action)

            # Delayed policy updates
            if it % POLICY_FREQUENCY == 0:
                self.update_actor(state, right_action)

    def update_critic(self, state, action, u, reward, done, next_state):
        # Select action according to policy and add clipped noise 
        noise = torch.FloatTensor(u).data.normal_(0, POLICY_NOISE)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * GAMMA * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

    def update_actor(self, state):
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, directory="./td3_saves"):
        filename = self.agent_name

        torch.save(self.actor, '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_target, '%s/%s_actor_target.pth' % (directory, filename))
        torch.save(self.critic, '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_target, '%s/%s_critic_target.pth' % (directory, filename))

    def load(self, directory="./td3_saves"):
        filename = self.agent_name

        self.actor = torch.load('%s/%s_actor.pth' % (directory, filename))
        self.actor_target = torch.load('%s/%s_actor_target.pth' % (directory, filename))
        self.critic = torch.load('%s/%s_critic.pth' % (directory, filename))
        self.critic_target = torch.load('%s/%s_critic_target.pth' % (directory, filename))

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

    def create_agent(self):
        state_dim = self.state_dim
        action_dim = self.act_dim
        max_action = self.max_action

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())


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


class TestRepTD3(object):
    def __init__(self, state_dim, action_dim, max_action, agent_name):
        self.agent_name = agent_name
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.act_dim = action_dim
        self.last_action = None
        self.filename = "DataRecords/buffer"

    def act(self, state, noise=0.0):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.max_action, self.max_action)

    def load(self, directory="./td3_saves"):
        filename = self.agent_name

        self.actor = torch.load('%s/%s_actor.pth' % (directory, filename))
        self.actor_target = torch.load('%s/%s_actor_target.pth' % (directory, filename))
        self.critic = torch.load('%s/%s_critic.pth' % (directory, filename))
        self.critic_target = torch.load('%s/%s_critic_target.pth' % (directory, filename))


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


    state_dim = 12
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


