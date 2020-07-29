import gym
import collections
import random
import sys
import numpy as np 
from matplotlib import pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import LibFunctions as lib
from TrainEnv_Switch import TrainEnv_Switch
from CommonTestUtils import ReplayBuffer, single_rep_eval, PurePursuit


#Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
h_size = 512
BATCH_SIZE = 64


EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

"""Switch part"""
class ValueNet(nn.Module):
    def __init__(self, obs_space):
        super(ValueNet, self).__init__()
        h_vals = int(h_size / 2)
        self.fc1 = nn.Linear(obs_space + 1, h_vals)
        self.fc2 = nn.Linear(h_vals, h_vals)
        self.fc3 = nn.Linear(h_vals, 1)
        self.exploration_rate = EXPLORATION_MAX
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DecisionDQN:
    def __init__(self, obs_space, action_space, name):
        self.value_model = None
        self.value_target = None

        self.action_space = action_space
        self.obs_space = obs_space

        self.update_steps = 0
        self.name = name

    def decide(self, obs, pp_action):
        value_obs = np.append(obs, pp_action)
        value_obs_t = torch.from_numpy(value_obs).float()
        safe_value = self.value_model.forward(value_obs_t)

        return safe_value.detach().item() 

    def train_switching(self, buffer0):
        n_train = 1
        # train system0: switching system
        for i in range(n_train):
            if buffer0.size() < BATCH_SIZE:
                return 0
            s, a_sys, r, s_p, done = buffer0.memory_sample(BATCH_SIZE)

            q_update = r.float() # always done

            pp_act = self.get_pp_acts(s.numpy())
            cat_act = torch.from_numpy(pp_act[:, None]).float()
            obs = torch.cat([s, cat_act], dim=1)
            q_vals = self.value_model.forward(obs)

            loss = F.mse_loss(q_vals, q_update.detach())

            self.value_model.optimizer.zero_grad()
            loss.backward()
            self.value_model.optimizer.step()

        self.update_networks()
        l = loss.item()

        return l

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.value_target.load_state_dict(self.value_model.state_dict())
        if self.update_steps % 12 == 1:
            self.value_model.exploration_rate *= EXPLORATION_DECAY 
            self.value_model.exploration_rate = max(EXPLORATION_MIN, self.value_model.exploration_rate)

    def save(self, directory="./dqn_saves"):
        filename = self.name
        torch.save(self.value_model, '%s/%s_Vmodel.pth' % (directory, filename))
        torch.save(self.value_target, '%s/%s_Vtarget.pth' % (directory, filename))
        print(f"Saved Agent: {self.name}")

    def load(self, directory="./dqn_saves"):
        filename = self.name
        self.value_model = torch.load('%s/%s_Vmodel.pth' % (directory, filename))
        self.value_target = torch.load('%s/%s_Vtarget.pth' % (directory, filename))
        print(f"Loaded Agent: {self.name}")

    def try_load(self, load=True):
        if load:
            try:
                self.load()
            except:
                print(f"Unable to load model")
                self.create_model()
        else:
            self.create_model()

    def create_model(self):
        self.value_model = ValueNet(self.obs_space)
        self.value_target = ValueNet(self.obs_space)
        print(f"Creating new model")

    def act(self, obs):
        pp_action = self.get_pursuit_action(obs)
        safe_value = self.decide(obs, pp_action)
        
        return pp_action # this is the action to be taken

    def get_pursuit_action(self, obs):
        if abs(obs[0]) > 0.01:
            grad = obs[1] / obs[0] # y/x
        else:
            grad = 10000
        angle = np.arctan(grad)
        if angle > 0:
            angle = np.pi - angle
        else:
            angle = - angle
        dth = np.pi / (self.action_space - 1)
        action = int(angle / dth)

        return action

    def get_pp_acts(self, obses):
        arr = np.zeros((len(obses)))
        for i, obs in enumerate(obses):
            arr[i] = self.get_pursuit_action(obs)

        return arr


"""Collect obs"""
def collect_mod_observations(buffer, n_itterations=10000):
    env = TrainEnv_Switch()
    pp = PurePursuit(env.state_space, env.action_space)
    s, done = env.reset(), False

    for i in range(n_itterations):
        state = env.reset()
        a = pp.act(state)
        s_prime, r, done, _ = env.step(a)
        buffer.put((state, a, r, s_prime, done))

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()
    print(" ")


"""Training loops"""
def TrainDecAgent(agent_name, buffer, i=0, load=True):
    env = BasicTrainModEnv()
    agent = DecisionDQN(env.state_space, env.action_space, agent_name)
    agent.try_load(load)

    print_n = 100
    rewards, score = [], 0.0
    for n in range(1000):
        state = env.reset()
        a = agent.act(state)
        s_prime, r, done, _ = env.step(a)
        buffer.put((state, a, r, s_prime, done)) 
        loss = agent.train_switching(buffer)
        score += loss

        if n % print_n == 0 and n > 0:
            rewards.append(score)
            env.render()    
            exp = agent.value_model.exploration_rate
            mean = np.mean(rewards[-20:])
            b0 = buffer.size()
            print(f"Run: {n} --> Score: {score:.4f} --> Mean: {mean:.4f} --> exp: {exp:.4f} --> Buf: {b0}")
            score = 0
            lib.plot(rewards, figure_n=2)

    agent.save()

    return rewards

# supervised learning approach
def SuperLearn(agent_name, buffer, load=True):
    agent = DecisionDQN(12, 10, agent_name)
    agent.try_load(load)
    print_n = 100
    rewards, score = [], 0.0
    for n in range(2000):
        loss = agent.train_switching(buffer)
        score += loss

        if n % print_n == 0 and n > 0:
            rewards.append(score)
            mean = np.mean(rewards[-20:])
            print(f"Run: {n} --> Score: {score:.4f} --> Mean: {mean:.4f}")
            score = 0.0
            lib.plot(rewards, figure_n=2)

    agent.save()

    return rewards

def RunSuperLearn(agent_name, create=True):
    buffer = ReplayBuffer()
    total_rewards = []

    collect_mod_observations(buffer, 2000)
    if create:
        r = SuperLearn(agent_name, buffer, False)
        total_rewards += r
        lib.plot(total_rewards)

    for i in range(100):
        collect_mod_observations(buffer, 1000)
        r = SuperLearn(agent_name, buffer, True)
        total_rewards += r
        lib.plot(total_rewards, figure_n=3)

    




"""Run Training sets"""

def RunDecDQNTraining(agent_name, start=1, n_runs=5, create=False):
    buffer = ReplayBuffer()
    total_rewards = []

    # collect_mod_observations(buffer, 5000)

    if create:
        rewards = TrainDecAgent(agent_name, buffer, 0, False)
        total_rewards += rewards
        lib.plot(total_rewards, figure_n=3)

    evals = []
    for i in range(start, start + n_runs):
        print(f"Running batch: {i}")
        rewards = TrainDecAgent(agent_name, buffer, 0, True)
        total_rewards += rewards

        lib.plot(total_rewards, figure_n=3)
        plt.figure(3).savefig("PNGs/Training_DQN" + str(i))
        np.save('DataRecords/' + agent_name + '_rewards1.npy', total_rewards)
        # agent = TestDecDQN(12, 10, agent_name)
        # s = single_rep_eval(agent)
        # evals.append(s)


if __name__ == "__main__":
    # mod_name = "ModTestDqnIntermediate"
    switch_name = "DecBuild"
    switch_name = "TestingSR"
    switch_name = "SwitchSR"


    # agent = TestModDQN(12, 10, mod_name)
    # single_rep_eval(agent, True)
    #  
    RunSuperLearn(switch_name)

    # RunDecDQNTraining(dec_name, 0, 5, True)
    # RunDecDQNTraining(dec_name, 5, 5, False)
    # RunDecDQNTraining(dec_name, 10, 5, False)




