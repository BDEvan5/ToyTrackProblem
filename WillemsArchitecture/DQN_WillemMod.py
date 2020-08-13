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
from CommonTestUtilsDQN import single_evaluation, ReplayBufferDQN
from TrainEnvWillemMod import TrainEnvWillem


#Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
h_size = 512
BATCH_SIZE = 64


EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

name00 = 'DataRecords/TrainTrack1000.npy'
name10 = 'DataRecords/TrainTrack1010.npy'
name20 = 'DataRecords/TrainTrack1020.npy'


class Qnet(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Qnet, self).__init__()
        
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.exploration_rate = EXPLORATION_MAX

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      


class TrainWillemModDQN:
    def __init__(self, obs_space, action_space, name):
        self.action_space = action_space
        self.obs_space = obs_space
        self.name = name

        self.model = None
        self.target = None

        self.update_steps = 0

    def act(self, obs):
        if random.random() < self.model.exploration_rate:
            return [random.randint(0, self.action_space-1)]
        else: 
            obs_t = torch.from_numpy(obs).float()
            a = self.greedy_action(obs_t)
            return [a]

    def greedy_action(self, obs):
        out = self.model.forward(obs)
        return out.argmax().item()

    def train_modification(self, buffer):
        n_train = 1
        for i in range(n_train):
            if buffer.size() < BATCH_SIZE:
                return
            s, a, r, s_p, done = buffer.memory_sample(BATCH_SIZE)

            target = r.detach()
            a = torch.squeeze(a, dim=-1)

            q_vals = self.model.forward(s)
            q_a = q_vals.gather(1, a)

            loss = F.mse_loss(q_a, target)

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
        self.update_networks()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
            self.model.exploration_rate *= EXPLORATION_DECAY 
            self.model.exploration_rate = max(EXPLORATION_MIN, self.model.exploration_rate)

    def save(self, directory="./dqn_saves"):
        filename = self.name
        torch.save(self.model, '%s/%s_model.pth' % (directory, filename))
        torch.save(self.target, '%s/%s_target.pth' % (directory, filename))
        print(f"Saved Agent: {self.name}")

    def load(self, directory="./dqn_saves"):
        filename = self.name
        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        self.target = torch.load('%s/%s_target.pth' % (directory, filename))
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
        self.model = Qnet(self.obs_space, self.action_space)
        self.target = Qnet(self.obs_space, self.action_space)
        print(f"Created new model")



class TestWillemModDQN:
    def __init__(self, obs_space, act_space, name="best_avg"):
        self.mod_space = 9
        self.model = Qnet(obs_space + 1, self.mod_space)
        self.name = name
        self.action_space = act_space

        self.load()

    def load(self, directory="./dqn_saves"):
        filename = self.name
        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        print(f"Loaded Agent: {self.name}")

    def act(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        a = out.argmax().item()
        return [a]



"""Training functions"""
def collect_willem_mod_observations(buffer, n_itterations=5000):
    env = TrainEnvWillem()

    for i in range(n_itterations):
        s = env.reset()
        action = env.random_action()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        buffer.put((s, action, r, s_p, done_mask))

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()
    print(" ")

def TrainWillemModAgent(agent_name, buffer, i=0, load=True):
    env = TrainEnvWillem()
    agent = TrainWillemModDQN(env.state_space, env.action_space, agent_name)
    agent.try_load(load)

    print_n = 100
    rewards = []
    score = 0.0
    for n in range(1000):
        state = env.reset()
        a = agent.act(state)
        s_prime, r, done, _ = env.step(a)
        done_mask = 0.0 if done else 1.0
        buffer.put((state, a, r, s_prime, done_mask)) # never done
        score += r
        agent.train_modification(buffer)

        # env.render(True)

        if n % print_n == 0 and n > 0:
            rewards.append(score)
            env.render()    
            exp = agent.model.exploration_rate
            mean = np.mean(rewards)
            b = buffer.size()
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
            score = 0
            lib.plot(rewards, figure_n=2)

            agent.save()
            test_agent = TestWillemModDQN(12, 5, agent_name)
            s = single_evaluation(test_agent)
            
    agent.save()

    return rewards

def RunWillemModTraining(agent_name, start=0, n_runs=5, create=False):
    buffer = ReplayBufferDQN()
    total_rewards = []

    evals = []

    if create:
        collect_willem_mod_observations(buffer, 50)
        rewards = TrainWillemModAgent(agent_name, buffer, 0, False)
        total_rewards += rewards
        lib.plot(total_rewards, figure_n=3)
        agent = TestWillemModDQN(12, 10, agent_name)
        s = single_evaluation(agent)
        evals.append(s)

    for i in range(start, start + n_runs):
        print(f"Running batch: {i}")
        rewards = TrainWillemModAgent(agent_name, buffer, i, True)
        total_rewards += rewards

        lib.plot(total_rewards, figure_n=3)
        plt.figure(2).savefig("PNGs/Training_DQN_rep" + str(i))
        np.save('DataRecords/' + agent_name + '_rewards1.npy', total_rewards)
        agent = TestWillemModDQN(12, 10, agent_name)
        s = single_evaluation(agent)
        evals.append(s)

    try:
        print(evals)
        print(f"Max: {max(evals)}")
    except:
        pass

if __name__ == "__main__":
    agent_name = "TestingWillemMod"
    # agent_name = "ModTest"

    # agent = TestWillemModDQN(12, 5, agent_name)
    # single_evaluation(agent, True, True)

    RunWillemModTraining(agent_name, 0, 5, create=True)
    # RunWillemModTraining(agent_name, 5, 5, False)
    # RunWillemModTraining(agent_name, 10, 5, create=False)

    # agent = TestWillemModDQN(12, 10, agent_name)
    # single_evaluation(agent, True)





