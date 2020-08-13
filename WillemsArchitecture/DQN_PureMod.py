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
from CommonTestUtilsDQN import single_evaluationDQN, ReplayBufferDQN
from TrainEnvDQN import TrainEnvDQN


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
    def __init__(self, obs_space, mod_space):
        super(Qnet, self).__init__()
        
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, mod_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.exploration_rate = EXPLORATION_MAX

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      

class ModificationDQN:
    def __init__(self):
        self.model = None
        self.target = None

        self.update_steps = 0

    def learning_mod_act(self, obs):
        if random.random() < self.model.exploration_rate:
            return random.randint(0, self.mod_space-1)
        else: 
            obs_t = torch.from_numpy(obs).float()
            return self.mod_act(obs_t)

    def mod_act(self, obs):
        out = self.model.forward(obs)
        return out.argmax().item()

    def train_modification(self, buffer1):
        n_train = 1
        for i in range(n_train):
            if buffer1.size() < BATCH_SIZE:
                return
            s, a, r, s_p, done = buffer1.memory_sample(BATCH_SIZE)

            pp_acts = self.get_pp_acts(s)[:, None]
            mod_acts = a - pp_acts + (self.mod_space-1) / 2
            mod_acts = mod_acts.clamp(0, self.mod_space-1)

            q_update = r.detach()

            pp_acts = torch.from_numpy(pp_acts).float()
            nn_s = torch.cat([s, pp_acts], dim=1)
            q_vals = self.model.forward(nn_s)
            q_a = q_vals.gather(1, mod_acts.long())

            loss = F.mse_loss(q_a, q_update.detach())

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
        self.model = Qnet(self.obs_space, self.mod_space)
        self.target = Qnet(self.obs_space, self.mod_space)
        print(f"Creating new model")


class TrainPureModDQN(ModificationDQN):
    def __init__(self, obs_space, action_space, name="best_avg"):
        self.mod_space = 9 # internal parameter
        self.action_space = action_space
        self.obs_space = obs_space + 1 # for the pp action
        self.name = name

        ModificationDQN.__init__(self)

    def act(self, obs):
        pp_action = self.get_pursuit_action(obs)
        
        nn_state = np.concatenate([obs, [pp_action]])
        mod = self.learning_mod_act(nn_state)
        new_action = pp_action + mod - (self.mod_space-1) / 2 # this means it can go straight or swerve up to 3 units in each direction
        new_action = np.clip(new_action, 0, self.action_space-1) # check this doesn't cause training problems

        return new_action
        
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


class TestPureModDQN:
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

    def mod_act(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        return out.argmax().item()

    def act(self, obs):
        pp_action = self.get_pursuit_action(obs)
        
        nn_state = np.concatenate([obs, [pp_action]])
        mod = self.mod_act(nn_state)
        new_action = pp_action + mod - (self.mod_space-1) / 2 # this means it can go straight or swerve up to 3 units in each direction
        new_action = np.clip(new_action, 0, self.action_space-1)

        return new_action

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

"""Training functions"""
def collect_pure_mod_observations(buffer, n_itterations=5000):
    env = TrainEnvDQN()
    env.pure_mod()
    s, done = env.reset(), False
    for i in range(n_itterations):
        action = env.random_action()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        buffer.put((s, action, r, s_p, done_mask))
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()
    print(" ")

def TrainPureModAgent(agent_name, buffer, i=0, load=True):
    env = TrainEnv()
    env.pure_mod()
    agent = TrainPureModDQN(env.state_space, env.action_space, agent_name)
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
            test_agent = TestPureModDQN(12, 10, agent_name)
            s = single_evaluation(test_agent)
            
    agent.save()

    return rewards

def RunPureModTraining(agent_name, start=0, n_runs=5, create=False):
    buffer = ReplayBufferDQN()
    total_rewards = []

    evals = []

    if create:
        collect_pure_mod_observations(buffer, 5000)
        rewards = TrainPureModAgent(agent_name, buffer, 0, False)
        total_rewards += rewards
        lib.plot(total_rewards, figure_n=3)
        agent = TestPureModDQN(12, 10, agent_name)
        s = single_evaluation(agent)
        evals.append(s)

    for i in range(start, start + n_runs):
        print(f"Running batch: {i}")
        rewards = TrainPureModAgent(agent_name, buffer, i, True)
        total_rewards += rewards

        lib.plot(total_rewards, figure_n=3)
        plt.figure(2).savefig("PNGs/Training_DQN_rep" + str(i))
        np.save('DataRecords/' + agent_name + '_rewards1.npy', total_rewards)
        agent = TestPureModDQN(12, 10, agent_name)
        s = single_evaluation(agent)
        evals.append(s)

    try:
        print(evals)
        print(f"Max: {max(evals)}")
    except:
        pass

if __name__ == "__main__":
    agent_name = "TestingPureMod"
    agent_name = "ModTest"

    # agent = TestPureModDQN(12, 10, agent_name)
    # single_evaluation(agent, True)

    RunPureModTraining(agent_name, 0, 5, create=True)
    # RunPureModTraining(agent_name, 5, 5, False)
    # RunPureModTraining(agent_name, 10, 5, create=False)

    agent = TestPureModDQN(12, 10, agent_name)
    single_evaluation(agent, True)





