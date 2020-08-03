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
from TrainEnv import TrainEnv
from CommonTestUtils import single_evaluation, ReplayBuffer

#Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995



class Qnet(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Qnet, self).__init__()
        h_size = 512
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc_angle1 = nn.Linear(h_size, h_size)
        self.fc_angle2 = nn.Linear(h_size, action_space)
        self.fc_vel1 = nn.Linear(h_size, h_size)
        self.fc_vel2 = nn.Linear(h_size, action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.exploration_rate = EXPLORATION_MAX

    def forward(self, obs):
        x = F.relu(self.fc1(obs))

        h_angle = F.relu(self.fc_angle1(x))
        angle = self.fc_angle2(h_angle)

        h_vel = F.relu(self.fc_vel1(x))
        vel = self.fc_vel2(h_vel)

        # aim for a 10x2 dim
        ret = torch.stack([angle, vel], dim=-1)

        return ret

class TrainRepDQN:
    def __init__(self, obs_space, action_space, name="best_avg"):
        self.obs_space = obs_space
        self.action_space = action_space
        self.model = None
        self.target = None
        
        self.update_steps = 0
        self.name = name

    def learning_act(self, obs):
        if random.random() < self.model.exploration_rate:
            r1 = random.randint(0, self.action_space-1)
            r2 = random.randint(0, self.action_space-1)
            action = [r1, r2]
        else: 
            action = self.act(obs)

        return action

    def act(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)

        a = torch.argmax(out, dim=0).numpy()
        return a

    def experience_replay(self, buffer):
        n_train = 1
        for i in range(n_train):
            if buffer.size() < BATCH_SIZE:
                return 0
            s, a, r, s_p, done = buffer.memory_sample(BATCH_SIZE)

            target = r.detach().squeeze(dim=-1)
            angle_target = target[:, :, 0]
            vel_target = target[:, :, 1]

            q_vals = self.model.forward(s).double()
            q_angles = q_vals[:, :, 0]
            q_vels = q_vals[:, :, 1]
            a_angles = a[:,:, 0]
            a_vels = a[:, :, 1]
            q_angles_a = q_angles.gather(1, a_angles)
            q_vels_a = q_vels.gather(1, a_vels)

            # q_a = torch.cat([q_angles_a, q_vels_a], dim=1)
            # q_a = torch.gather(q_vals, 1)
            # loss = F.mse_loss(q_a, q_update.detach())

            # uses two seperate losses
            loss = F.mse_loss(q_vels_a, vel_target) + F.mse_loss(q_angles_a, angle_target)
            # loss = F.mse_loss(q_angles_a, target)
            # angle_loss = F.mse_loss(q_angles_a, target)
            # vel_loss = 

            self.model.optimizer.zero_grad()
            loss.double()
            loss.backward()
            self.model.optimizer.step()

        self.update_networks()

        return loss.item()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
            self.model.exploration_rate *= EXPLORATION_DECAY 
            self.model.exploration_rate = max(EXPLORATION_MIN, self.model.exploration_rate)

    def save(self, directory="./dqn_saves"):
        filename = self.name
        # torch.save(self.model.state_dict(), '%s/%s_model.pth' % (directory, filename))
        # torch.save(self.target.state_dict(), '%s/%s_target.pth' % (directory, filename))
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
                pass
        else:
            self.model = Qnet(self.obs_space, self.action_space)
            self.target = Qnet(self.obs_space, self.action_space)
            print(f"Not loading - restarting training")

class TestRepDQN:
    def __init__(self, obs_space, act_space, name="best_avg"):
        # self.model = Qnet(obs_space, act_space)
        self.model = None
        self.name = name

        self.load() # always load by name

    def load(self, directory="./dqn_saves"):
        filename = self.name
        # self.model.load_state_dict(torch.load('%s/%s_model.pth' % (directory, filename)))
        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        print(f"Loaded Agent: {self.name}")

    def act(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        action = torch.argmax(out, dim=0).numpy()

        return action


"""Training functions"""
def collect_rep_observations(buffer, n_itterations=5000):
    env = TrainEnv()
    env.pure_rep()
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

def TrainRepAgent(agent_name, buffer, i=0, load=True):
    env = TrainEnv()
    env.pure_rep()
    agent = TrainRepDQN(env.state_space, env.action_space, agent_name)
    agent.try_load(load)

    print_n = 100
    rewards = []
    score = 0.0
    for n in range(1000):
        state = env.reset()

        a = agent.learning_act(state)
        s_prime, r, done, _ = env.step(a)
        done_mask = 0.0 if done else 1.0
        buffer.put((state, a, r, s_prime, done_mask)) # never done
        l = agent.experience_replay(buffer)
        # score += l
        score += np.mean(r)

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
            test_agent = TestRepDQN(12, 10, agent_name)
            single_evaluation(test_agent, True, False)

    return rewards

def TrainRepAgentDebug(agent_name, buffer, i=0, load=True):
    env = TrainEnv()
    env.pure_rep()
    agent = TrainRepDQN(env.state_space, env.action_space, agent_name)
    agent.try_load(load)

    print_n = 100
    rewards = []
    score = 0.0
    for n in range(1000):
        state = env.reset()

        a = agent.learning_act(state)
        s_prime, r, done, _ = env.step(a)
        done_mask = 0.0 if done else 1.0
        buffer.put((state, a, r, s_prime, done_mask)) # never done
        env.render()
        l = agent.experience_replay(buffer)
        # score += l
        score += r

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
            test_agent = TestRepDQN(12, 10, agent_name)
            single_evaluation(test_agent, True)

    return rewards


def RunRepDQNTraining(agent_name, start=0, n_runs=5, create=False):
    buffer = ReplayBuffer()
    total_rewards = []

    evals = []

    if create:
        collect_rep_observations(buffer, 500)
        rewards = TrainRepAgent(agent_name, buffer, 0, False)
        total_rewards += rewards
        lib.plot(total_rewards, figure_n=3)
        agent = TestRepDQN(12, 10, agent_name)
        s = single_evaluation(agent)
        evals.append(s)

    for i in range(start, start + n_runs):
        print(f"Running batch: {i}")
        rewards = TrainRepAgent(agent_name, buffer, i, True)
        # TrainRepAgentDebug(agent_name, buffer, i, True)
        total_rewards += rewards

        lib.plot(total_rewards, figure_n=3)
        plt.figure(2).savefig("PNGs/Training_DQN_rep" + str(i))
        np.save('DataRecords/' + agent_name + '_rewards1.npy', total_rewards)
        agent = TestRepDQN(12, 10, agent_name)
        s = single_evaluation(agent)
        evals.append(s)

    try:
        print(evals)
        print(f"Max: {max(evals)}")
    except:
        pass


"""Training loops - Super Learn"""
def SuperLearn(agent_name, buffer, load=True):
    agent = TrainRepDQN(12, 10, agent_name)
    agent.try_load(load)
    print_n = 100
    rewards, score = [], 0.0
    for n in range(2000):
        loss = agent.experience_replay(buffer)
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

    collect_rep_observations(buffer, 5000)
    if create:
        r = SuperLearn(agent_name, buffer, False)
        total_rewards += r
        lib.plot(total_rewards)

    for i in range(100):
        collect_rep_observations(buffer, 1000)
        r = SuperLearn(agent_name, buffer, True)
        total_rewards += r
        lib.plot(total_rewards, figure_n=3)
        agent = TestRepDQN(12, 10, agent_name)
        single_evaluation(agent, True)

    

if __name__ == "__main__":
    # rep_name = "RepSR"
    rep_name = "Testing"
    # rep_name = "RepTest"


    RunRepDQNTraining(rep_name, 0, 5, create=True)
    # RunRepDQNTraining(rep_name, 5, 5, False)
    # RunRepDQNTraining(rep_name, 10, 5, create=False)

    # agent = TestRepDQN(12, 10, rep_name)
    # single_evaluation(agent, True)
