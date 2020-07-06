import numpy as np
import matplotlib.pyplot as plt 
import sys
import collections
import gym
import timeit
import random
import torch

import LibFunctions as lib
from TestEnv import TestEnv
from TrainReplacementEnv import TrainRepEnv
from TrainModEnv import TrainModEnv
from Corridor import CorridorAgent, PurePursuit
from ReplacementDQN import TestRepDQN, TrainRepDQN
from ModificationDQN import TestModDQN, TrainModDQN

name00 = 'DataRecords/TrainTrack1000.npy'
name10 = 'DataRecords/TrainTrack1010.npy'
name20 = 'DataRecords/TrainTrack1020.npy'
name30 = 'DataRecords/TrainTrack1030.npy'
name40 = 'DataRecords/TrainTrack1040.npy'
name50 = 'DataRecords/TrainTrack1050.npy'
name60 = 'DataRecords/TrainTrack1060.npy'
name70 = 'DataRecords/TrainTrack1070.npy'
name80 = 'DataRecords/TrainTrack1080.npy'
name90 = 'DataRecords/TrainTrack1090.npy'
name01 = 'DataRecords/TrainTrack1100.npy'

MEMORY_SIZE = 100000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=MEMORY_SIZE)
    
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


def evaluate_agent(env, agent, show=True):
    agent.load()
    print_n = 10
    show_n = 10

    rewards = []
    for n in range(100):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.act(state)
            s_prime, _, done, _ = env.step(a)
            state = s_prime
            score += 1 # counts number of steps
            if show:
                # env.box_render()
                env.full_render()
                pass
            
        rewards.append(score)

        if n % print_n == 1:
            mean = np.mean(rewards[-20:])
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} ")
        if show and n%show_n == 1:
            lib.plot(rewards, figure_n=2)
            plt.figure(2).savefig("Testing_" + agent.name)
            env.render()


def collect_observations(buffer, env_track_name, n_itterations=10000):
    env = TrainRepEnv(env_track_name)
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

def collect_mod_observations(buffer0, buffer1, env_track_name, n_itterations=10000):
    env = TrainModEnv(env_track_name)
    pp = PurePursuit(env.state_space, env.action_space)
    s, done = env.reset(), False
    for i in range(n_itterations):
        pre_mod = pp.act(s)
        system = random.randint(0, 1)
        if system == 1:
            mod_act = random.randint(0, 1)
            action_modifier = 2 if mod_act == 1 else -2
            action = pre_mod + action_modifier # swerves left or right
            action = np.clip(action, 0, env.action_space-1)
        else:
            action = pre_mod
        # action = env.random_action()
        s_p, r, done, r2 = env.step(action)
        done_mask = 0.0 if done else 1.0
        buffer0.put((s, system, r2, s_p, done_mask))
        if system == 1: # mod action
            buffer1.put((s, mod_act, r, s_p, done_mask))
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()
    print(" ")



def DebugAgentTraining(agent, env):
    rewards = []
    
    # observe_myenv(env, agent, 5000)
    for n in range(5000):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.learning_act(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            agent.buffer.append((state, a, r, s_prime, done_mask))
            state = s_prime
            score += r
            agent.experience_replay()
            env.box_render()
            
        rewards.append(score)

        exp = agent.exploration_rate
        mean = np.mean(rewards[-20:])
        b = len(agent.buffer)
        print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
        env.render()
        agent.save()

def DebugModAgent(i=0, load=False):
    track_name = name50
    agent_name = "DebugMod"
    buffer0 = ReplayBuffer()
    buffer1 = ReplayBuffer()


    env = TrainModEnv(track_name)
    agent = TrainModDQN(env.state_space, env.action_space, agent_name)
    agent.try_load(load)

    rewards = []
    for n in range(201):
        score, done, state = 0, False, env.reset()
        while not done:
            a, system, mod_act = agent.full_action(state)
            s_prime, r, done, r2 = env.step(a)
            done_mask = 0.0 if done else 1.0
            buffer0.put((state, a, r2, s_prime, done_mask))
            if system == 1: # mod action
                buffer1.put((state, mod_act, r, s_prime, done_mask))
            state = s_prime
            score += r
            agent.experience_replay(buffer0, buffer1)
            env.box_render()
        rewards.append(score)

        env.render()    
        exp = agent.model.exploration_rate
        mean = np.mean(rewards[-20:])
        b0 = buffer0.size()
        b1 = buffer1.size()
        print(f"Run: {n} --> Score: {score:.4f} --> Mean: {mean:.4f} --> exp: {exp:.4f} --> Buf: {b0, b1}")

    # agent.save()
    lib.plot(rewards, figure_n=2)
    plt.figure(2).savefig("DataRecords/Training_DQN" + str(i))

    return rewards


def TrainRepAgent(track_name, agent_name, buffer, i=0, load=True):
    env = TrainRepEnv(track_name)
    agent = TrainRepDQN(env.state_space, env.action_space, agent_name)
    agent.try_load(load)

    print_n = 10
    rewards = []
    for n in range(102):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.learning_act(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            buffer.put((state, a, r, s_prime, done_mask))
            state = s_prime
            score += r
            agent.experience_replay(buffer)
            # env.box_render()
        rewards.append(score)

        if n % print_n == 1:
            env.render()    
            exp = agent.model.exploration_rate
            mean = np.mean(rewards[-20:])
            b = buffer.size()
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")

    agent.save()

    # lib.plot(rewards, figure_n=2)
    # plt.figure(2).savefig("DataRecords/Training_DQN" + str(i))

    return rewards

def TrainModAgent(track_name, agent_name, buffer0, buffer1, i=0, load=True):
    env = TrainModEnv(track_name)
    agent = TrainModDQN(env.state_space, env.action_space, agent_name)
    agent.try_load(load)

    print_n = 20
    rewards = []
    for n in range(201):
        score, done, state = 0, False, env.reset()
        while not done:
            a, system, mod_act = agent.full_action(state)
            s_prime, r, done, r2 = env.step(a)
            done_mask = 0.0 if done else 1.0
            buffer0.put((state, a, r2, s_prime, done_mask))
            if system == 1: # mod action
                buffer1.put((state, mod_act, r, s_prime, done_mask))
            state = s_prime
            score += r
            agent.experience_replay(buffer0, buffer1)
            # env.box_render()
        rewards.append(score)

        if n % print_n == 1:
            env.render()    
            exp = agent.model.exploration_rate
            mean = np.mean(rewards[-20:])
            b0 = buffer0.size()
            b1 = buffer1.size()
            print(f"Run: {n} --> Score: {score:.4f} --> Mean: {mean:.4f} --> exp: {exp:.4f} --> Buf: {b0, b1}")

    agent.save()
    lib.plot(rewards, figure_n=2)
    plt.figure(2).savefig("DataRecords/Training_DQN" + str(i))

    return rewards

# run training
def RunRepDQNTraining():
    track_name = name50
    agent_name = "DQNtrain2"
    buffer = ReplayBuffer()
    total_rewards = []

    collect_observations(buffer, track_name, 500)

    rewards = TrainRepAgent(track_name, agent_name, buffer, 0, False)
    total_rewards += rewards
    for i in range(1, 100):
        print(f"Running batch: {i}")
        rewards = TrainRepAgent(track_name, agent_name, buffer, i, True)
        total_rewards += rewards

        lib.plot(total_rewards, figure_n=2)
        plt.figure(2).savefig("DataRecords/Training_DQN" + str(i))

def RunModDQNTraining():
    track_name = name80
    agent_name = "DQNtrainMod3"
    buffer0 = ReplayBuffer()
    buffer1 = ReplayBuffer()
    total_rewards = []

    collect_mod_observations(buffer0, buffer1, track_name, 5000)

    rewards = TrainModAgent(track_name, agent_name, buffer0, buffer1, 0, False)
    # total_rewards += rewards
    for i in range(1, 100):
        print(f"Running batch: {i}")
        rewards = TrainModAgent(track_name, agent_name, buffer0, buffer1, 0, True)
        total_rewards += rewards

        lib.plot(total_rewards, figure_n=3)
        plt.figure(2).savefig("DataRecords/Training_DQN" + str(i))

        


# testing algorithms
def RunRepDQNTest():
    env = TestEnv(name30)
    env.run_path_finder()
    # env = gym.make('CartPole-v1')
    agent = TestRepDQN(env.state_space, env.action_space, "DQNtrain2")

    evaluate_agent(env, agent)

def RunModDQNTest():
    env = TestEnv(name30)
    env.run_path_finder()
    # env = gym.make('CartPole-v1')
    agent = TestModDQN(env.state_space, env.action_space, "TestDQN-CartPole")

    evaluate_agent(env, agent)

def RunCorridorTest():
    env = TestEnv(name00)

    corridor_agent = CorridorAgent(env.state_space ,env.action_space)

    evaluate_agent(env, corridor_agent, True)

def RunPurePursuitTest():
    env = TestEnv(name30)
    env.run_path_finder()
    agent = PurePursuit(env.state_space ,env.action_space)

    evaluate_agent(env, agent, True)


if __name__ == "__main__":
    # DebugModAgent()

    # RunRepDQNTraining()
    RunModDQNTraining()

    # RunCorridorTest()
    # RunRepDQNTest()
    # RunModDQNTest()
    # RunPurePursuitTest()
