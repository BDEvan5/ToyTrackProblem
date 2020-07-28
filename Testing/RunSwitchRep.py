import numpy as np
import matplotlib.pyplot as plt 
import sys
import collections
import gym
import timeit
import random

import LibFunctions as lib
from CommonTestUtils import ReplayBuffer, single_rep_eval
from Corridor import CorridorAgent, PurePursuit

from BasicTrainModEnv import BasicTrainModEnv
# from ModificationDQN import TestModDQN, TrainModDQN
from SwitchRepDQN import DecisionDQN


"""Collect obs"""

def collect_mod_observations(buffer, n_itterations=10000):
    env = BasicTrainModEnv()
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

    

# def TestSuperLearn(agent_name):
#     # write function here to generate different test data and determine the loss
#     buffer = ReplayBuffer()

#     collect_mod_observations(buffer, 2000)

#     agent = DecisionDQN(12, 10, agent_name)
#     agent.try_load(load)

#     print_n = 100
#     rewards, score = [], 0.0
#     for n in range(10000):
#         loss = agent.train_switching(buffer)
#         score += loss

#         if n % print_n == 0 and n > 0:
#             rewards.append(score)
#             mean = np.mean(rewards[-20:])
#             print(f"Run: {n} --> Score: {score:.4f} --> Mean: {mean:.4f}")
#             score = 0
#             lib.plot(rewards, figure_n=2)

#     agent.save()

#     return rewards


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
    dec_name = "DecBuild"

    # RunModDQNTraining(mod_name, 0, 2, True)

    # agent = TestModDQN(12, 10, mod_name)
    # single_rep_eval(agent, True)
    #  
    RunSuperLearn(dec_name)

    # RunDecDQNTraining(dec_name, 0, 5, True)
    # RunDecDQNTraining(dec_name, 5, 5, False)
    # RunDecDQNTraining(dec_name, 10, 5, False)
