import numpy as np
import matplotlib.pyplot as plt 
import sys
import collections
import gym
import timeit
import random
import torch

import LibFunctions as lib
from CommonTestUtils import single_rep_eval, ReplayBuffer
from Corridor import CorridorAgent, PurePursuit

from BasicTrainRepEnv import BasicTrainRepEnv
from ReplacementDQN import TestRepDQN, TrainRepDQN



def collect_rep_observations(buffer, env_track_name, n_itterations=5000):
    # env = TrainRepEnv(env_track_name)
    env = BasicTrainRepEnv()
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
    env = BasicTrainRepEnv()
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
        score += r
        agent.experience_replay(buffer)

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

    return rewards

def RunRepDQNTraining(agent_name, start=0, n_runs=5, create=False):
    buffer = ReplayBuffer()
    total_rewards = []

    collect_rep_observations(buffer, 5000)
    evals = []

    if create:
        rewards = TrainRepAgent(agent_name, buffer, 0, False)
        total_rewards += rewards
        lib.plot(total_rewards, figure_n=3)

    for i in range(start, start + n_runs) :
        print(f"Running batch: {i}")
        rewards = TrainRepAgent(agent_name, buffer, i, True)
        total_rewards += rewards

        lib.plot(total_rewards, figure_n=3)
        plt.figure(2).savefig("PNGs/Training_DQN_rep" + str(i))
        np.save('DataRecords/' + agent_name + '_rewards1.npy', total_rewards)
        agent = TestRepDQN(12, 10, agent_name)
        s = single_rep_eval(agent)
        evals.append(s)

    try:
        print(evals)
        print(f"Max: {max(evals)}")
    except:
        pass

if __name__ == "__main__":
    # rep_name = "RepTestDqnStd"
    # rep_name = "RepTestDqnSquare"
    # rep_name = "RepOpt1"
    # rep_name = "RepBasicTrain2"
    rep_name = "Testing"

    # RunRepDQNTraining(rep_name, 0, 5, create=True)
    RunRepDQNTraining(rep_name, 5, 5, False)
    # RunRepDQNTraining(rep_name, 10, 5, create=False)


