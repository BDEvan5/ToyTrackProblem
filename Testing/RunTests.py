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
from CommonTestUtils import CorridorAgent, PurePursuit, single_evaluation

from DQN_PureRep import TestRepDQN, Qnet
from DQN_PureMod import TestPureModDQN
from DQN_SwitchRep import TestSwitchRep

"""Final group Evals"""
def evaluate_agent(env, agent, show=True):
    agent.load()
    print_n = 10
    show_n = 10

    rewards, steps = [], []
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
        steps.append(env.steps) # record how many steps it took

        if n % print_n == 1 or show:
            mean = np.mean(rewards[-20:])
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> steps: {steps[-1]}")
        if (show and n%show_n == 1) or show:
            lib.plot(rewards, figure_n=2)
            plt.figure(2).savefig("PNGs/Testing_" + agent.name)
            env.render()

    mean_steps = np.mean(steps)
    print(f"Mean steps = {mean_steps}")



"""Run Tests"""
def RunPureRepDQNTest():
    rep_name = "RepTest"

    agent = TestRepDQN(12, 10, rep_name)
    single_evaluation(agent, True, True)

def RunPureModDQNTest():
    agent_name = "ModTest"

    agent = TestPureModDQN(12, 10, agent_name)
    single_evaluation(agent, True, True)

def RunSwitchRepDQNTest():
    switch_name = "SwitchSR"
    rep_name = "RepSR"

    agent = TestSwitchRep(12, 10, switch_name, rep_name)
    agent.load()
    single_evaluation(agent, True, True)

def RunCorridorTest():
    agent = CorridorAgent(12, 10)

    single_evaluation(agent, True, True)

def RunPurePursuitTest():
    agent = PurePursuit(12, 10)

    single_evaluation(agent, True, True)


if __name__ == "__main__":


    # RunCorridorTest()
    # RunPurePursuitTest()

    RunPureRepDQNTest()
    # RunPureModDQNTest()

    # single_rep_eval(rep_name, test00, True)
