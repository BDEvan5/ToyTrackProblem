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
from Corridor import CorridorAgent, PurePursuit

from BasicTrainRepEnv import BasicTrainRepEnv
from ReplacementDQN import TestRepDQN, TrainRepDQN

from BasicTrainModEnv import BasicTrainModEnv
from ModificationDQN import TestModDQN, TrainModDQN

test00 = 'TestTrack1000'
test10 = 'TestTrack1010'
test20 = 'TestTrack1020'
test30 = 'TestTrack1030'
test40 = 'TestTrack1040'
test50 = 'TestTrack1050'

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
def RunRepDQNTest(map_name, agent_name):
    env = TestEnv(map_name)
    # agent_name = "DQNtrain2"
    agent = TestRepDQN(env.state_space, env.action_space, agent_name)
    # env.show_map()
    evaluate_agent(env, agent, True)
    # evaluate_agent(env, agent, False)

def RunModDQNTest(map_name, agent_name):
    env = TestEnv(map_name)
    # agent_name = "DQNtrainMod3"
    agent = TestModDQN(env.state_space, env.action_space, agent_name)

    evaluate_agent(env, agent, True)
    # evaluate_agent(env, agent, False)

def RunCorridorTest(map_name):
    env = TestEnv(map_name)
    corridor_agent = CorridorAgent(env.state_space ,env.action_space)

    evaluate_agent(env, corridor_agent, True)
    evaluate_agent(env, corridor_agent, False)

def RunPurePursuitTest(map_name):
    env = TestEnv(map_name)
    agent = PurePursuit(env.state_space ,env.action_space)

    # evaluate_agent(env, agent, True)
    evaluate_agent(env, agent, False)


if __name__ == "__main__":
    # DebugModAgent()

    map_name = test00
    # map_name = test10

    # rep_name = "RepTestDqnStd"
    # rep_name = "RepTestDqnSquare"
    # rep_name = "RepOpt1"
    # rep_name = "RepBasicTrain2"
    # rep_name = "Testing"
    # mod_name = "ModTestDqnIntermediate"
    mod_name = "ModBuild"

    # RunRepDQNTraining(rep_name, 0, 5, create=True)
    # RunRepDQNTraining(rep_name, 5, 5, False)
    # RunRepDQNTraining(rep_name, 10, 5, create=False)

    RunModDQNTraining(mod_name, 0, 5, True)
    RunModDQNTraining(mod_name, 5, 5, False)
    # RunModDQNTraining(mod_name, 10, 5, False)


    # RunCorridorTest(map_name)
    # RunPurePursuitTest(map_name)
    # RunRepDQNTest(map_name, rep_name)
    # RunModDQNTest(map_name, mod_name)

    # single_rep_eval(rep_name, test00, True)
