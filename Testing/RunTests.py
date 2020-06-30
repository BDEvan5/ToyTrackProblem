import numpy as np
import matplotlib.pyplot as plt 
import sys
import psutil as ps

import LibFunctions as lib
from TestEnv import TestEnv
from TrainEnv import TrainEnv
from Corridor import CorridorAgent, PurePursuit
from ReplacementDQN import TestDQN, TrainDQN

name00 = 'DataRecords/TrainTrack1000.npy'
name10 = 'DataRecords/TrainTrack1010.npy'
name20 = 'DataRecords/TrainTrack1020.npy'
name30 = 'DataRecords/TrainTrack1030.npy'


import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

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
                env.box_render()
                pass
            
        rewards.append(score)

        if n % print_n == 1:
            mean = np.mean(rewards[-20:])
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} ")
        if show and n%show_n == 1:
            lib.plot(rewards, figure_n=2)
            plt.figure(2).savefig("Testing_" + agent.name)
            env.render()

def collect_observations(env, agent, n_itterations=10000):
    s, done = env.reset(), False
    for i in range(n_itterations):
        action = env.random_action()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        agent.buffer.append((s, action, r, s_p, done_mask))
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()
    print(" ")


def TrainAgent(agent, env):
    print_n = 20
    rewards = []
    collect_observations(env, agent, 5000)
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
        rewards.append(score)

        if ps.virtual_memory().free < 5e8:
            print("Memory Error: Breaking")
            break

        if n % print_n == 1:
            exp = agent.exploration_rate
            mean = np.mean(rewards[-20:])
            b = len(agent.buffer)
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
            agent.save()

            agent_size = getsize(agent) / 1e6
            env_size = getsize(env) / 1e6
            print(f"Sizes --> Agent: {agent_size} -- > Env: {env_size}")

            # lib.plot(rewards, figure_n=2)
            # plt.figure(2).savefig("Training_" + agent_name)

    agent.save()
    lib.plot(rewards, figure_n=2)
    plt.figure(2).savefig("Training_DQN")


# run training
def RunDQNTraining():
    env = TrainEnv(name20)
    agent = TrainDQN(env.state_space, env.action_space, "TestDQN")

    TrainAgent(agent, env)


# testing algorithms
def RunDQNTest():
    env = TestEnv(name30)
    env.run_path_finder()
    agent = TestDQN(env.state_space, env.action_space, "TestDQN")

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
    # RunCorridorTest()
    # RunDQNTraining()
    # RunDQNTest()
    RunPurePursuitTest()
