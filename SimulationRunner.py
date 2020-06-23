"""
This class is the high level box that holds all the parts and makes them interact
It will have a funciton with the big training loop

"""
import numpy as np 
from collections import deque
from matplotlib import pyplot as plt

from Env import MakeEnv
from AgentTD3 import TD3
import LibFunctions as lib
from TrackMapInterface import load_map, render_ep, make_new_map

def observe(agent, env):
    state, score, done = env.reset(), 0.0, False
    for i in range(10000):
        action = agent.get_new_target(state)

        new_state, reward, done = env.step(action)

        agent.add_data(state, new_state, reward, done)
        state = new_state
        if done:
            state = env.reset()
        print("\rObserving {}/10000".format(i), end="")
    agent.save_buffer()


def RunSimulationLearning2():
    name = "TrainTrack1"
    # make_new_map(name)
    track = load_map(name)
    env = MakeEnv(track)
    agent = TD3(7, 1, 1)
    show_n = 10
    ep_histories = []

    # observe(agent, env)
    # agent.save_buffer()
    agent.load_buffer()
    print(f"Running learning ")
    all_scores = []
    for i in range(10000):
        state, score, done = env.reset(), 0, False

        length, memory = 0, []
        while not done:
            action = agent.get_new_target(state)
            old_ss = env.ss()
            new_state, reward, done = env.step(action)
            memory.append((old_ss, action, reward, env.ss(), done))
            score += reward
            length += 1
            state = new_state

            agent.add_data(state, new_state, reward, done)
            agent.train()

        print(f"{i}:-> Score: {score} -> Length: {length}")
        all_scores.append(score)
        lib.plot(all_scores)
        # if i % show_n == 1:
        #     render_ep(track, memory, pause=True)
        ep_histories.append(memory)

def RunSimulationLearning3():
    # name = "TrainTrack1"
    name = "TrainTrackEmpty"
    # make_new_map(name)
    track = load_map(name)
    env = MakeEnv(track)
    agent = TD3(2, 2, 2)
    avg_n = 10
    save_n = 50
    agent_name = "NoRanges"

    agent.load_buffer()
    agent.load(agent_name)
    # observe(agent, env)
    print(f"Running learning ")
    all_scores = []
    state, score = env.reset(), 0
    for i in range(100000): # batches
        for j in range(32): # batch size
            action = agent.get_new_target(state)
            new_state, reward, done = env.step(action)
            agent.add_data(state, new_state, reward, done)
            score += reward
            state = new_state

            agent.train()

            if done:
                state = env.reset()
                print(f"{i}.{j}:-> Score: {score}")
                all_scores.append(score)
                score = 0

        if i % avg_n == 1:
            lib.plot(all_scores, figure_n=3, moving_avg_period=20)

        if i % save_n == 1:
            fig = plt.figure(3)
            fig.savefig("Training Rewards")
            agent.save(agent_name)


def RunSimulationTest():
    name = "ValueTrack1"
    # make_new_map(name)
    track = load_map(name)
    vehicle = Vehicle()
    full_path = vehicle.plan_path(track, load=True)
    simulator = Simulation(track)

    ep_histories = []

    for i in range(1):
        state, score, done = simulator.reset(), 0, False
        vehicle.reset()
        length, memory = 0, []
        while not done:
            action = vehicle.get_action(state)

            new_state, reward, done = simulator.step(action)
            memory.append((state, action, reward, new_state, done))
            score += reward
            length += 1
            state = new_state

        print(f"{i}:-> Score: {score} -> Length: {length}")
        render_ep(track, full_path, memory, True)
        ep_histories.append(memory)
        

if __name__ == "__main__":
    # RunSimulationLearning2()
    RunSimulationLearning3()