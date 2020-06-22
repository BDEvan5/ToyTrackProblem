"""
This class is the high level box that holds all the parts and makes them interact
It will have a funciton with the big training loop

"""
import numpy as np 
from collections import deque

from Simulation import Simulation
from Vehicle import Vehicle

from TrackMapInterface import load_map, render_ep, make_new_map

def observe(vehicle, simulator):
    for i in range(5000):
        state, score, done = simulator.reset(), 0.0, False
        while not done:
            action = vehicle.get_action(state)

            new_state, reward, done = simulator.step(action)
            score += reward
            state = new_state

            vehicle.agent.add_data(state, new_state, reward, done)
        print("\rObserving {}/5000".format(i), end="")


def RunSimulationLearning():
    name = "ValueTrack1"
    track = load_map(name)
    vehicle = Vehicle()
    vehicle.plan_path(track, load=True)
    simulator = Simulation(track)

    print_n = 1

    # observe(vehicle, simulator)
    # vehicle.agent.replay_buffer.save_buffer()
    vehicle.agent.load_buffer()
    print(f"Running learning ")
    for i in range(1000):
        state, score, done = simulator.reset(), 0.0, False
        vehicle.reset()
        while not done:
            action = vehicle.get_action(state)

            new_state, reward, done = simulator.step(action)
            score += reward
            state = new_state

            vehicle.agent.add_data(state, new_state, reward, done)
            vehicle.agent.train()
            # print(f"{simulator.steps} ")

        # train agent
        # if i % print_n == 1:
        print(f"{i}:-> Score: {score}")

def RunSimulationLearning2():
    name = "ValueTrack1"
    # make_new_map(name)
    track = load_map(name)
    vehicle = Vehicle()
    full_path = vehicle.plan_path(track, load=True)
    simulator = Simulation(track)

    ep_histories = []

    # vehicle.agent.load_buffer()
    print(f"Running learning ")
    for i in range(1000):
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

            vehicle.agent.add_data(state, new_state, reward, done)
            vehicle.agent.train()

        print(f"{i}:-> Score: {score} -> Length: {length}")
        render_ep(track, full_path, memory, True)
        ep_histories.append(memory)

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
    RunSimulationLearning2()
    # RunSimulationTest()