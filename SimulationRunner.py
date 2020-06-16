"""
This class is the high level box that holds all the parts and makes them interact
It will have a funciton with the big training loop

"""
import numpy as np 
from collections import deque

from Simulation import Simulation
from Vehicle import Vehicle

from TrackMapInterface import load_map, render_ep, make_new_map


def RunSimulationLearning():
    name = "MyTrack1"
    track = load_map(name)
    vehicle = vehicle(track)
    vehicle.plan_path(load=False)
    simulator = Simulation(track)

    for i in range(1000):
        state, score, done = simulator.reset(), 0, False
        while not done:
            action = vehicle.get_action(state)

            new_state, reward, done = simulator.step(action)
            score += reward
            state = new_state

            # memory.store(new state)

        # train agent
        print(f"{i}:-> Score: {score}")




def RunSimulationTest():
    name = "ValueTrack1"
    make_new_map(name)
    track = load_map(name)
    vehicle = Vehicle()
    full_path = vehicle.plan_path(track, load=False)
    simulator = Simulation(track)

    ep_histories = []

    for i in range(10):
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
        

"""Helpers"""




if __name__ == "__main__":

    RunSimulationTest()