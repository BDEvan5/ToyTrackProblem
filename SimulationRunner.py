"""
This class is the high level box that holds all the parts and makes them interact
It will have a funciton with the big training loop

"""
import numpy as np 

from Simulation import Simulation
from Vehicle import Vehicle

from TrackMapInterface import load_map


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
    track = load_map(name)
    vehicle = Vehicle()
    vehicle.plan_path(track, load=True)
    simulator = Simulation(track)

    for i in range(1000):
        state, score, done = simulator.reset(), 0, False
        vehicle.reset()
        length = 0
        while not done:
            action = vehicle.get_action(state)

            new_state, reward, done = simulator.step(action)
            score += reward
            length += 1
            state = new_state

        print(f"{i}:-> Score: {score} -> Length: {length}")


"""Helpers"""
def make_new_path(name):
    print(f"Generating Map: {name}")
    # generate
    myTrackMap = TrackGenerator(name)
    myTrackMap.name_var.set(name)
    myTrackMap.save_map()


if __name__ == "__main__":

    RunSimulationTest()