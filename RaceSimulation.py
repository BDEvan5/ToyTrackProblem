import numpy as np 
import logging
import time
from matplotlib import pyplot as plt
import os
import pickle

from RaceEnv import RaceEnv
from Models import TrackData
from Config import create_sim_config
from PathPlanner import A_StarPathFinder
from TrajectoryOptimisation import optimise_trajectory, add_velocity
from PathTracker import Tracker
from Interface import show_path, render_ep
from TrackMapInterface import load_map

def get_path(load=False):
    filename = "DataRecords/path_obj.npy"
    db_file = open(filename, 'br+')

    track = load_map()

    if load:
        path_obj = pickle.load(db_file)
    else: 
        # plan path
        myPlanner = A_StarPathFinder(track)
        path = myPlanner.run_search(5)

        # optimise path
        path = optimise_trajectory(path)
        path_obj = add_velocity(path)

        # save
        pickle.dump(path_obj, db_file)

    db_file.close()

    return path_obj, track



def simulation_runner(config):
    path_obj, track = get_path(True)
    # show_path(track, path_obj)

    # for i, pt in enumerate(path_obj.route):
    #     print(i)
    #     pt.print_point()

    # run sim
    env = RaceEnv(config, track)
    state, done = env.reset(), False
    tracker = Tracker(path_obj)


    while not done:
        # state.print_point("State")
        ref_action = tracker.act(state)
        state, reward, done = env.step(ref_action)
        # print(f"Action: [{ref_action[0]:.2f} ; {ref_action[1]:.4f}]")

    # env.render_episode("DataRecords/PathTracker", False)
    render_ep(track, path_obj, env.sim_mem)





if __name__ == "__main__":
    get_path(False)
    # config = create_sim_config()
    # simulation_runner(config)
