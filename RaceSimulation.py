import numpy as np 
import logging
import time
from matplotlib import pyplot as plt
import os
import pickle

from RaceEnv import RaceEnv
from Models import TrackData
from Config import create_sim_config
from PathPlanner import A_StarPathFinder, A_StarTrackWrapper
from TrajectoryOptimisation import add_velocity
from TrajectoryOptimisation import reduce_path, reduce_path_diag, optimise_track_trajectory # debugging
from PathTracker import Tracker
from Interface import show_path, render_ep
from TrackMapInterface import load_map, show_track_path, render_track_ep
import LibFunctions as f

from ValueAgent import Model, ReplayBuffer, RunnerVanilla
from NewRunner import NewRunner


def get_track_path(load_opti_path=True, load_path=True):
    track = load_map()

    filename = "DataRecords/path_obj_db"
    
    if load_opti_path:
        db_file = open(filename, 'br+')
        path_obj = pickle.load(db_file)
    else:
        db_file = open(filename, 'bw+')
        path_file = "DataRecords/path_list.npy"
        if load_path:
            try:
                path = np.load(path_file)
            except:
                path = A_StarTrackWrapper(track, 1)
                np.save(path_file, path)
        else:
            path = A_StarTrackWrapper(track, 1)
            np.save(path_file, path)

        # show_track_path(track, path)
        path = reduce_path_diag(path)

        path = optimise_track_trajectory(path, track)
        # show_track_path(track, path)
        path_obj = add_velocity(path)

        pickle.dump(path_obj, db_file)

    db_file.close()

    return path_obj, track


def learn(config):
    path_obj, track = get_track_path(True, True)
    # path_obj, track = get_track_path(False, False)
    # show_path(track, path_obj)

    # run sim
    env = RaceEnv(config, track)
    # tracker = Tracker(path_obj)

    print("Running Vanilla")
    replay_ratio = 8 #replay ratio of on to off policy learning

    model = Model(3)
    replay_buffer = ReplayBuffer(20)

    runner = NewRunner(env, model, path_obj)
    losses = []
    for _ in range(100):
        b = runner.run_batch(track)
        replay_buffer.add_batch(b)
        model.update_model(b)

        losses.append(model._loss_fcn())
        print(f"Loss {model.update_n}: {losses[-1]}")
        # f.plot(losses, figure_n=3)

        # env.sim_mem.print_ep()
        # render_track_ep(track, path_obj, env.sim_mem, pause=True)

        for _ in range(replay_ratio):
            b = replay_buffer.get_random_batch()
            model.update_model(b)

    # env.sim_mem.print_ep()
    # render_track_ep(track, path_obj, env.sim_mem, pause=True)



if __name__ == "__main__":
    # get_track_path(False, True)
    # get_track_path(False, False)
    config = create_sim_config()
    learn(config)
