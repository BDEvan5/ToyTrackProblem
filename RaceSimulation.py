import numpy as np 
import logging
import time
from matplotlib import pyplot as plt
import os
import pickle

from RaceEnv import RaceEnv
from Config import create_sim_config
from PathPlanner import A_StarTrackWrapper, A_StarFinderMod
from TrackMapInterface import load_map, show_track_path, render_track_ep
from PathPrep import process_path

from ValueAgent import Model, ReplayBuffer, RunnerVanilla
from NewRunner import NewRunner


def get_track_path(load_opti_path=True, load_path=True):
    track = load_map("myTrack4")

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
                # path = A_StarTrackWrapper(track, 1)
                path = A_StarFinderMod(track, 1)
                np.save(path_file, path)
        else:
            path = A_StarFinderMod(track, 1)
            np.save(path_file, path)

        path_obj, path = process_path(path)

        pickle.dump(path_obj, db_file)

        show_track_path(track, path)
    db_file.close()

    return path_obj, track


def learn(config):
    # path_obj, track = get_track_path(True, True)
    path_obj, track = get_track_path(False, False)
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
