import numpy as np 
import logging
import time
from matplotlib import pyplot as plt
import os
import pickle

from RaceEnv import RaceEnv
from Config import create_sim_config
from TrackMapInterface import load_map, show_track_path, render_track_ep
from PathPrep import load_generated_map
import LibFunctions as f 

from ValueAgent import Model, ReplayBuffer
from NewRunner import NewRunner



def learn(config):
    map_name = "ValueTrack1"
    track, path_obj = load_generated_map(map_name, False)
    # track, path_obj = load_generated_map(map_name, True)

    # run sim
    env = RaceEnv(config, track)

    replay_ratio = 8 #replay ratio of on to off policy learning

    model = Model(3)
    replay_buffer = ReplayBuffer(20)

    runner = NewRunner(env, model, path_obj)
    losses = []
    for _ in range(500):
        b = runner.run_batch(track)
        replay_buffer.add_batch(b)
        model.update_model(b)

        losses.append(model._loss_fcn())
        # print(f"Loss {model.update_n}: {losses[-1]}")
        f.plot(losses, figure_n=3)

        for _ in range(replay_ratio):
            b = replay_buffer.get_random_batch()
            model.update_model(b)

    # env.sim_mem.print_ep()
    # render_track_ep(track, path_obj, env.sim_mem, pause=True)



if __name__ == "__main__":
    config = create_sim_config()
    learn(config)
