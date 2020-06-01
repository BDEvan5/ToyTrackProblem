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

def plot(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    plt.ylim(0, 0.2)

    moving_avg = f.get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    moving_avg = f.get_moving_average(moving_avg_period * 5, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)


def test(config):
    f_show = 20
    map_name = "ValueTrack1"
    track, path_obj = load_generated_map(map_name, False)

    # run sim
    env = RaceEnv(config, track)
    model = Model(3)
    model.load_weights()

    runner = NewRunner(env, model, path_obj)
    avg = runner.run_test(track)
    print(f"Average: {avg}")


def learn(config):
    f_show = 20
    map_name = "ValueTrack1"
    track, path_obj = load_generated_map(map_name, False)
    # track, path_obj = load_generated_map(map_name, True)

    # run sim
    env = RaceEnv(config, track)

    replay_ratio = 1 #replay ratio of on to off policy learning

    model = Model(3)
    replay_buffer = ReplayBuffer(20)

    runner = NewRunner(env, model, path_obj)
    losses = []
    for n in range(1000):

        b = runner.run_batch(track)
        replay_buffer.add_batch(b)
        show_plot = False
        if n % f_show == 1:
            show_plot = True
        model.update_model(b, show_plot)

        losses.append(model._loss_fcn_value())
        plot(losses, figure_n=3, title="Lossses")

        for _ in range(replay_ratio):
            b = replay_buffer.get_random_batch()
            model.update_model(b)

    model.save_model()
    fig = plt.figure(3)
    fig.savefig('DataRecords/LosssesTraining')
    # env.sim_mem.print_ep()
    # render_track_ep(track, path_obj, env.sim_mem, pause=True)



if __name__ == "__main__":
    config = create_sim_config()
    # learn(config)
    test(config)
