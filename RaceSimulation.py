import numpy as np 
import logging
import time
from matplotlib import pyplot as plt
import os

from RaceEnv import RaceEnv
# from ReplayBuffer import ReplayBuffer
from Models import TrackData
from ClassicalAgent import Classical
from AgentWrapper import AgentWrapper



class RaceSimulation: # for single agent testing
    def __init__(self, config):
        self.config = config
        
        track = TrackData()
        self.env = RaceEnv(self.config, track)
        classical = Classical(track, self.env.car)
        rl = None
        self.agent = AgentWrapper(classical, rl, self.env)


    # def clear_test_files(self):
    #     file_path_list = ["EpHistories/", "Plots/", "TrainingImages/"]
    #     for path in file_path_list:
    #         file_path = self.agent_file_path + path
    #         for file_name in os.listdir(os.getcwd() + "/" + file_path):
    #             os.remove(file_path + file_name) # deletes old files
    #             print("File deleted: " + str(file_name))

    def run_agent_training_set(self, num_sets, set_name=""):
        print(set_name)
        # run a training set
        ep_rewards = []
        ep_loss = []
        self.agent.classic.plan_path()
        for i in range(num_sets):
            rewards = self.agent.run_sim()
            ep_rewards.append(rewards)
            plot(ep_rewards, 10, set_name, 2)

            if i % self.config.render_rate == 1 and self.config.render:
                self.env.render_episode(set_name + ":ep_pic")

        plt.figure(2)
        plt.savefig(set_name + ":training.png")

        return ep_rewards

    def run_agent_training(self):
        self.clear_test_files()
        
        self.env.track.straight_track()

        self.run_agent_training_set(300, "Train1: StraightTrack")

        self.env.track.add_obstacle()
        self.run_agent_training_set(15000, "Train2: SingleObstacle")

        self.env.track.add_obstacle()
        self.run_agent_training_set(10000, "Train3: DoubleObstacle")

        # self.env.track.add_obstacle()
        # self.run_agent_training_set(1000, "Train4: TripleObstacle")

    def debug_agent_test(self):
        # self.clear_test_files()
        
        self.env.track.simple_maze()

        self.run_agent_training_set(2, "Debugging...")

    def plan_path(self):
        self.env.track.simple_maze()
        self.agent.classic.plan_path()

 


def plot(values, moving_avg_period, title, figure_n):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    moving_avg = get_moving_average(moving_avg_period * 5, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)
    # print("Episode", (len(values)), "\n", \
    #     moving_avg_period, "episode moving avg:", moving_avg)

def plot_comp(values1, values2, moving_avg_period, title, figure_n):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    # plt.plot(values1)

    moving_avg = get_moving_average(moving_avg_period, values1)
    plt.plot(moving_avg)    

    # plt.plot(values2)
    moving_avg = get_moving_average(moving_avg_period, values2)
    plt.plot(moving_avg)    
 
    plt.legend(['RL Moving Avg', "Classical Moving Avg"])
    # plt.legend(['RL Agent', 'RL Moving Avg', 'Classical Agent', "Classical Moving Avg"])
    plt.pause(0.001)

def get_moving_average(period, values):

    moving_avg = np.zeros_like(values)

    for i, avg in enumerate(moving_avg):
        if i > period:
            moving_avg[i] = np.mean(values[i-period:i])
        # else already zero
    return moving_avg



