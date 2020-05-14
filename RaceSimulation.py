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
from PathOptimisation import add_velocity
from TrajectoryOptimisation import optimise_trajectory
from PathTracker import Tracker

# from ClassicalAgent import Classical
# from AgentWrapper import AgentWrapper



class RaceSimulation: # for single agent testing
    def __init__(self, config):
        self.config = config
        
        track = TrackData()
        self.env = RaceEnv(self.config, track)
        classical = Classical(track, self.env.car)
        rl = None
        self.agent = AgentWrapper(classical, rl, self.env)
        
  
    def run_agent_training_set(self, num_sets, set_name=""):
        print(set_name)
        # run a training set
        ep_rewards = []
        ep_loss = []
        self.agent.get_path_plan()
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
        self.agent.get_path_plan()


def simulation_runner(config):
    load = True
    # load = False
    filename = "path_obj.npy"
    db_file = open(filename, 'br+')
    track = TrackData()
    track.simple_maze()

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

    # path_obj.show()

    for i, pt in enumerate(path_obj.route):
        print(i)
        pt.print_point()

    # run sim
    env = RaceEnv(config, track)
    state, done = env.reset(), False
    tracker = Tracker(path_obj)

    # while not done
        # get ref action for state
        # take action and get next state
    while not done:
        state.print_point("State")
        ref_action = tracker.act(state)
        state, reward, done = env.step(ref_action)
        print(f"Action: [{ref_action[0]:.2f} ; {ref_action[1]:.2f}]")

    env.render_episode("PathTracker")

 


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


if __name__ == "__main__":
    config = create_sim_config()
    simulation_runner(config)
