import numpy as np 
from matplotlib import pyplot as plt

import TrackEnv1
# import RL_controller
import TrackInterfac
import matplotlib.pyplot as plt
from tkinter import *
import multiprocessing as mp
import time
import logging
import EpisodeMem
import GlobalOpti
import Controller
import PathSmoothing
import PathPlanner


class RacingSimulation:
    def __init__(self, track, car):
        logging.basicConfig(filename="Documents/ToyTrackProblem/AgentLog.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.track = track
        self.car = car

        self.path_planner = PathPlanner.PathPlanner(self.track, self.car, self.logger)
        self.env = TrackEnv1.RaceEnv(self.track, self.car, self.logger)
        self.controller = Controller.Controller(self.env, self.logger)

        self.player = None
        self.ep_mem = None

        self.rewards = []

    def run_control(self):
        self.controller.run_control()
        self.ep_mem = self.env.sim_mem
         # ep_mem.print_ep()

    def show_simulation(self):
        self.player = TrackInterfac.ShowInterface(self.track, 100)
        # myPlayer.show_route()

        self.player.run_replay(self.ep_mem)

    def run_standard_simulation(self):
        self.path_planner.plan_path()
        self.controller.run_standard_control()
        self.ep_mem = self.env.sim_mem
        self.show_simulation()

    def run_learning_sim(self, episodes):
        self.path_planner.plan_path()
        max_allow_reward = 500
        for i in range(episodes):
            print("Episode: %d"%i)
            ep_reward = self.controller.run_control()
            self.rewards.append(np.min([ep_reward, max_allow_reward]))

        self.plot_rewards()

        print(self.controller.agent_q.q_table)
        self.ep_mem = self.env.get_ep_mem()
        self.show_simulation()

    def show_sim(self):
        self.path_planner.plan_path()
        self.controller.agent_q.load_q_table()
        self.controller.run_control()
        print(self.controller.agent_q.q_table)
        self.ep_mem = self.env.sim_mem
        self.show_simulation()

    def plot_rewards(self):
        i = range(len(self.rewards))
        # print(self.rewards)
        plt.plot(i, self.rewards)
        plt.show()
