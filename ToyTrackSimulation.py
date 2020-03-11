import numpy as np 

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


    def run_control(self):
        self.controller.run_control()
        self.ep_mem = self.env.sim_mem
         # ep_mem.print_ep()

    def show_simulation(self):
        self.player = TrackInterfac.ShowInterface(self.track)
        # myPlayer.show_route()

        self.player.run_replay(self.ep_mem)

    def run_simulation(self):
        self.path_planner.plan_path()
        self.run_control()
        self.show_simulation()

    def run_learning_sim(self, episodes):
        self.path_planner.plan_path()
        for i in range(episodes):
            print("Episode: %d"%i)
            self.controller.run_control()

        # self.controller.agent_q.save_q_table()
        # self.controller.agent_q.load_q_table()
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


