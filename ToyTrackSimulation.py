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