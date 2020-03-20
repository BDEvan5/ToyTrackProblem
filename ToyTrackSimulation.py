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
import LearningAgent


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
        # self.agent = LearningAgent.AgentQ(3, 5)
        action_space = 3
        self.agent = LearningAgent.AgentLamTD(action_space, 5)
        self.env.action_space = action_space

        self.player = None
        self.ep_mem = None

        self.rewards = []
    
    def run_standard_simulation(self):
        self.path_planner.plan_path()
        self.controller.run_standard_control()
        self.ep_mem = self.env.sim_mem
        self.show_simulation()

    def run_learning_sim(self, episodes):
        # self.path_planner.plan_path()
        self.path_planner.get_single_path()

        max_allow_reward = 500
        best_reward = -1000

        for i in range(episodes):
            ep_reward = self.run_episode()
            print("Episode: %d -> Reward: %d"%(i, ep_reward))

            self.rewards.append(np.min([ep_reward, max_allow_reward]))
            
            self.agent.print_params()
            if ep_reward >= best_reward:
                best_reward = ep_reward
                self.env.sim_mem.save_ep("BestRun")
            
        print("Best rewards: %d" % best_reward)
        self.env.sim_mem.save_ep("Last_ep")

        self.agent.print_q()
        # self.plot_rewards()

    def run_episode(self):
        # this is the code to run a single episode
        max_steps = 200
        ep_reward = 0

        car_state = self.env.reset()
        self.agent.reset_agent()
        # print("ep" + str(type(car_state)))
        agent_action = self.agent.get_action(car_state)
        for i in range(max_steps):
            # agent_action = 1 # do nothing

            next_state, reward, done = self.env.step(agent_action)

            ep_reward += reward
            agent_action = self.agent.update_q_table(car_state, agent_action, reward, next_state)
            car_state = next_state

            if done:
                # print(ep_reward)
                break
        return ep_reward


    def plot_rewards(self):
        i = range(len(self.rewards))
        # print(self.rewards)
        plt.plot(i, self.rewards, 'x')
        plt.show()

    def show_simulation(self):
        self.player = TrackInterfac.ShowInterface(self.track, 100)
        self.player.run_replay(self.ep_mem)


class ReplayEp:
    def __init__(self, track):
        dt = 60
        self.player = TrackInterfac.ShowInterface(track, dt)
        self.ep_history = EpisodeMem.SimMem()


    def replay_last(self):
        self.ep_history.load_ep("Last_ep")
        self.player.run_replay(self.ep_history)

    def replay_best(self):
        self.ep_history.load_ep("BestRun")
        self.player.run_replay(self.ep_history)

