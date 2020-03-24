import numpy as np 
from matplotlib import pyplot as plt
import logging

import RaceEnv 
import Models
import Interface
import LearningAgent

class RaceSimulation:
    def __init__(self, track, car):
        logging.basicConfig(filename="AgentLogger.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.track = track
        self.car = car

        self.env = RaceEnv.RaceEnv(self.track, self.car, self.logger)
        action_space = 3 # todo: move to config file
        self.agent = LearningAgent.AgentLamTD(action_space, 5)
        self.env.action_space = action_space

        self.player = None
        self.ep_mem = None

        self.rewards = []

    def run_learning_sim(self, episodes):
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

    def run_episode(self):
        max_steps = 200
        ep_reward = 0

        car_state = self.env.reset()
        self.agent.reset_agent()

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
        plt.plot(i, self.rewards, 'x')
        plt.show()


