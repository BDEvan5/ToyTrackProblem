import numpy as np 
import logging
import time
from matplotlib import pyplot as plt
import os

from RaceEnv import RaceEnv
from Models import TrackData, CarModel, Obstacle
from Agent_A2C import Agent_A2C
from Agent_ActionValue import Agent_ActionValue, Trainer_AV
from ReplayBuffer import ReplayBuffer
from Networks import Network_AV
from ClassicalAgent import ClassicalAgent


class RaceSimulation: # for single agent testing
    def __init__(self, config):
        self.config = config
        
        # shared resources
        self.buffer = ReplayBuffer(self.config)
        self.network_av_model = Network_AV(self.config)
        self.network_av_model.compile_self()
        self.network_av_target = Network_AV(self.config)
        self.network_av_target.compile_self()

        self.env = RaceEnv(self.config)

        self.agent_av = Agent_ActionValue(self.config, self.network_av_model, self.buffer, self.env)
        self.trainer_av = Trainer_AV(self.config, self.network_av_model)

        self.classical_agent = ClassicalAgent(self.config, self.buffer, self.env)

        self.agent_file_path = "Agent_AV_SimTests/"
        # self.agent_test_path = "Agent_A2C_SimTests/AgentTests/"
        self.weight_path = self.agent_file_path + "ModelWeights/target_weights"

    def clear_test_files(self):
        file_path_list = ["EpHistories/", "Plots/", "TrainingImages/"]
        for path in file_path_list:
            file_path = self.agent_file_path + path
            for file_name in os.listdir(os.getcwd() + "/" + file_path):
                os.remove(file_path + file_name) # deletes old files
                print("File deleted: " + str(file_name))

    def run_agent_training_set(self, num_sets, set_name=""):
        print(set_name)
        # run a training set
        ep_rewards = []
        ep_loss = []
        for i in range(num_sets):
            rewards = self.agent_av.run_sim()
            ep_rewards.append(rewards)
            plot(ep_rewards, 10, set_name, 2)

            minibatch = self.buffer.sample_batch()
            self.trainer_av.train_network(minibatch)

            if i % self.config.network_update == 1:
                self.network_av_model.save_weights(self.weight_path)
                self.network_av_target.load_weights(self.weight_path)

            if i % self.config.render_rate == 1 and self.config.render:
                self.env.render_episode(self.agent_file_path + "TrainingImages/" + set_name + ":%d"%i)

            if i% self.config.test_rate == 1:
                minibatch = self.buffer.sample_batch()
                avg_loss = self.trainer_av.test_network(minibatch)
                ep_loss.append(avg_loss)
                plot(ep_loss, 5, set_name + "Loss", 3)

        plt.figure(2)
        plt.savefig(self.agent_file_path + "Plots/" + set_name + ":training")
        plt.figure(3)
        plt.savefig(self.agent_file_path + "Plots/" + set_name + ":loss")
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
        self.clear_test_files()
        
        self.env.track.straight_track()

        self.run_agent_training_set(2, "Debugging...")

    def run_classical_set(self, num_sets, set_name=""):
        print(set_name)
        # run a training set
        ep_rewards = []
        ep_loss = []
        for i in range(num_sets):
            rewards = self.classical_agent.run_sim()
            ep_rewards.append(rewards)
            plot(ep_rewards, 10, set_name, 2)

            if i % self.config.render_rate == 1 and self.config.render:
                self.env.render_episode(self.agent_file_path + "TrainingImages/" + set_name + ":%d"%i)

            # if i% self.config.test_rate == 1:
            #     minibatch = self.buffer.sample_batch()
            #     avg_loss = self.trainer_av.test_network(minibatch)
            #     ep_loss.append(avg_loss)
            #     plot(ep_loss, 5, set_name + "Loss", 3)

        plt.figure(2)
        plt.savefig(self.agent_file_path + "Plots/" + set_name + ":training")
        # plt.figure(3)
        # plt.savefig(self.agent_file_path + "Plots/" + set_name + ":loss")

        print(set_name + " ->  Complete: %d" %np.mean(ep_rewards))

        return ep_rewards

    def run_mixed_classical_training_set(self, num_sets, set_name=""):
        print(set_name)
        epochs = 10 # number of batch updates per round
        # run a training set
        ep_rewards = []
        ep_loss = []
        for i in range(num_sets):
            rewards = self.classical_agent.run_sim()
            ep_rewards.append(rewards)
            plot(ep_rewards, 10, set_name, 2)

            for j in range(epochs):
                minibatch = self.buffer.sample_batch()  
                self.trainer_av.train_network(minibatch)

            if i % self.config.render_rate == 1 and self.config.render:
                self.env.render_episode(self.agent_file_path + "TrainingImages/" + set_name + ":%d"%i)

            if i% self.config.test_rate == 1:
                minibatch = self.buffer.sample_batch()
                avg_loss = self.trainer_av.test_network(minibatch)
                ep_loss.append(avg_loss)
                plot(ep_loss, 5, set_name + "Loss", 3)

        plt.figure(2)
        plt.savefig(self.agent_file_path + "Plots/" + set_name + ":training")
        plt.figure(3)
        plt.savefig(self.agent_file_path + "Plots/" + set_name + ":loss")

        print(set_name + " ->  Complete: %d" %np.mean(ep_rewards))

        return ep_rewards

    def run_agent_mixed_training(self):
        self.clear_test_files()
        
        self.env.track.straight_track()

        self.run_mixed_classical_training_set(50, "ClassicSet1")

        self.env.track.add_obstacle()
        self.run_mixed_classical_training_set(50, "ClassicSet Obstacle")
        self.run_agent_training_set(50, "AgentSet")

        self.env.track.add_obstacle()
        self.run_mixed_classical_training_set(50, "ClassicSet Two Obstacles")

    def run_classical_agent(self):
        self.clear_test_files()
        
        self.env.track.straight_track()

        self.run_classical_set(50, "ClassicSet1")

        self.env.track.add_obstacle()
        self.run_classical_set(50, "ClassicSet Obstacle")
        self.env.track.add_obstacle()
        self.run_classical_set(50, "ClassicSet Two Obstacles")

    def rerun_old_network(self):
        self.network_av_model.load_weights(self.weight_path)

        self.env.track.straight_track()

        # self.run_agent_training_set(300, "Train1: StraightTrack")

        self.env.track.add_obstacle()
        self.run_agent_training_set(1500, "Train2: SingleObstacle")

    def run_report_comp(self):
        state = self.env.reset()
        rand_predict = self.network_av_model.predict(state)[0]
        print(rand_predict)
        self.network_av_model.load_weights(self.weight_path)
        predict = self.network_av_model.predict(state)[0]
        print(predict)

        self.network_av_model.summary()

        self.env.track.straight_track()
        self.env.track.add_obstacle()

        self.run_comparison_set(100, "Agent Comparison")

    def run_comparison_set(self, num_sets, set_name):
        print(set_name)

        ep_rewards_classic = []
        ep_rewards_av = []
        for i in range(num_sets):
            self.env.track.set_up_random_obstacles()
            rewards = self.agent_av.run_test()
            ep_rewards_av.append(rewards)

            rewards = self.classical_agent.run_sim()
            ep_rewards_classic.append(rewards)

            plot_comp(ep_rewards_av, ep_rewards_classic, 10, set_name, 2)

            if i % self.config.render_rate == 1 and self.config.render:
                self.env.render_episode(self.agent_file_path + "TrainingImages/" + set_name + ":%d"%i)


        plt.figure(2)
        plt.savefig(self.agent_file_path + "Plots/" + set_name + ":comparison")




class AgentComparison: # for testing agents against other agents
    def __init__(self):
        logging.basicConfig(filename="AgentLogger.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.track = TrackData()
        self.car = CarModel()

        self.env = RaceEnv(self.track, self.car, self.logger)
        self.agent_ac = Agent_A2C(self.env.state_space, self.env.action_space)
        self.agent_av = Agent_ActionValue(self.env.state_space, self.env.action_space)
        print("Space sizes: state:%d -> action:%d"%(self.env.state_space, self.env.action_space))

    def train_agents(self):
        self.agent_ac.clear_test_files()
        self.agent_av.clear_test_files()
        standard_car(self.car)
        straight_track(self.track)

        self.agent_ac.train(self.env, 1200, "Train1: StraightTrack")
        self.agent_av.train(self.env, 1200, "Train1: StraightTrack")

        single_obstacle(self.env.track)
        self.agent_ac.train(self.env, 3000, "Train2: SingleObstacle")
        self.agent_av.train(self.env, 3000, "Train2: SingleObstacle")

        double_obstacle(self.env.track)
        self.agent_ac.train(self.env, 5000, "Train3: DoubleObstacle")
        self.agent_av.train(self.env, 5000, "Train3: DoubleObstacle")

    def test_agents(self, env, test_name, num_tests):
        ep_rewards_ac = np.zeros(num_tests)
        ep_rewards_av = np.zeros(num_tests)
        for i in range(num_tests):
            reward_ac = self.agent_ac.test(env, test_name + "->TestNum:%d"%i)
            reward_av = self.agent_av.test(env, test_name + "->TestNum:%d"%i)
            ep_rewards_ac[i] = reward_ac
            ep_rewards_av[i] = reward_av
        
        return ep_rewards_ac, ep_rewards_av

    def run_agent_comparison(self):
        standard_car(self.car)
        straight_track(self.track)
        test_track(self.track)
        #Todo: make list of test envs

        # loads the last saved weights
        self.agent_ac.load_weights()
        self.agent_av.load_weights()

        number_of_tests = 5
        ac_rewards, av_rewards = self.test_agents(self.env, "ComparisonTest", number_of_tests)

        plt.figure(3)
        plt.clf()  
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(ac_rewards)
        plt.plot(av_rewards)
        plt.show()
        plt.savefig("AgentTest")

    



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



