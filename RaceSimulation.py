import numpy as np 
from matplotlib import pyplot as plt
import logging
import time

from RaceEnv import RaceEnv
from Models import TrackData, CarModel, Obstacle
from Agent_A2C import Agent_A2C
from Agent_ActionValue import Agent_ActionValue
from AgentMCTS_AV import AgentMCTS_AV, ReplayBuffer

class RaceSimulation: # for single agent testing
    def __init__(self, config):
        self.config = config
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

    def run_agent_training(self):
        # agent = self.agent_ac
        agent = self.agent_av
        agent.clear_test_files()
        standard_car(self.car)
        straight_track(self.track)


        agent.train(self.env, 800, "Train1: StraightTrack", f_test=20)

        add_obstacles(self.env.track, 1)
        agent.train(self.env, 3000, "Train2: SingleObstacle")

        add_obstacles(self.env.track, 1)
        agent.train(self.env, 5000, "Train3: DoubleObstacle")

        add_obstacles(self.env.track, 1)
        agent.train(self.env, 5000, "Train4: TripleObstacle")

    def test_agent(self):
        agent = self.agent_ac
        # agent = self.agent_av
        agent.clear_test_files()
        standard_car(self.car)
        straight_track(self.track)

        agent.train(self.env, 100, "Train1: StraightTrack")

    def run_delta_zero(self):
        replay_buffer = ReplayBuffer(config)

        for i in self.config.training_loops:
            self.agent.run_ep(self.env)
            training_batch = replay_buffer.sample_batch()
            self.agent.train_on_batch(training_batch)


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

# trakcs
def straight_track(myTrack):
    start_location = [50.0, 95.0]
    end_location = [50.0, 10.0]
    o1 = (0, 0, 30, 100)
    o2 = (70, 0, 100, 100)
    o3 = (35, 60, 51, 70)
    o4 = (49, 30, 65, 40)
    b = (1, 1, 99, 99)

    myTrack.add_locations(start_location, end_location)
    myTrack.boundary = b
    myTrack.add_obstacle(o1)
    myTrack.add_obstacle(o2)

def single_corner(myTrack):
    start_location = [80.0, 95.0]
    end_location = [5.0, 20.0]
    o1 = (0, 0, 100, 5)
    o2 = (0, 35, 65, 100)
    o3 = (95, 0, 100, 100)
    b = (1, 1, 99, 99)

    myTrack.add_locations(start_location, end_location)
    myTrack.boundary = b
    myTrack.add_obstacle(o1)
    myTrack.add_obstacle(o2)
    myTrack.add_obstacle(o3)

def simple_maze(myTrack):
    start_location = [95.0, 85.0]
    end_location = [10.0, 10.0]
    o1 = (20, 0, 40, 70)
    o2 = (60, 30, 80, 100)
    b = (1, 1, 99, 99)

    myTrack.add_locations(start_location, end_location)
    myTrack.boundary = b
    myTrack.add_obstacle(o1)
    myTrack.add_obstacle(o2)

def diag_path(myTrack):
    start_location = [95.0, 85.0]
    end_location = [10.0, 10.0]
    # o1 = (20, 0, 40, 70)
    # o2 = (60, 30, 80, 100)
    b = (1, 1, 99, 99)

    myTrack.add_locations(start_location, end_location)
    myTrack.boundary = b
    # myTrack.add_obstacle(o1)
    # myTrack.add_obstacle(o2)

#Cars
def standard_car(myCar):
    max_v = 5

    myCar.set_up_car(max_v)

#Obstacles
def add_obstacles(track, number_to_add=1):
    for i in range(number_to_add):
        o = Obstacle([15, 10])
        o.bounding_box = [40, 20, 60, 80]
        track.add_hidden_obstacle(o)



def double_obstacle(track):
    o2 = (49, 30, 65, 40)
    track.add_hidden_obstacle(o2)

def test_track(track):
    o1 = (35, 30, 51, 40)
    track.add_hidden_obstacle(o1)

    o2 = (49, 60, 65, 70)
    track.add_hidden_obstacle(o2)





