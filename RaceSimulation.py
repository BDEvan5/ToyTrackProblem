import numpy as np 
from matplotlib import pyplot as plt
import logging
import time

from RaceEnv import RaceEnv
from Models import TrackData, CarModel
from Agent import Agent

class RaceSimulation:
    def __init__(self):
        logging.basicConfig(filename="RunFiles/AgentLogger.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.track = TrackData()
        self.car = CarModel()

        self.env = RaceEnv(self.track, self.car, self.logger)
        self.agent = Agent(self.env.state_space, self.env.action_space)
        print("Space sizes: state:%d -> action:%d"%(self.env.state_space, self.env.action_space))

        self.rewards = []

    def run_sim_course(self):
        self.agent.clear_test_files()
        standard_car(self.car)
        straight_track(self.track)

        self.agent.train(self.env, 1200, "Train1: StraightTrack")

        single_obstacle(self.env.track)

        self.agent.train(self.env, 3000, "Train2: SingleObstacle")

        double_obstacle(self.env.track)
        self.agent.train(self.env, 5000, "Train3: DoubleObstacle")


    def test_agent(self):
        standard_car(self.car)
        single_obstacle(self.env.track)
        straight_track(self.track)
        # self.agent.train(self.env, 1, 100)
        self.agent.test(self.env, 1)
        # self.agent.test(self.env, 2)



def straight_track(myTrack):
    start_location = [50.0, 95.0]
    end_location = [50.0, 15.0]
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

def standard_car(myCar):
    max_v = 5

    myCar.set_up_car(max_v)

def single_obstacle(myTrack):
    o1 = (35, 60, 51, 70)
    myTrack.add_hidden_obstacle(o1)

def double_obstacle(track):
    o2 = (49, 30, 65, 40)
    track.add_hidden_obstacle(o2)
