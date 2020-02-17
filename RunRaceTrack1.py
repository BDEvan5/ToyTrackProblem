import TrackEnv1
import Controller1
import TrackInterfac
import matplotlib.pyplot as plt
from tkinter import *
import multiprocessing as mp
import time
import logging

def run_random_agent():
    track_interface = TrackInterfac.Interface(50)
    env = TrackEnv1.RaceTrack(track_interface)
    myAgent = Controller1.RandomAgent(env)

    # set up start and end.
    start_location = [8, 8]
    end_location = [2, 2]
    env.add_locations(start_location, end_location)

    root = mp.Process(target=track_interface.setup_root)
    agent = mp.Process(target=myAgent.random_agent)

    root.start()
    agent.start()

    agent.join()

    root.terminate()


def run_optimal_agent():
    logging.basicConfig(filename="Documents/ToyTrackProblem/AgentLog.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.debug("Hi")

    track_interface = TrackInterfac.Interface(500)
    env = TrackEnv1.RaceTrack(track_interface)
    myAgent = Controller1.OptimalAgent(env, logger)

    # set up start and end.
    start_location = [80.0, 95.0]
    end_location = [20.0, 35.0]
    myAgent.set_locations(start_location, end_location)

    root = mp.Process(target=track_interface.setup_root)
    # agent = mp.Process(target=myAgent.random_agent)
    agent = mp.Process(target=myAgent.opti_agent)

    root.start()
    agent.start()

    agent.join()

    root.terminate()

    
if __name__ == "__main__":
    # run_random_agent()
    run_optimal_agent()







    print("Finished")