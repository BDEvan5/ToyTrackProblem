import TrackEnv1
import RL_controller
import Controller1
import TrackInterfac
import matplotlib.pyplot as plt
from tkinter import *
import multiprocessing as mp
import time
import logging
import EpisodeMem

def set_up_env(env):
    o1 = (20, 40, 80, 60)
    env.add_obstacle(o1)

def set_up(env, myPlayer, myAgent):
    start_location = [50.0, 95.0]
    end_location = [50.0, 15.0]
    o1 = (0, 0, 30, 100)
    o2 = (70, 0, 100, 100)
    b = (1, 1, 99, 99)

    env.add_obstacle(o1)
    env.add_obstacle(o2)
    env.add_boundaries(b)
    myAgent.set_locations(start_location, end_location)
    myPlayer.interface.set_locations(start_location, end_location)
    myPlayer.interface.add_obstacle(o1)
    myPlayer.interface.add_obstacle(o2)


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

    track_interface = TrackInterfac.Interface(500)
    env = TrackEnv1.RaceTrack(track_interface)
    # myAgent = Controller1.OptimalAgent(env, logger)
    myAgent = Controller1.StarAOpti(env, logger)

    # set up start and end.
    start_location = [80.0, 95.0]
    end_location = [30.0, 25.0]
    myAgent.set_locations(start_location, end_location)
    add_boundaries(env)
    set_up_env(env)


    root = mp.Process(target=track_interface.setup_root)
    agent = mp.Process(target=myAgent.StarA)
    # agent = mp.Process(target=myAgent.opti_agent)

    root.start()
    agent.start()

    agent.join()

    root.terminate()


def run_rl_agent():
    logging.basicConfig(filename="Documents/ToyTrackProblem/rl_AgentLog.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    track_interface = TrackInterfac.Interface(500)
    env = TrackEnv1.RaceTrack(track_interface, logger)
    myAgent = RL_controller.RL_Controller(env, logger)

    # set up start and end.
    start_location = [75.0, 65.0]
    end_location = [20.0, 35.0]
    myAgent.set_locations(start_location, end_location)
    add_boundaries(env)
    # set_up_env(env)

    # currently running test RL agent
    root = mp.Process(target=track_interface.setup_root)
    agent = mp.Process(target=myAgent.run_test_learning)
    # agent = mp.Process(target=myAgent.opti_agent)

    # root.start()
    agent.start()

    agent.join()
    # root.join()

    # root.terminate()


def run_rl_tube():
    logging.basicConfig(filename="Documents/ToyTrackProblem/rl_AgentLog.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # track_interface = TrackInterfac.Interface(500)
    env = TrackEnv1.RaceTrack(logger)
    myAgent = RL_controller.RL_Controller(env, logger)
    myPlayer = TrackInterfac.ReplayEpisode()

    # set up start and end.
    set_up(env, myPlayer, myAgent)

    myAgent.run_learning()
    ep_mem = myAgent.get_ep_mem()
    # ep_mem.print_ep()

    myPlayer.run_replay(ep_mem)
    


    
if __name__ == "__main__":
    # run_random_agent()
    # run_optimal_agent()
    # run_rl_agent()
    run_rl_tube()







    print("Finished")