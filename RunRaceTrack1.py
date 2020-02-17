import TrackEnv1
import Controller1
import TrackInterfac
import matplotlib.pyplot as plt
from tkinter import *
import multiprocessing


if __name__ == "__main__":

    track_interface = TrackInterfac.Interface(100)
    env = TrackEnv1.RaceTrack(track_interface)
    myAgent = Controller1.Agent(env)


    root = multiprocessing.Process(target=track_interface.setup_root)
    agent = multiprocessing.Process(target=myAgent.random_agent)

    root.start()
    agent.start()

    agent.join()

    root.terminate()







    print("Finished")