import TrackEnv1
import Controller1
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = TrackEnv1.RaceTrack()
    myAgent = Controller1.Agent(env)
    myAgent.random_agent()

    print("Finished")