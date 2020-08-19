import numpy as np
import matplotlib.pyplot as plt

from Simulator import F110Env, CorridorAction
from RaceMaps import TestMap, EnvironmentMap
from OptimalAgent import OptimalAgent


def simulation_test():
    # race_map = TestMap()
    # race_map.map_1000(True)
    env_map = EnvironmentMap('TestTrack1000')

    env = F110Env(env_map)
    agent = OptimalAgent(env_map)

    done, state, score = False, env.reset(None), 0.0
    wpts = agent.init_agent()
    while not done:
        action = agent.act(state)
        s_p, r, done, _ = env.step(action, updates=20)
        score += r
        state = s_p

        # env.render(True)
        env.render(False, wpts)

    print(f"Score: {score}")

if __name__ == "__main__":
    simulation_test()

