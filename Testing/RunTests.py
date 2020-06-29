import numpy as np
import matplotlib.pyplot as plt 

import LibFunctions as lib
from TestEnv import TestEnv
from Corridor import CorridorAgent 


def evaluate_agent(env, agent, show=True):
    agent.load()
    print_n = 10
    show_n = 10

    rewards = []
    for n in range(100):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.act(state)
            s_prime, r, done, _ = env.step(a)
            env.box_render()
            done_mask = 0.0 if done else 1.0
            state = s_prime
            score += r
            
        rewards.append(score)

        if show:
            if n % print_n == 1:
                mean = np.mean(rewards[-20:])
                print(f"Run: {n} --> Score: {score} --> Mean: {mean} ")
                lib.plot(rewards, figure_n=2)
                plt.figure(2).savefig("Testing_" + agent.name)
            if n % show_n == 1:
                env.render()



if __name__ == "__main__":
    name00 = 'DataRecords/TrainTrack1000.npy'
    name10 = 'DataRecords/TrainTrack1010.npy'
    env = TestEnv(name00)
    env.show_map()

    corridor_agent = CorridorAgent(env.state_space ,env.action_space)
    # define other agents here

    evaluate_agent(env, corridor_agent)
