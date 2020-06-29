import numpy as np
import matplotlib.pyplot as plt 

import LibFunctions as lib
from TestEnv import TestEnv
from TrainEnv import TrainEnv
from Corridor import CorridorAgent 
from ReplacementDQN import TestDQN, TrainDQN

name00 = 'DataRecords/TrainTrack1000.npy'
name10 = 'DataRecords/TrainTrack1010.npy'
name20 = 'DataRecords/TrainTrack1020.npy'

def evaluate_agent(env, agent, show=True):
    agent.load()
    print_n = 10
    show_n = 10

    rewards = []
    for n in range(100):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.act(state)
            s_prime, _, done, _ = env.step(a)
            state = s_prime
            score += 1 # counts number of steps
            if show:
                # env.box_render()
                pass
            
        rewards.append(score)

        if n % print_n == 1:
            mean = np.mean(rewards[-20:])
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} ")
        if show and n%show_n == 1:
            lib.plot(rewards, figure_n=2)
            plt.figure(2).savefig("Testing_" + agent.name)
            env.render()

def collect_observations(env, agent, n_itterations=10000):
    s, done = env.reset(), False
    for i in range(n_itterations):
        action = env.random_action()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        agent.buffer.append((s, action, r, s_p, done_mask))
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()
    print(" ")


def TrainAgent(agent, env):
    print_n = 20
    rewards = []
    observe_myenv(env, agent, 5000)
    for n in range(5000):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.learning_act(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            agent.buffer.append((state, a, r, s_prime, done_mask))
            state = s_prime
            score += r
            agent.experience_replay()
        rewards.append(score)

        if n % print_n == 1:
            exp = agent.exploration_rate
            mean = np.mean(rewards[-20:])
            b = len(agent.buffer)
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
            agent.save()
            # lib.plot(rewards, figure_n=2)
            # plt.figure(2).savefig("Training_" + agent_name)

    agent.save()
    lib.plot(rewards, figure_n=2)
    plt.figure(2).savefig("Training_DQN")




# run training
def RunDQNTraining():
    env = TrainEnv(name20)
    agent = TrainDQN(env.state_space, env.action_space, "TestDQN")

    DebugAgentTraining(agent, env)
    TrainAgent(agent, env)


# testing algorithms
def RunDQNTest():
    env = TestEnv(name00)
    agent = TestDQN(env.state_space, env.action_space, "TestDQN")

    evaluate_agent(env, agent)

def RunCorridorTest():
    env = TestEnv(name00)

    corridor_agent = CorridorAgent(env.state_space ,env.action_space)

    evaluate_agent(env, corridor_agent, True)


if __name__ == "__main__":
    RunCorridorTest()
    # RunDQNTraining()
