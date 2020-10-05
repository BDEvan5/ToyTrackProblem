import numpy as np
import matplotlib.pyplot as plt
import sys

import torch

from TrackSimulator import TrackSim
from RaceTrackMap import TrackMap
from ModelsRL import ReplayBufferDQN, ReplayBufferTD3
import LibFunctions as lib

from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain
from AgentRefGen import RefGenVehicleTrain, RefGenVehicleTest


def RunOptimalAgent():
    env_map = TrackMap()

    env = TrackSim(env_map)
    agent = OptimalAgent()

    done, state, score = False, env.reset(None), 0.0
    wpts = agent.init_agent(env_map)
    # env.render(True, wpts)
    while not done:
        action = agent.act(state)
        s_p, r, done, _ = env.step(action)
        score += r
        state = s_p

        # env.render(True)
        # env.render(False, wpts)

    print(f"Score: {score}")
    env.show_history()
    env.render_snapshot(wait=True, wpts=wpts)


"""Training functions: PURE MOD"""
def TrainModVehicle(agent_name, load=True):
    # buffer = ReplayBufferDQN()
    buffer = ReplayBufferTD3()

    env_map = TrackMap('TrackMap1000')
    vehicle = ModVehicleTrain(agent_name, load)

    env = TrackSim(env_map)

    print_n = 500
    plot_n = 0
    rewards, reward_crashes, lengths = [], [], []
    completes, crash_laps = 0, 0
    complete_his, crash_his = [], []

    done, state, score, crashes = False, env.reset(None), 0.0, 0.0
    env_map.reset_map()

    wpts = vehicle.init_agent(env_map)
    for n in range(200000):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        nr = vehicle.add_memory_entry(r, done, s_prime, buffer)
        score += nr
        crashes += r
        state = s_prime
        
        # env.render(False)
        vehicle.agent.train(buffer, 2)

        if n % print_n == 0 and n > 0:
            
            reward_crashes.append(crashes)
            mean = np.mean(rewards)
            b = buffer.size()
            print(f"Run: {n} --> Score: {score:.2f} --> Mean: {mean:.2f} --> ")
            
            lib.plot(rewards, figure_n=2)

            vehicle.agent.save()
        
        if done:
            rewards.append(score)
            score = 0
            lengths.append(env.steps)
            vehicle.show_vehicle_history()
            env.render_snapshot(wpts=wpts, wait=False)
            if plot_n % 10 == 0:

                crash_his.append(crash_laps)
                complete_his.append(completes)
                crash_laps = 0
                completes = 0

            plot_n += 1
            env_map.reset_map()
            vehicle.reset_lap()
            state = env.reset()

            if r == -1:
                crash_laps += 1
            else:
                completes += 1


    vehicle.agent.save()

    return rewards


"""Training for Ref Gen agent"""
def TrainRefGenVehicle(agent_name, load):
    print(f"Training Full Vehicle performance")
    env_map = TrackMap('TrackMap1000')
    buffer = ReplayBufferAuto()

    env = TrackSim(env_map)
    vehicle = RefGenVehicleTrain(agent_name, load)

    # env_map.reset_map()
    wpts = vehicle.init_agent(env_map)
    done, state, score = False, env.reset(None), 0.0
    total_rewards = []
    for i in range(100000):
        action = vehicle.act(state)
        s_p, r, done, _ = env.step(action)
        state = s_p

        n_r = vehicle.add_memory_entry(buffer, r, s_p, done)
        l = vehicle.agent.train(buffer, 2)
        # score += n_r
        score += l
        # env.render(False, wpts)

        if done:
            print(f"#{i}: Ep done in {env.steps} steps --> NewReward: {score} ")
            vehicle.show_history()
            env.render_snapshot(wait=False)
            env.reset()
            total_rewards.append(score)

            plt.figure(5)
            plt.clf()
            plt.title('Total rewards')
            plt.plot(total_rewards)
            score = 0
            vehicle.reset_lap()
            vehicle.agent.save()


"""General test function"""
def testVehicle(vehicle, show=False, obs=True):
    env_map = TrackMap('TrackMap1000')
    env = TrackSim(env_map)

    crashes = 0
    completes = 0
    lap_times = []

    wpts = vehicle.init_agent(env_map)
    done, state, score = False, env.reset(None), 0.0
    for i in range(10): # 10 laps
        print(f"Running lap: {i}")
        if obs:
            env_map.reset_map()
        while not done:
            a = vehicle.act(state)
            s_p, r, done, _ = env.step(a)
            state = s_p
            # env.render(False, wpts)
        print(f"Lap time updates: {env.steps}")
        if show:
            # vehicle.show_vehicle_history()
            env.render_snapshot(wpts=wpts, wait=False)

        if r == -1:
            state = env.reset(None)
            crashes += 1
        else:
            completes += 1
            lap_times.append(env.steps)
        
        env.reset_lap()
        vehicle.reset_lap()
        done = False

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {(completes / (completes + crashes) * 100):.2f} %")
    print(f"Lap times: {lap_times} --> Avg: {np.mean(lap_times)}")



"""Total functions"""
def RunModAgent():
    agent_name = "TestingWillem"
    
    # TrainModVehicle(agent_name, False)
    # TrainModVehicle(agent_name, True)

    vehicle = ModVehicleTrain(agent_name, True)
    testVehicle(vehicle, obs=True, show=True)

def RunRefGenAgent():
    agent_name = "TestingFull"

    TrainRefGenVehicle(agent_name, False)
    # TrainRefGenVehicle(agent_name, True)

    # vehicle = RefGenVehicleTest(agent_name, True)
    # testVehicle(vehicle, True)


if __name__ == "__main__":

    RunModAgent()
    # RunOptimalAgent()
    # RunRefGenAgent()




    
