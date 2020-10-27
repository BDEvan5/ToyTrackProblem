import numpy as np
import matplotlib.pyplot as plt
import sys, os, shutil

import timeit

from TrackSimulator import TrackSim
from RaceTrackMap import  SimMap, ForestMap
from ModelsRL import ReplayBufferDQN, ReplayBufferTD3
import LibFunctions as lib

from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain

names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[5]
myMap = 'TrackMap1000'
forest_name = 'forest'

def RunOptimalAgent():
    # env_map = SimMap(name)
    env_map = ForestMap(forest_name)

    env = TrackSim(env_map)
    agent = OptimalAgent()

    env_map.reset_map()
    done, state, score = False, env.reset(None), 0.0
    wpts = agent.init_agent(env_map)
    # env.render(True, wpts)
    while not done:
        action = agent.act(state)
        s_p, r, done, _ = env.step(action)
        score += r
        state = s_p

        # env.render(True, wpts)
        # env.env_map.render_map(4, True)
        # env.render(False, wpts)

    print(f"Score: {score}")
    # env.show_history()
    env.render(wait=True)


"""Training functions: PURE MOD"""
def TrainModVehicle(agent_name, load=True):
    buffer = ReplayBufferTD3()
    path = 'Vehicles/' + agent_name 

    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)


    env_map = SimMap(name)
    vehicle = ModVehicleTrain(agent_name, load, 200, 10)

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

            vehicle.agent.save(directory=path)
        
        if done:
            rewards.append(score)
            score = 0
            lengths.append(env.steps)
            vehicle.show_vehicle_history()
            env.render(wait=False)
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


    vehicle.agent.save(directory=path)

    return rewards

"""General test function"""
def testVehicle(vehicle, show=False, obs=True):
    # env_map = SimMap(name)
    env_map = ForestMap(forest_name)
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
            # env.render(False, vehicle.scan_sim)
        print(f"Lap time updates: {env.steps}")
        if show:
            # vehicle.show_vehicle_history()
            env.render(wait=False)
            # env.render(wait=True)

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
    agent_name = "TestingMod"
    
    # TrainModVehicle(agent_name, False)
    # TrainModVehicle(agent_name, True)

    vehicle = ModVehicleTest(agent_name)
    testVehicle(vehicle, obs=True, show=True)
    # testVehicle(vehicle, obs=False, show=True)


def testOptimal():
    agent = OptimalAgent()

    testVehicle(agent, obs=False, show=True)


def timing():
    # t = timeit.timeit(stmt=RunModAgent, number=1)
    # print(f"Time: {t}")
    
    t = timeit.timeit(stmt=testOptimal, number=1)
    print(f"Time: {t}")


if __name__ == "__main__":

    # RunModAgent()
    RunOptimalAgent()

    # timing()





    
