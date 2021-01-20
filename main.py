import numpy as np
import matplotlib.pyplot as plt
from HistoryStructs import TrainHistory

import timeit

from Simulator import ForestSim
from SimMaps import  ForestMap
from ModelsRL import ReplayBufferDQN, ReplayBufferTD3
import LibFunctions as lib

from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain
from RefGen import GenVehicleTest, GenVehicleTestV, GenVehicleTrainDistance, GenVehicleTrainSteering, GenVehicleTrainVelocity

names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[5]
myMap = 'TrackMap1000'
forest_name = 'forest'


def RunOptimalAgent():
    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    agent = OptimalAgent()

    env_map.reset_map()
    done, state, score = False, env.reset(), 0.0
    wpts = agent.init_agent(env_map)
    env.render(wait=True)
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
    path = 'Vehicles/' + agent_name
    buffer = ReplayBufferTD3()

    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    vehicle = ModVehicleTrain(agent_name, load, 200, 10)

    t_his = TrainHistory(agent_name)
    print_n = 500

    done, state = False, env.reset()
    wpts = vehicle.init_agent(env_map)

    for n in range(10000):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        new_r = vehicle.add_memory_entry(r, done, s_prime, buffer)
        t_his.add_step_data(new_r)

        state = s_prime
        vehicle.agent.train(buffer, 2)
        
        # env.render(False)

        if n % print_n == 0 and n > 0:
            t_his.print_update()
            vehicle.agent.save(directory=path)
        
        if done:
            t_his.lap_done()
            # vehicle.show_vehicle_history()
            env.render(wait=False, save=False)

            vehicle.reset_lap()
            state = env.reset()


    vehicle.agent.save(directory=path)

    return t_his.rewards


"""General test function"""
def testVehicle(vehicle, show=False, obs=True):
    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    crashes = 0
    completes = 0
    lap_times = []

    wpts = vehicle.init_agent(env_map)
    done, state, score = False, env.reset(), 0.0
    for i in range(100): # 10 laps
        print(f"Running lap: {i}")
        # if obs:
        #     env_map.reset_map()
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
            crashes += 1
        else:
            completes += 1
            lap_times.append(env.steps)
        state = env.reset()
        
        # env.reset_lap()
        env.reset()
        vehicle.reset_lap()
        done = False

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {(completes / (completes + crashes) * 100):.2f} %")
    print(f"Lap times: {lap_times} --> Avg: {np.mean(lap_times)}")

"""RefGen Train"""
def TrainGenVehicle(agent_name, load):
    path = 'Vehicles/' + agent_name
    buffer = ReplayBufferTD3()

    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    # vehicle = GenVehicleTrainDistance(agent_name, load, 200, 10)
    vehicle = GenVehicleTrainSteering(agent_name, load, 200, 10)
    # vehicle = GenVehicleTrainVelocity(agent_name, load, 200, 10)


    t_his = TrainHistory(agent_name)
    print_n = 500

    done, state = False, env.reset()
    wpts = vehicle.init_agent(env_map)

    for n in range(10000):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        new_r = vehicle.add_memory_entry(r, done, s_prime, buffer)
        t_his.add_step_data(new_r)

        state = s_prime
        vehicle.agent.train(buffer, 2)
        
        # env.render(False)

        if n % print_n == 0 and n > 0:
            t_his.print_update()
            vehicle.agent.save(directory=path)
        
        if done:
            t_his.lap_done()
            # vehicle.show_vehicle_history()
            env.render(wait=False, save=False)

            vehicle.reset_lap()
            state = env.reset()


    vehicle.agent.save(directory=path)

    return t_his.rewards





"""Total functions"""
def RunModAgent():
    agent_name = "TestingMod"
    
    TrainModVehicle(agent_name, False)
    # TrainModVehicle(agent_name, True)

    # vehicle = ModVehicleTest(agent_name)

    # testVehicle(vehicle, obs=True, show=True)
    # testVehicle(vehicle, obs=False, show=True)

def RunGenAgent():
    agent_name = "TestingGenD"
    # agent_name = "TestingGenV"

    # TrainGenVehicle(agent_name, False)

    vehicle = GenVehicleTest(agent_name)
    # vehicle = GenVehicleTestV(agent_name)

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
    RunGenAgent()
    # RunOptimalAgent()

    # timing()





    
