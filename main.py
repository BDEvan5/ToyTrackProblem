import numpy as np
import matplotlib.pyplot as plt
import sys

import torch

from Simulator import F110Env, CorridorAction
from RaceMaps import EnvironmentMap
from RaceTrackMaps import RaceMap
from CommonTestUtils import ReplayBufferDQN, ReplayBufferSuper
import LibFunctions as lib

from OptimalAgent import OptimalAgent
from WillemsPureMod import ModRaceVehicle, ModTrainVehicle
from MyPureRep import SuperTrainRep, SuperRepVehicle, RaceRepVehicle


def simulation_test():
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


"""Training functions: PURE MOD"""
def collect_willem_mod_observations(buffer, agent_name, n_itterations=5000):
    env_map = EnvironmentMap('TrainTrackEmpty')
    env = F110Env(env_map)
    vehicle = ModTrainVehicle(agent_name, env.obs_space, 3, False)

    env_map.reset_map()
    done, state, score = False, env.reset(None), 0.0
    wpts = vehicle.init_plan(env_map)
    for n in range(n_itterations):
        a = vehicle.random_act(state)
        s_prime, r, done, _ = env.step(a, 20)
        vehicle.add_memory_entry(r, done, s_prime, buffer)
        state = s_prime
        
        if done:
            env_map.reset_map()
            state = env.reset()
            wpts = vehicle.init_plan()

        print("\rPopulating Buffer {}/{}.".format(n, n_itterations), end="")
        sys.stdout.flush()
    print(" ")

def TrainModVehicle(agent_name, load=True):
    buffer = ReplayBufferDQN()
    # if not load:
    #     collect_willem_mod_observations(buffer, agent_name, 1000)

    # normal train env
    # env_map = EnvironmentMap('TrainTrackEmpty')
    # vehicle = ModTrainVehicle(agent_name, 11, 3, load)

    # race train env
    env_map = RaceMap('RaceTrack1000')
    vehicle = ModRaceVehicle(agent_name, 11, 3, load)

    env = F110Env(env_map)

    
    print_n = 500
    rewards = []

    env_map.reset_map()
    done, state, score = False, env.reset(None), 0.0
    wpts = vehicle.init_plan(env_map)
    for n in range(10000):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a, 20, race=True)

        vehicle.add_memory_entry(r, done, s_prime, buffer)
        score += r
        state = s_prime
        
        vehicle.agent.train_episodes(buffer)

        if n % print_n == 0 and n > 0:
            rewards.append(score)
            exp = vehicle.agent.model.exploration_rate
            mean = np.mean(rewards)
            b = buffer.size()
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
            score = 0
            lib.plot(rewards, figure_n=2)

            vehicle.agent.save()
        
        if done:
            vehicle.show_vehicle_history()
            env.render_snapshot(wpts=wpts, wait=False)
            if r == -1:
                pass
            env_map.reset_map()
            vehicle.reset_lap()
            state = env.reset()
            wpts = vehicle.init_plan()

    vehicle.agent.save()

    return rewards


def RaceModVehicle(agent_name):
    env_map = RaceMap('RaceTrack1000')
    vehicle = ModRaceVehicle(agent_name, 11, 3, True)
    env = F110Env(env_map)

    wpts = vehicle.init_plan(env_map)
    done, state, score = False, env.reset(None), 0.0
    for i in range(10): # 10 laps
        print(f"Running lap: {i}")
        env_map.reset_map()
        while not done:
            action = vehicle.act(state, True)
            s_p, r, done, _ = env.step(action, updates=20, race=True)

            state = s_p
            # env.render(False, wpts)

        print(f"Lap time updates: {env.steps}")
        vehicle.show_vehicle_history()
        env.render_snapshot(wpts=wpts, wait=False)

        if r == -1:
            state = env.reset(None)
            done = False
        
        env.reset_lap()
        vehicle.reset_lap()
        done = False

 









"""Training functions: PURE REP"""
def generate_data_buffer(b_length=10000):
    env_map = EnvironmentMap('TrainTrackEmpty')

    env = F110Env(env_map)
    vehicle = SuperRepVehicle(env_map)

    buffer = ReplayBufferSuper()

    env_map.generate_random_start()
    wpts = vehicle.init_agent()
    done, state, score = False, env.reset(None), 0.0
    # env.render(True, wpts)
    for n in range(b_length):
        action = vehicle.opti_act(state)
        vehicle.add_mem_step(buffer, state)
        s_p, r, done, _ = env.step(action, updates=20)

        state = s_p

        # env.render(False, wpts)

        if done:
            env.render_snapshot(wpts=wpts)
            if r == -1:
                print(f"The vehicle has crashed: check this out")
                plt.show()

            print(f"Ep done in {env.steps} steps --> B Number: {n}")
            env_map.generate_random_start()
            wpts = vehicle.init_agent()
            state = env.reset()


    return buffer

def create_buffer(load=True, n_steps=1000):
    if load:
        try:
            buffer = ReplayBufferSuper()
            buffer.load_buffer()
            print(f"Buffer loaded")
        except:
            print(f"Load error")

    else:
        buffer = generate_data_buffer(n_steps)
        buffer.save_buffer()
        print(f"Buffer generated and saved")

    return buffer

def TrainRepAgent(agent_name, load):
    # buffer = create_buffer(True)
    buffer = create_buffer(False)

    agent = SuperTrainRep(11, 1, agent_name)
    agent.try_load(load)

    print_n = 100
    test_n = 1000
    avg_loss = 0
    for i in range(50000):
        l = agent.train(buffer)
        avg_loss += l

        # print(f"{i}-> loss: {l}")

        if i % print_n == 0:
            print(f"It: {i} --> Loss: {avg_loss}")
            agent.save()
            avg_loss = 0

        # if i % test_n == 0:
        #     TestRepAgentEmpty(agent_name)
        #     TestRepAgentTest(agent_name)

    agent.save()


# new rep train functions
def run_ep(vehicle, env, buffer):
    # env.env_map.generate_random_start()
    env.env_map.set_start()
    wpts = vehicle.init_agent()
    done, state, score = False, env.reset(None), 0.0
    while not done:
        action = vehicle.opti_act(state)
        vehicle.add_mem_step(buffer, state)
        s_p, r, done, _ = env.step(action, updates=20)

        state = s_p
        # env.render()
    
    vehicle.show_history()
    env.render_snapshot(wpts=wpts)
    if r == -1:
        print(f"The vehicle has crashed: check this out")
        # plt.show()
    print(f"Ep done in {env.steps} steps --> B Number: {buffer.length}")

def test_rep(vehicle):
    print(f"Testing vehicle performance")
    store_map = vehicle.env_map
    env_map = EnvironmentMap('TrainTrackEmpty')

    env = F110Env(env_map)
    vehicle.env_map = env_map

    env_map.generate_random_start()
    wpts = vehicle.init_agent()
    done, state, score = False, env.reset(None), 0.0
    while not done:
        action = vehicle.act(state)
        s_p, r, done, _ = env.step(action, updates=20)
        state = s_p
        # env.render(False, wpts)

    vehicle.show_history()
    env.render_snapshot(wpts=wpts)
    if r == -1:
        print(f"The vehicle has crashed: check this out")
        # plt.show()
    print(f"Ep done in {env.steps} steps ")

    vehicle.env_map = store_map

def RunRepTrain(agent_name, load):
    # env_map = EnvironmentMap('TrainTrackEmpty')
    env_map = RaceMap('RaceTrack1000')
    env = F110Env(env_map)
    vehicle = SuperRepVehicle(env_map, agent_name, load)

    buffer = ReplayBufferSuper()

    print(f"Creating data")
    # for i in range(100):
    i = 0
    while buffer.length < 1000:
        run_ep(vehicle, env, buffer)

        vehicle.agent.train(buffer, 100)

        if i % 5 == 0:
            test_rep(vehicle)

        i += 1

    vehicle.agent.save()

    print(f"Starting training")
    score, losses = 0, []
    for n in range(10000):
        l = vehicle.agent.train(buffer)
        score += l 

        if n % 100 == 1:
            losses.append(score)
            vehicle.agent.save()
            print(f"Loss: {score}")
            score = 0

            test_rep(vehicle)



def TestRepAgentTest(agent_name):
    env_map = EnvironmentMap('TestTrack1000')

    env = F110Env(env_map)
    vehicle = SuperRepVehicle(env_map, agent_name)
    
    wpts = vehicle.init_agent()
    done, state, score = False, env.reset(None), 0.0
    # env.render(True, wpts)
    while not done:

        action = vehicle.act(state)
        s_p, r, done, _ = env.step(action, updates=20)
        state = s_p
        env.render(False, wpts)

    env.render_snapshot(wpts=wpts, wait=True)
    if r == -1:
        print(f"The vehicle has crashed: check this out")

def TestRepAgentEmpty(agent_name):
    env_map = EnvironmentMap('TrainTrackEmpty')

    env = F110Env(env_map)
    vehicle = SuperRepVehicle(env_map, agent_name, True)
    
    env_map.generate_random_start()
    wpts = vehicle.init_agent()
    # wpts = vehicle.init_straight_plan()
    done, state, score = False, env.reset(None), 0.0
    # env.render(True, wpts)
    while not done:
        action = vehicle.act(state)
        s_p, r, done, _ = env.step(action, updates=20)
        state = s_p
        env.render(False, wpts)

    vehicle.show_history()
    env.render_snapshot(wpts=wpts, wait=True)
    if r == -1:
        print(f"The vehicle has crashed: check this out")

"""Race functions"""
def RaceRepTime(agent_name):
    env_map = RaceMap('RaceTrack1000')
    vehicle = RaceRepVehicle(env_map, agent_name, True)
    env = F110Env(env_map)

    # env_map.random_obstacles() # coming later
    wpts = vehicle.init_race_plan()
    done, state, score = False, env.reset(None), 0.0
    for i in range(10): # 10 laps
        print(f"Running lap: {i}")
        while not done:
            action = vehicle.act(state)
            s_p, r, done, _ = env.step(action, updates=20, race=True)

            state = s_p
            env.render(False, wpts)

        print(f"Lap time updates: {env.steps}")

        vehicle.show_history()
        env.render_snapshot(wpts=wpts, wait=True)

        vehicle.reset_lap_count()
        env.reset_lap()




"""Total functions"""
def WillemsMod():
    agent_name = "TestingWillem"
    
    # TrainModVehicle(agent_name, False)
    # TrainModVehicle(agent_name, True)

    RaceModVehicle(agent_name)

def SuperRep():
    agent_name = "TestingRep"

    for i in range(10):
        pass
        # TestRepAgentTest(agent_name)
        # TestRepAgentEmpty(agent_name)

    # TrainRepAgent(agent_name, False)
    # TrainRepAgent(agent_name, True)

    # RunRepTrain(agent_name, False)
    # RunRepTrain(agent_name, True)

    RaceRepTime(agent_name)

if __name__ == "__main__":
    # simulation_test()

    WillemsMod()
    # SuperRep()

    # view_nn()



    
