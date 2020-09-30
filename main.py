import numpy as np
import matplotlib.pyplot as plt
import sys

import torch

from TrackSimulator import TrackSim
from RaceTrackMap import TrackMap
from CommonTestUtils import ReplayBufferDQN, ReplayBufferSuper, ReplayBufferAuto
import LibFunctions as lib

from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain
from AgentRep import RepTrainVehicle, RepRaceVehicle
from AgentAuto import AutoTrainVehicle
from AgentFull import FullTrainVehicle


def RunOptimalAgent():
    env_map = TrackMap()

    env = TrackSim(env_map)
    agent = OptimalAgent()

    done, state, score = False, env.reset(None), 0.0
    wpts = agent.init_agent(env_map)
    # env.render(True, wpts)
    while not done:
        action = agent.act(state)
        s_p, r, done, _ = env.step(action, updates=1)
        score += r
        state = s_p

        # env.render(True)
        # env.render(False, wpts)

    print(f"Score: {score}")
    env.show_history()
    env.render_snapshot(wait=True, wpts=wpts)


def RunOptimalControlAgent():
    env_map = TrackMap()

    env = TrackSim(env_map)
    agent = OptimalAgent()

    done, state, score = False, env.reset(None), 0.0
    wpts = agent.init_agent(env_map)
    # env.render(True, wpts)
    while not done:
        action = agent.act_cs(state)
        s_p, r, done, _ = env.step_cs(action)
        score += r
        state = s_p

        # env.render(True)
        env.render(False, wpts)

    print(f"Score: {score}")
    env.show_history()
    env.render_snapshot(wait=True, wpts=wpts)



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

    env_map = TrackMap('TrackMap1000')
    vehicle = ModVehicleTrain(agent_name, load)

    env = TrackSim(env_map)

    print_n = 500
    plot_n = 0
    rewards, reward_crashes, lengths = [], [], []
    completes, crash_laps = 0, 0
    complete_his, crash_his = [], []

    done, state, score, crashes = False, env.reset(None), 0.0, 0.0
    wpts = vehicle.init_agent(env_map)
    for n in range(100000):
        # a = vehicle.act(state)
        # s_prime, r, done, _ = env.step(a, 1)
        a = vehicle.act_cs(state)
        s_prime, r, done, _ = env.step_cs(a)

        nr = vehicle.add_memory_entry(r, done, s_prime, buffer)
        score += nr
        crashes += r
        state = s_prime
        
        # env.render(False)
        vehicle.agent.train_episodes(buffer)

        if n % print_n == 0 and n > 0:
            rewards.append(score)
            reward_crashes.append(crashes)
            exp = vehicle.agent.model.exploration_rate
            mean = np.mean(rewards)
            b = buffer.size()
            print(f"Run: {n} --> Score: {score:.2f} --> Mean: {mean:.2f} --> exp: {exp} --> ")
            score = 0
            lib.plot(rewards, figure_n=2)
            plt.figure(2)
            plt.plot(reward_crashes)
            plt.pause(0.0001)
            crashes = 0

            vehicle.agent.save()
        
        if done:
            lengths.append(env.steps)
            if plot_n % 10 == 0:
                vehicle.show_vehicle_history()
                env.render_snapshot(wpts=wpts, wait=False)

                # 10 ep moving avg of laps
                plt.figure(5)
                plt.clf()
                plt.plot(crash_his)
                plt.plot(complete_his)

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

    plt.figure(1)
    plt.savefig('Figure 1')
    plt.figure(2)
    plt.savefig('Figure 2')
    plt.figure(3)
    plt.savefig('Figure 3')
    plt.figure(4)
    plt.savefig('Figure 4')

    return rewards

def RaceModVehicle(agent_name):
    env_map = TrackMap('TrackMap1000')
    vehicle = ModVehicleTrain(agent_name, True)

    env = TrackSim(env_map)

    crashes = 0
    completes = 0
    lap_times = []

    wpts = vehicle.init_agent(env_map)
    done, state, score = False, env.reset(None), 0.0
    for i in range(100): # 10 laps
        print(f"Running lap: {i}")
        env_map.reset_map()
        # env.render(False, wpts)
        while not done:
            a = vehicle.act_cs(state)
            s_p, r, done, _ = env.step_cs(a)
            state = s_p
            # env.render(False, wpts)
        # env.render(False, wpts)
        print(f"Lap time updates: {env.steps}")
        # vehicle.show_vehicle_history()
        # env.render_snapshot(wpts=wpts, wait=False)

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
    print(f"Completes: {completes} --> {completes / (completes + crashes) * 100} %")
    print(f"Lap times: {lap_times} --> Avg: {np.mean(lap_times)}")


"""Training functions: PURE REP"""
def collect_rep_ep_obs(vehicle, env, buffer):
    # env.env_map.reset_map()
    wpts = vehicle.init_plan(env.env_map)
    done, state, score = False, env.reset(None), 0.0
    while not done:
        action = vehicle.train_act(state)
        vehicle.add_mem_step(buffer, state)
        s_p, r, done, _ = env.step(action, updates=20)

        state = s_p
        # env.render(True)
    
    vehicle.show_history()
    env.render_snapshot(wpts=wpts)
    if r == -1:
        print(f"The vehicle has crashed: check this out")
    print(f"Ep done in {env.steps} steps --> B Number: {buffer.length}")

def RaceRepVehicle(agent_name):
    print(f"Testing vehicle performance")
    env_map = TrackMap('TrackMap1000')

    env = TrackSim(env_map)
    vehicle = RepRaceVehicle(agent_name, True)

    # env_map.reset_map()
    wpts = vehicle.init_plan(env_map)
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
    plt.show()
    print(f"Ep done in {env.steps} steps ")

def TrainRepTrack(agent_name, load=False):
    env_map = TrackMap('TrackMap1000')

    buffer = ReplayBufferSuper()
    env = TrackSim(env_map)
    vehicle = RepTrainVehicle(agent_name, load)

    print(f"Creating data")
    i = 0
    while buffer.length < 1000:
        collect_rep_ep_obs(vehicle, env, buffer)
        vehicle.agent.train(buffer, 100)

        if i % 5 == 0:
            collect_rep_ep_obs(vehicle, env, buffer)
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

            RaceRepVehicle(agent_name)


"""Training for AUTO agent"""
def TrainAutoVehicle(agent_name, load):
    print(f"Training Auto Vehicle performance")
    env_map = TrackMap('TrackMap1000')
    buffer = ReplayBufferAuto()

    env = TrackSim(env_map)
    vehicle = AutoTrainVehicle(agent_name, load)

    # env_map.reset_map()
    wpts = vehicle.init_agent(env_map)
    done, state, score = False, env.reset(None), 0.0
    total_rewards = []
    for i in range(100000):
        # action = vehicle.act(state)
        # s_p, r, done, _ = env.step(action, updates=1)
        action = vehicle.act_cs(state)
        s_p, r, done, _ = env.step_cs(action)
        state = s_p

        n_r = vehicle.add_memory_entry(buffer, r, s_p, done)
        vehicle.agent.train(buffer, 2)
        score += n_r
        # env.render(False, wpts)

        if done:
            print(f"#{i}: Ep done in {env.steps} steps --> NewReward: {score} ")
            vehicle.show_history()
            env.render_snapshot(wpts=wpts, wait=False)
            env.reset()
            total_rewards.append(score)
            plt.figure(5)
            plt.clf()
            plt.plot(total_rewards)
            score = 0
            vehicle.reset_lap()
            vehicle.agent.save()


"""Training for Full agent"""
def TrainFullVehicle(agent_name, load):
    print(f"Training Full Vehicle performance")
    env_map = TrackMap('TrackMap1000')
    buffer = ReplayBufferAuto()

    env = TrackSim(env_map)
    vehicle = FullTrainVehicle(agent_name, load)

    # env_map.reset_map()
    vehicle.init_agent(env_map)
    done, state, score = False, env.reset(None), 0.0
    total_rewards = []
    for i in range(100000):
        # action = vehicle.act(state)
        # s_p, r, done, _ = env.step(action, updates=1)
        action = vehicle.act_cs(state)
        s_p, r, done, _ = env.step_cs(action)
        state = s_p

        n_r = vehicle.add_memory_entry(buffer, r, s_p, done)
        vehicle.agent.train(buffer, 2)
        score += n_r
        # env.render(False, wpts)

        if done:
            print(f"#{i}: Ep done in {env.steps} steps --> NewReward: {score} ")
            vehicle.show_history()
            env.render_snapshot(wait=False)
            env.reset()
            total_rewards.append(score)
            plt.figure(5)
            plt.clf()
            plt.plot(total_rewards)
            score = 0
            vehicle.reset_lap()
            vehicle.agent.save()




"""Total functions"""
def RunModAgent():
    agent_name = "TestingWillem"
    
    # TrainModVehicle(agent_name, False)
    # TrainModVehicle(agent_name, True)

    RaceModVehicle(agent_name)

def RunRepAgent():
    agent_name = "TestingRep"

    TrainRepTrack(agent_name, False)
    # TrainRepVehicle(agent_name, True)

    # RaceRepVehicle(agent_name)

def RunAutoAgent():
    agent_name = "TestingAuto"

    TrainAutoVehicle(agent_name, False)
    # TrainAutoVehicle(agent_name, True)

def RunFullAgent():
    agent_name = "TestingFull"

    # TrainFullVehicle(agent_name, False)
    TrainFullVehicle(agent_name, True)


if __name__ == "__main__":

    RunModAgent()
    # RunRepAgent()
    # RunAutoAgent()
    # RunOptimalControlAgent()
    # RunOptimalAgent()
    # RunFullAgent()



    
