import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import shutil

from TrackSimulator import TrackSim
from RaceTrackMap import SimMap, ForestMap
from ModelsRL import ReplayBufferTD3
import LibFunctions as lib

from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain

names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[5]
myMap = 'TrackMap1000'
forest_name = 'forest'

"""Testing Function"""
def RunVehicleLap(vehicle, env, show=False):
    vehicle.reset_lap()
    wpts = vehicle.init_agent(env.env_map)
    done, state, score = False, env.reset(None), 0.0
    while not done:
        a = vehicle.act(state)
        s_p, r, done, _ = env.step(a)
        state = s_p
        # env.render(False, wpts)

    if show:
        # vehicle.show_vehicle_history()
        # env.render(wait=False)
        env.render(wait=True)

    return r, env.steps

def test_vehicles(vehicle_list, laps, eval_name, add_obs):
    N = len(vehicle_list)

    # env_map = SimMap(name)
    env_map = ForestMap(forest_name)
    env = TrackSim(env_map)

    completes = np.zeros((N))
    crashes = np.zeros((N))
    lap_times = np.zeros((laps, N))
    endings = np.zeros((laps, N)) #store env reward
    lap_times = [[] for i in range(N)]

    for i in range(laps):
        if add_obs:
            env_map.reset_map()
        for j in range(N):
            vehicle = vehicle_list[j]

            # r, steps = RunVehicleLap(vehicle, env, False)
            r, steps = RunVehicleLap(vehicle, env, True)
            print(f"#{i}: Lap time for ({vehicle.name}): {env.steps} --> Reward: {r}")
            endings[i, j] = r
            if r == -1:
                crashes[j] += 1
            else:
                completes[j] += 1
                lap_times[j].append(steps)

    test_name = 'Vehicles/' + eval_name + '.txt'
    with open(test_name, 'w') as file_obj:
        file_obj.write(f"\nTesting Complete \n")
        file_obj.write(f"Map name: {name} \n")
        file_obj.write(f"-----------------------------------------------------\n")
        file_obj.write(f"-----------------------------------------------------\n")
        for i in range(N):
            file_obj.write(f"Vehicle: {vehicle_list[i].name}\n")
            file_obj.write(f"Crashes: {crashes[i]} --> Completes {completes[i]}\n")
            percent = (completes[i] / (completes[i] + crashes[i]) * 100)
            file_obj.write(f"% Finished = {percent:.2f}\n")
            file_obj.write(f"Avg lap times: {np.mean(lap_times[i])}\n")
            file_obj.write(f"-----------------------------------------------------\n")


    print(f"\nTesting Complete ")
    print(f"-----------------------------------------------------")
    print(f"-----------------------------------------------------")
    for i in range(N):
        print(f"Vehicle: {vehicle_list[i].name}")
        print(f"Crashes: {crashes[i]} --> Completes {completes[i]}")
        percent = (completes[i] / (completes[i] + crashes[i]) * 100)
        print(f"% Finished = {percent:.2f}")
        print(f"Avg lap times: {np.mean(lap_times[i])}")
        print(f"-----------------------------------------------------")

def test_mod(vehicle_name):
    vehicle_list = []

    mod_vehicle = ModVehicleTest(vehicle_name)
    vehicle_list.append(mod_vehicle)

    opt_vehicle = OptimalAgent()
    vehicle_list.append(opt_vehicle)
    
    test_vehicles(vehicle_list, 100, vehicle_name + "/Eval_Obs" , True)

    # opt_vehicle = OptimalAgent()
    # vehicle_list.append(opt_vehicle)

    test_vehicles(vehicle_list, 10, vehicle_name + "/Eval_NoObs", False)


"""Training Functions"""            
def train_mod(agent_name, recreate=True):
    path = 'Vehicles/' + agent_name + '/'
    csv_path = path + f"TrainingData_{agent_name}.csv"

    if recreate:
        print(f"Recreating path")
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)

    buffer = ReplayBufferTD3()

    # env_map = SimMap(name)
    env_map = ForestMap(forest_name)
    env = TrackSim(env_map)
    vehicle = ModVehicleTrain(agent_name, not recreate, 300, 10) # restart every time

    rewards, lengths, plot_n = [], [], 0

    done, state, score = False, env.reset(None), 0.0
    vehicle.init_agent(env_map)
    env_map.reset_map()
    for n in range(50000):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        nr = vehicle.add_memory_entry(r, done, s_prime, buffer)
        score += nr
        # score += r
        state = s_prime
        # env.render(False, vehicle.scan_sim)
        vehicle.agent.train(buffer, 2)

        if done:
            rewards.append(score)
            lengths.append(env.steps)
            if plot_n % 10 == 0:
                # vehicle.show_vehicle_history()
                env.render(scan_sim=vehicle.scan_sim, wait=False)
                
                mean = np.mean(rewards)
                print(f"#{n} --> Score: {score:.2f} --> Mean: {mean:.2f} ")

                lib.plot(rewards, figure_n=2)
                vehicle.agent.save(directory=path)
                save_csv_data(rewards, csv_path)

            score = 0
            plot_n += 1
            env_map.reset_map()
            vehicle.reset_lap()
            state = env.reset()

    vehicle.agent.save(directory=path)


    plt.figure(2)
    plt.savefig(path + f"TrainingPlot_{vehicle.name}")

    save_csv_data(rewards, csv_path)

    return rewards


"""Helpers"""
def save_csv_data(rewards, path):
    data = []
    for i in range(len(rewards)):
        data.append([i, rewards[i]])
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

def get_moving_avg(vehicle_name, show=False):
    path = 'Vehicles/' + vehicle_name + f"/TrainingData_{vehicle_name}.csv"
    smoothpath = 'Vehicles/' + vehicle_name + f"/TrainingData.csv"
    rewards = []
    with open(path, 'r') as csvfile:
        csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
        for lines in csvFile:  
            rewards.append(lines)
    rewards = np.array(rewards)[:, 1]

    smooth_rewards = lib.get_moving_average(50, rewards)

    new_rewards = []
    l = 10
    N = int(len(smooth_rewards) / l)
    for i in range(N):
        avg = np.mean(smooth_rewards[i*l:(i+1)*l])
        new_rewards.append(avg)
    smooth_rewards = np.array(new_rewards)

    save_csv_data(smooth_rewards, smoothpath)

    if show:
        lib.plot_no_avg(rewards, figure_n=1)
        lib.plot_no_avg(smooth_rewards, figure_n=2)
        plt.show()




"""Main functions"""

def main():
    vehicle_name = "ModICRA_forest2"

    # train_mod(vehicle_name, True)
    # train_mod(vehicle_name, False)

    test_mod(vehicle_name)

if __name__ == "__main__":


    main()

    # get_moving_avg()
