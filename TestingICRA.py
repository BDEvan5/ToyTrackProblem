import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import shutil

from TrackSimulator import TrackSim
from RaceTrackMap import SimMap
from ModelsRL import ReplayBufferTD3
import LibFunctions as lib

from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain

names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[3]
myMap = 'TrackMap1000'

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
        env.render(wait=False)
        # env.render_snapshot(wpts=wpts, wait=True)

    return r, env.steps


def test_vehicles(vehicle_list, laps, eval_name):
    N = len(vehicle_list)

    env_map = SimMap(name)
    env = TrackSim(env_map)

    completes = np.zeros((N))
    crashes = np.zeros((N))
    lap_times = np.zeros((laps, N))
    endings = np.zeros((laps, N)) #store env reward
    lap_times = [[] for i in range(N)]

    for i in range(laps):
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


"""Training Functions"""            
def train_mod(agent_name, recreate=True):
    path = 'Vehicles/' + agent_name + '/'
    csv_path = path + f"TrainingData_{agent_name}.csv"

    if recreate:
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)

    buffer = ReplayBufferTD3()

    env_map = SimMap(name)
    env = TrackSim(env_map)
    vehicle = ModVehicleTrain(agent_name, not recreate, 300, 10) # restart every time

    rewards, lengths, plot_n = [], [], 0

    done, state, score = False, env.reset(None), 0.0
    vehicle.init_agent(env_map)
    env_map.reset_map()
    for n in range(100000):
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
                # env.render(scan_sim=vehicle.scan_sim, wait=False)
                
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


def save_csv_data(rewards, path):
    data = []
    for i in range(len(rewards)):
        data.append([i, rewards[i]])
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)


"""Main functions"""
def main_train():
    mod_name = "ModICRA_build2"

    # train_mod(mod_name, True)
    train_mod(mod_name, False)



def main_test():
    vehicle_list = []

    vehicle_name = "ModICRA_build2"
    mod_vehicle = ModVehicleTest(vehicle_name)
    vehicle_list.append(mod_vehicle)
    
    # vehicle_name = "ModICRA_build"
    # mod_vehicle = ModVehicleTest(vehicle_name)
    # vehicle_list.append(mod_vehicle)

    # opt_vehicle = OptimalAgent()
    # vehicle_list.append(opt_vehicle)

    test_vehicles(vehicle_list, 100, "EvalThree")
    # test_vehicles(vehicle_list, 5, "EvalTwo")


if __name__ == "__main__":


    # main_train()
    main_test()
