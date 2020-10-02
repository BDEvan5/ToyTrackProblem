import numpy as np
import matplotlib.pyplot as plt

from TrackSimulator import TrackSim
from RaceTrackMap import TrackMap
from CommonTestUtils import ReplayBufferDQN, ReplayBufferSuper, ReplayBufferAuto
import LibFunctions as lib

from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain
from AgentRep import RepTrainVehicle, RepRaceVehicle


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

    print(f"Lap time for({vehicle.name}): {env.steps}")
    if show:
        # vehicle.show_vehicle_history()
        # env.render_snapshot(wpts=wpts, wait=False)
        env.render_snapshot(wpts=wpts, wait=True)

    return r, env.steps


def test_vehicles(vehicle_list, laps=100):
    N = len(vehicle_list)

    env = env_map = TrackMap('TrackMap1000')
    env = TrackSim(env_map)

    completes = np.zeros((N))
    crashes = np.zeros((N))
    lap_times = np.zeros((laps, N))
    endings = np.zeros((laps, N)) #store env reward

    for i in range(laps):
        env_map.reset_map()
        for j in range(N):
            vehicle = vehicle_list[j]

            # r, steps = RunVehicleLap(vehicle, env, False)
            r, steps = RunVehicleLap(vehicle, env, True)
            endings[i, j] = r
            lap_times[i, j] = steps
            if r == -1:
                crashes[j] += 1
            else:
                completes[j] += 1

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
def train_mod(agent_name):
    buffer = ReplayBufferDQN()

    env_map = TrackMap('TrackMap1000')
    vehicle = ModVehicleTrain(agent_name, False) # restart every time

    env = TrackSim(env_map)

    print_n = 500
    plot_n = 0
    rewards, lengths = [], []
    completes, crash_laps = 0, 0
    complete_his, crash_his = [], []

    done, state, score = False, env.reset(None), 0.0
    wpts = vehicle.init_agent(env_map)
    env_map.reset_map()
    for n in range(100000):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        nr = vehicle.add_memory_entry(r, done, s_prime, buffer)
        score += nr
        state = s_prime
        
        vehicle.agent.train_episodes(buffer)

        if n % print_n == 0 and n > 0:
            rewards.append(score)
            exp = vehicle.agent.model.exploration_rate
            mean = np.mean(rewards)
            print(f"#{n} --> Score: {score:.2f} --> Mean: {mean:.2f} --> exp: {exp:.3f}")
            score = 0

            lib.plot(rewards, figure_n=2)

            vehicle.agent.save()
        
        if done:
            lengths.append(env.steps)
            if plot_n % 10 == 0:
                vehicle.show_vehicle_history()
                env.render_snapshot(wpts=wpts, wait=False)

                # # 10 ep moving avg of laps
                # plt.figure(5)
                # plt.clf()
                # plt.title('Crash history vs complete history (10)')
                # plt.plot(crash_his)
                # plt.plot(complete_his)
                # plt.legend(['Crashes', 'Complete'])

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

    plt.figure(2)
    plt.savefig(f"Training Results: {vehicle.name}")


    return rewards





"""Main functions"""
def main_train():
    mod_name = "ModICRA"
    gen_name = "GenICRA"

    train_mod(mod_name)
    # train_gen(gen_name)


def main_test():
    mod_name = "ModICRA"
    gen_name = "GenICRA"

    vehicle_list = []
    
    mod_vehicle = ModVehicleTest(mod_name, True)
    vehicle_list.append(mod_vehicle)

    opt_vehicle = OptimalAgent()
    vehicle_list.append(opt_vehicle)

    # gen_vehicle = RepRaceVehicle(gen_name)
    # vehicle_list.append(gen_vehicle)

    test_vehicles(vehicle_list, 2)


if __name__ == "__main__":

    # RunModAgent()
    # RunRepAgent()
    # RunAutoAgent()
    # RunOptimalControlAgent()
    # RunOptimalAgent()
    # RunFullAgent()

    # main_test()
    main_train()