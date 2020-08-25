import numpy as np
import matplotlib.pyplot as plt
import sys

import torch

from Simulator import F110Env, CorridorAction
from RaceMaps import EnvironmentMap
from CommonTestUtils import ReplayBufferDQN, ReplayBufferSuper
import LibFunctions as lib

from OptimalAgent import OptimalAgent
from WillemsPureMod import WillemsVehicle, TrainWillemModDQN
from MyPureRep import PureRepDataGen, SuperTrainRep, SuperRepVehicle


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

def willem_training():
    env_map = EnvironmentMap('TrainTrackEmpty')

    env = F110Env(env_map)
    agent = WillemsVehicle(env_map)

    env_map.generate_random_start()
    done, state, score = False, env.reset(None), 0.0
    wpts = agent.init_agent()
    for n_ep in range(1000):
        while not done:
            action = agent.act(state)
            s_p, r, done, _ = env.step(action, updates=20)
            score += r
            state = s_p

            # env.render(True)
            env.render(False, wpts)

        print(f"Score: {score}")

def single_evaluation_vehicle(vehicle):
    env_map = EnvironmentMap('TestTrack1000')

    env = F110Env(env_map)

    done, state, score = False, env.reset(None), 0.0
    wpts = vehicle.init_plan()
    while not done:
        action = vehicle.act(state)

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
    vehicle = WillemsVehicle(env_map, agent_name, env.obs_space, 5, False)

    done, state, score = False, env.reset(None), 0.0
    wpts = vehicle.init_straight_plan()
    for n in range(n_itterations):
        a = vehicle.random_act(state)
        s_prime, r, done, _ = env.step(a, 50)
        vehicle.add_memory_entry(r, done, s_prime, buffer)
        
        state = s_prime
        
        if done:
            # env.render_snapshot(wpts=wpts)
            env_map.generate_random_start()
            state = env.reset()
            wpts = vehicle.init_straight_plan()

        print("\rPopulating Buffer {}/{}.".format(n, n_itterations), end="")
        sys.stdout.flush()
    print(" ")

def TrainWillemModAgentEps(agent_name, buffer, i=0, load=True):
    env_map = EnvironmentMap('TrainTrackFixed')

    env = F110Env(env_map)
    vehicle = WillemsVehicle(env_map, agent_name, 11, 3, load)
    
    print_n = 500
    rewards = []

    # env_map.set_start()
    # env_map.random_obstacles()
    done, state, score = False, env.reset(None), 0.0
    wpts = vehicle.init_straight_plan()
    for n in range(10000):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a, 20)

        # if n % 4 == 0 or r == -1:
        vehicle.add_memory_entry(r, done, s_prime, buffer)
        
        score += r
        # vehicle.agent.train_episodes(buffer)
        vehicle.agent.train_modification(buffer)

        # env.render(False, wpts)
        state = s_prime

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
            vehicle.show_vehcile_history()
            env.render_snapshot(wpts=wpts, wait=False)
            if r == -1:
                pass
            # env_map.generate_random_start()^
            # env_map.set_start()
            # env_map.random_obstacles() 
            state = env.reset()
            wpts = vehicle.init_straight_plan()

    vehicle.agent.save()

    return rewards

def view_nn(wait=True):
    nn = TrainWillemModDQN(11, 5, "TestingWillem")
    nn.try_load(True)

    phi = [0]
    beams = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 1, 1]
    obs = np.concatenate([phi, beams])
    obs_t = torch.from_numpy(obs).float()
    out = nn.model(obs_t)
    print(f"Out: {out}")

    car_x = 50
    car_y = 10
    fig = plt.figure(4)
    plt.clf()  
    plt.xlim(0, 100)
    plt.ylim(0, 50)

    plt.plot(car_x, car_y, '+', markersize=16)
    targ = [np.sin(phi), np.cos(phi)]
    end = lib.add_locations([car_x, car_y], targ, 40)
    plt.plot(end[0], end[1], 'x', markersize=20)

    dth = np.pi / 9
    for i in range(len(beams)):
        angle = i * dth  - np.pi/2
        fs = beams[i] * 30
        dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
        range_val = lib.add_locations([car_x, car_y], dx)
        x = [car_x, range_val[0]]
        y = [car_y, range_val[1]]
        plt.plot(x, y)

    plt.pause(0.0001)
    if wait:
        plt.show()
        




def RunWillemModTraining(agent_name, start=0, n_runs=5, create=False):
    buffer = ReplayBufferDQN()
    total_rewards = []

    if create:
        # collect_willem_mod_observations(buffer, agent_name, 1000)
        rewards = TrainWillemModAgentEps(agent_name, buffer, 0, False)
        total_rewards += rewards
        # lib.plot(total_rewards, figure_n=3)


    for i in range(start, start + n_runs):
        print(f"Running batch: {i}")
        rewards = TrainWillemModAgentEps(agent_name, buffer, i, True)
        total_rewards += rewards

        # lib.plot(total_rewards, figure_n=3)

def collect_willem_dec_observations(buffer, agent_name, n_itterations=5000):
    env_map = EnvironmentMap('TrainTrackEmpty')

    env = F110Env(env_map)
    vehicle = WillemsVehicle(env_map, agent_name, env.obs_space, 5, False)

    done, state, score = False, env.reset(None), 0.0
    wpts = vehicle.init_straight_plan()
    for n in range(n_itterations):
        a = vehicle.no_mod_act(state)
        s_prime, r, done, _ = env.step(a, 50)
        vehicle.add_decision_entry(r, done, s_prime, buffer)
        
        state = s_prime
        
        if done:
            # env.render_snapshot(wpts=wpts)
            env_map.generate_random_start()
            state = env.reset()
            wpts = vehicle.init_straight_plan()

        print("\rPopulating Buffer {}/{}.".format(n, n_itterations), end="")
        sys.stdout.flush()
    print(" ")

def run_decision_training():
    buffer = ReplayBufferDQN()

    collect_willem_dec_observations(buffer, 'Decisions', 2000)

    agent = TrainWillemDecideDQN(11, 'Decisions')
    agent.try_load(True)

    total_loss = []
    interval = 1000
    for i in range(10000):
        l = agent.train_episodes(buffer)

        if i % interval == 0:
            print(f"Loss: {l}")
            agent.save()
    agent.save()





"""Training functions: PURE REP"""
def generate_data_buffer(b_length=10000):
    env_map = EnvironmentMap('TrainTrackEmpty')

    env = F110Env(env_map)
    vehicle = PureRepDataGen(env_map)

    buffer = ReplayBufferSuper()

    env_map.generate_random_start()
    wpts = vehicle.init_agent()
    done, state, score = False, env.reset(None), 0.0
    # env.render(True, wpts)
    for n in range(b_length):
        action, nn_action = vehicle.act(state)
        s_p, r, done, _ = env.step(action, updates=20)

        nn_state = vehicle.get_nn_vals(state)
        buffer.add((nn_state, [nn_action]))

        state = s_p

        # env.render(False, wpts)

        if done:
            env.render_snapshot(wpts=wpts)
            if r == -1:
                print(f"The vehicle has crashed: check this out")
                plt.show()

            env_map.generate_random_start()
            wpts = vehicle.init_agent()
            state = env.reset()

            print(f"Ep done in {env.steps} steps --> B Number: {n}")

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
    buffer = create_buffer(True)
    # buffer = create_buffer(False)

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


def TestRepAgentTest(agent_name):
    env_map = EnvironmentMap('TestTrack1000')

    env = F110Env(env_map)
    vehicle = SuperRepVehicle(env_map, agent_name, env.obs_space, 1, True)
    
    wpts = vehicle.init_agent()
    done, state, score = False, env.reset(None), 0.0
    # env.render(True, wpts)
    while not done:

        action = vehicle.act(state)
        s_p, r, done, _ = env.step(action, updates=20)
        state = s_p
        env.render(False, wpts)

    env.render_snapshot(wpts=wpts, wait=False)
    if r == -1:
        print(f"The vehicle has crashed: check this out")

def TestRepAgentEmpty(agent_name):
    env_map = EnvironmentMap('TrainTrackEmpty')

    env = F110Env(env_map)
    vehicle = SuperRepVehicle(env_map, agent_name, env.obs_space, 1, True)
    
    env_map.generate_random_start()
    wpts = vehicle.init_agent()
    done, state, score = False, env.reset(None), 0.0
    # env.render(True, wpts)
    while not done:
        action = vehicle.act(state)
        s_p, r, done, _ = env.step(action, updates=20)
        state = s_p
        env.render(False, wpts)

    env.render_snapshot(wpts=wpts, wait=False)
    if r == -1:
        print(f"The vehicle has crashed: check this out")


"""Total functions"""
def WillemsMod():
    agent_name = "TestingWillem"
    RunWillemModTraining(agent_name, 0, 50, True)
    # RunWillemModTraining(agent_name, 0, 50, False)

    # run_decision_training()

    # env_map = EnvironmentMap('TestTrack1000')

    # env = F110Env(env_map)
    # vehicle = WillemsVehicle(env_map, agent_name, env.obs_space, 5, False)
    # single_evaluation_vehicle(vehicle)


def SuperRep():
    agent_name = "TestingRep"

    for i in range(10):
        TestRepAgentTest(agent_name)
        # TestRepAgentEmpty(agent_name)

    # TrainRepAgent(agent_name, False)
    # TrainRepAgent(agent_name, True)




if __name__ == "__main__":
    # simulation_test()
    WillemsMod()
    # SuperRep()

    # view_nn()



    
