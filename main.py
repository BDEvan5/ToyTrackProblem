import numpy as np
import matplotlib.pyplot as plt
import sys

from Simulator import F110Env, CorridorAction
from RaceMaps import EnvironmentMap
from CommonTestUtils import ReplayBufferDQN, ReplayBufferSuper
import LibFunctions as lib

from OptimalAgent import OptimalAgent
from WillemsPureMod import WillemsVehicle
from MyPureRep import PureRepDataGen, SuperTrainRep


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


"""Training functions: PURE MODE"""
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
    env_map = EnvironmentMap('TrainTrackEmpty')

    env = F110Env(env_map)
    vehicle = WillemsVehicle(env_map, agent_name, env.obs_space, 5, load)
    
    print_n = 100
    rewards = []

    done, state, score = False, env.reset(None), 0.0
    wpts = vehicle.init_straight_plan()
    for n in range(1000):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a, 10)
        # vehicle.add_memory_entry(r, done, s_prime, buffer)
        
        score += r
        # vehicle.agent.train_episodes(buffer)
        vehicle.agent.train_modification(buffer)

        env.render(False, wpts)
        state = s_prime

        if n % print_n == 0 and n > 0:
            rewards.append(score)
            exp = vehicle.agent.model.exploration_rate
            mean = np.mean(rewards)
            b = buffer.size()
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
            score = 0
            # lib.plot(rewards, figure_n=2)

            vehicle.agent.save()
        
        if done:
            env.render_snapshot(wpts=wpts)
            env_map.generate_random_start()
            state = env.reset()
            wpts = vehicle.init_straight_plan()

    vehicle.agent.save()

    return rewards

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
        action = vehicle.act(state)
        s_p, r, done, _ = env.step(action, updates=20)

        nn_state = vehicle.get_nn_vals(state)
        buffer.add((nn_state, action))

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

def create_buffer(load=True):
    if load:
        try:
            buffer = ReplayBufferSuper()
            buffer.load_buffer()
            print(f"Buffer loaded")
        except:
            buffer = generate_data_buffer(500)
        buffer.save_buffer()
        print(f"Buffer generated and saved")

    else:
        buffer = generate_data_buffer()
        buffer.save_buffer()
        print(f"Buffer generated and saved")

    return buffer

def TrainRepAgent(agent_name, load=True):
    buffer = create_buffer(True)

    agent = SuperTrainRep(15, 2, agent_name)

    print_n = 1000
    avg_loss = 0
    for i in range(50000):
        l = agent.train(buffer)
        avg_loss += l

        if i % print_n == 1:
            print(f"It: {i} --> Loss: {avg_loss}")
            agent.save()
            avg_loss = 0

    agent.save()







if __name__ == "__main__":
    # simulation_test()

    agent_name = "TestingWillem"
    # env_map = EnvironmentMap('TestTrack1000')

    # env = F110Env(env_map)
    # vehicle = WillemsVehicle(env_map, agent_name, env.obs_space, 5, False)
    # single_evaluation_vehicle(vehicle)

    # RunWillemModTraining(agent_name, 0, 50, True)
    # RunWillemModTraining(agent_name, 0, 50, False)

    agent_name = "TestingRep"
    TrainRepAgent(agent_name, False)
