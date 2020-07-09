import numpy as np 
import LibFunctions as lib 
from matplotlib import pyplot as plt
import sys
import collections
import random
import torch

from ReplacementDQN import TrainRepDQN
import sys
import collections


MEMORY_SIZE = 100000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=MEMORY_SIZE)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def memory_sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # todo: move the tensor bit to the agent file, just return lists for the moment.
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)



class MostBasicEnv:
    def __init__(self):
        self.n_ranges = 10
        self.state_space = self.n_ranges + 2
        self.action_space = 10
        self.action_scale = 20

        self.car_x = None
        self.theta = None

        self.steps = 0
        self.memory = []
        self.eps = 0
        self.x_bound = [1, 99]
        self.y_bound = [1, 99]
        
        self.target = None
        self.end = None
        self.start = [50, 0]
        self.last_distance = 50
        self.race_map = np.zeros((100, 100))

        self.ranges = np.zeros(self.n_ranges)
        self.range_angles = np.zeros(self.n_ranges)
        dth = np.pi/(self.n_ranges-1)
        for i in range(self.n_ranges):
            self.range_angles[i] = i * dth - np.pi/2

    def reset(self):
        self.theta = 0
        self.car_x = self.start
        self.race_map = np.zeros((100, 100))
        self._update_target()
        self._locate_obstacles()

        return self._get_state_obs()

    def _update_target(self):
        target_theta = np.random.random() * np.pi - np.pi/ 2# randome angle [-np.ip/2, pi/2]
        self.target = [np.sin(target_theta) , np.cos(target_theta)]
        fs = 50
        # the end is always 50 away in direction of target
        self.end = [self.target[0] * fs + self.car_x[0] , self.target[1] * fs + self.car_x[1]]

    def _locate_obstacles(self):
        n_obs = 3
        xs = np.random.randint(20, 60, (n_obs, 1))
        ys = np.random.randint(4, 30, (n_obs, 1))
        obs_locs = np.concatenate([xs, ys], axis=1)
        # obs_locs = np.random.random((n_obs, 2)) * 100 # [random coord]
        obs_size = [15, 8]

        for obs in obs_locs:
            for i in range(obs_size[0]):
                for j in range(obs_size[1]):
                    x = i + obs[0]
                    y = j + obs[1]
                    self.race_map[x, y] = 1

    def _update_ranges(self):
        step_size = 3
        n_searches = 15
        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            for j in range(n_searches): # number of search points
                fs = step_size * j
                dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
                search_val = lib.add_locations(self.car_x, dx)
                if self._check_location(search_val):
                    break             
            self.ranges[i] = (j) / (n_searches) # gives a scaled val to 1 

    def _get_state_obs(self):
        self._update_ranges()
        rel_target = lib.sub_locations(self.end, self.car_x)
        transformed_target = lib.transform_coords(rel_target, self.theta)
        normalised_target = lib.normalise_coords(transformed_target)
        obs = np.concatenate([normalised_target, self.ranges])

        return obs

    def step(self, action):
        self.memory.append(self.car_x)
        self.steps += 1

        new_x, new_theta = self._x_step_discrete(action)
        # crash = self._check_location(new_x) 
        crash = self._check_line(self.car_x, new_x)
        if not crash:
            self.car_x = new_x
            self.theta = new_theta
        reward, done = self._get_reward(crash, action)
        reward = reward * 0.01 # scale to -1, 1
        obs = self._get_state_obs()

        return obs, reward, done, None

    def _x_step_discrete(self, action):
        # actions in range [0, n_acts) are a fan in front of vehicle
        # no backwards
        fs = self.action_scale
        dth = np.pi / (self.action_space-1)
        angle = -np.pi/2 + action * dth 
        angle += self.theta # for the vehicle offset
        dx = [np.sin(angle)*fs, np.cos(angle)*fs] 
        
        new_x = lib.add_locations(dx, self.car_x)
        
        new_grad = lib.get_gradient(new_x, self.car_x)
        new_theta = np.pi / 2 - np.arctan(new_grad)
        if dx[0] < 0:
            new_theta += np.pi
        if new_theta >= 2*np.pi:
            new_theta = new_theta - 2*np.pi

        return new_x, new_theta

    def _check_location(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if x[1] > self.y_bound[1]:
            return True 

        if self.race_map[int(x[0]), int(x[1])]:
            return True

        return False

    def _check_line(self, start, end):
        n_checks = 5
        dif = lib.sub_locations(end, start)
        diff = [dif[0] / (n_checks), dif[1] / n_checks]
        for i in range(5):
            search_val = lib.add_locations(start, diff, i + 1)
            if self._check_location(search_val):
                return True
        return False

    def random_action(self):
        return np.random.randint(0, self.action_space-1)

    def _get_reward(self, crash, action):
        beta = 0.8 # scale to 
        r_done = 0
        # step_penalty = 5
        max_steps = 1000

        # done
        done = True if self.steps > max_steps else False

        # crash
        if crash:
            r_crash = -100
            return r_crash, True

        # end
        # cur_distance = lib.get_distance(self.car_x, self.end)
        # if cur_distance < 1 + self.action_scale:
        #     return r_done, True

        grad = lib.get_gradient(self.car_x, self.end)
        # try:
        #     grad = [1] / obs[0] # y/x
        # except:
        #     grad = 10000
        angle = np.arctan(grad)
        if angle > 0:
            angle = np.pi - angle
        else:
            angle = - angle
        dth = np.pi / (self.action_space - 1)
        best_action = int(angle / dth)

        d_action = abs(best_action - action) ** 2
        reward = - 5 * d_action

        return reward, done

    def render(self):
        car_x = int(self.car_x[0])
        car_y = int(self.car_x[1])
        fig = plt.figure(4)
        plt.clf()  
        plt.imshow(self.race_map.T, origin='lower')
        plt.xlim(0, 100)
        plt.ylim(-10, 100)
        plt.plot(self.start[0], self.start[1], '*', markersize=12)
        plt.plot(self.end[0], self.end[1], '*', markersize=12)
        plt.plot(self.car_x[0], self.car_x[1], '+', markersize=16)

        for i in range(self.n_ranges):
            angle = self.range_angles[i] + self.theta
            fs = self.ranges[i] * 15 * 3
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations(self.car_x, dx)
            x = [car_x, range_val[0]]
            y = [car_y, range_val[1]]
            plt.plot(x, y)

        
        plt.pause(0.001)


def collect_custom_obs(buffer, n_itterations=5000):
    env = MostBasicEnv()
    for n in range(n_itterations):
        state = env.reset()
        a = env.random_action()
        s_prime, r, done, _ = env.step(a)
        done_mask = 0.0 if done else 1.0
        buffer.put((state, a, r, s_prime, done_mask))
        print("\rPopulating Buffer {}/{}.".format(n, n_itterations), end="")
        sys.stdout.flush()
    print(" ")


def CustomTrainLoop(agent_name, buffer, load=True):
    env = MostBasicEnv()
    agent = TrainRepDQN(env.state_space, env.action_space, agent_name)
    agent.try_load(load)
    
    print_n = 100
    rewards = []
    score = 0.0
    for n in range(10000):
        state = env.reset()
        a = agent.learning_act(state)
        s_prime, r, done, _ = env.step(a)
        done_mask = 0.0 if done else 1.0
        buffer.put((state, a, r, s_prime, done_mask))

        agent.experience_replay(buffer)

        score += r

        if n % print_n == 1:
            env.render()    
            exp = agent.model.exploration_rate
            mean = np.mean(rewards[-20:])
            b = buffer.size()
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
            rewards.append(score) # score is per 500 steps
            score = 0
            lib.plot(rewards, figure_n=2)
    agent.save()

    # lib.plot(rewards, figure_n=2)
    # plt.figure(2).savefig("PNGs/Training_DQN_basic")

    return rewards

def runCustomLoop():
    agent_name = "RepBasicTrain"
    buffer = ReplayBuffer()

    collect_custom_obs(buffer)
    rewards = []

    r = CustomTrainLoop(agent_name, buffer, False)
    rewards += r
    for i in range(10):
        print(f"Running train: {i}")
        r = CustomTrainLoop(agent_name, buffer, True)   
        rewards += r
        lib.plot(rewards, figure_n=3)


if __name__ == "__main__":
    runCustomLoop()

