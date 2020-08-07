import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import sys
import pickle
import matplotlib.pyplot as plt 

import LibFunctions as lib 
from TrainEnv import TrainEnv


# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2

class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        y = F.relu(self.l2(x))
        z = self.l3(y)
        a = self.max_action * torch.tanh(z) 
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

class ReplayBufferTD3(object):
    def __init__(self, max_size=100000):     
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.filename = "DataRecords/buffer"

    def add(self, data):        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind: 
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)



class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, agent_name):
        self.agent_name = agent_name
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.act_dim = action_dim
        self.last_action = None
        self.filename = "DataRecords/buffer"

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.max_action, self.max_action)

    def get_new_target(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        a = action.clip(-self.max_action, self.max_action)

        self.last_action = a

        return a

    def add_data(self, obs, new_obs, reward, done):
        data = (obs, new_obs, self.last_action, reward, done)
        self.replay_buffer.add(data)

    def train(self, replay_buffer, iterations=1):
        # iterations = 1 # number of times to train per step
        for it in range(iterations):
            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(1 - d)
            reward = torch.FloatTensor(r)

            self.update_critic(state, action, u, reward, done, next_state)

            # Delayed policy updates
            if it % POLICY_FREQUENCY == 0:
                self.update_actor(state)

    def update_critic(self, state, action, u, reward, done, next_state):
        # Select action according to policy and add clipped noise 
        noise = torch.FloatTensor(u).data.normal_(0, POLICY_NOISE)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * GAMMA * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, state):
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, directory="./saves"):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, self.agent_name))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, self.agent_name))

    def load(self, directory="./saves"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, self.agent_name)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, self.agent_name)))

    # def try_load(self, load=True):
    #     if load:
    #         try:
    #             self.load()
    #         except:
    #             print(f"Unable to load model")
    #             pass
    #     else:
    #         self.actor = Actor(state_dim, action_dim, max_action)
    #         self.actor_target = Actor(state_dim, action_dim, max_action) 


    #         self.model = Qnet(self.obs_space, self.action_space)
    #         self.target = Qnet(self.obs_space, self.action_space)
            # print(f"Not loading - restarting training")

"""Cartpole loops"""
def observe(env,replay_buffer, observation_steps):
    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add((obs, new_obs, action, reward, done))

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

def evaluate_policy(policy, env, eval_episodes=100,render=False):
    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = policy.select_action(np.array(obs), noise=0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

def test():
    env = gym.make("Pendulum-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = env.action_space.high[0]

    agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBufferTD3()

    rewards = []
    observe(env, replay_buffer, 10000)
    for episode in range(500):
        score, done, obs, ep_steps = 0, False, env.reset(), 0
        while not done:
            action = agent.select_action(np.array(obs), noise=0.1)

            new_obs, reward, done, _ = env.step(action) 
            done_bool = 0 if ep_steps + 1 == 200 else float(done)
        
            replay_buffer.add((obs, new_obs, action, reward, done_bool))          
            obs = new_obs
            score += reward
            ep_steps += 1

            agent.train(replay_buffer, 2) # number is of itterations

        rewards.append(score)
        print(f"Ep: {episode} -> score: {score}")


"""My env loops"""
def collect_observations(buffer, observation_steps=5000):
    env = TrainEnv()
    env.pure_rep()

    s, done = env.reset(), False

    for i in range(observation_steps):
        action = env.random_action()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        buffer.add((s, s_p, action, r, done_mask))
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, observation_steps), end="")
        sys.stdout.flush()
    print(" ")


def evaluate_policy(policy, env, eval_episodes=100,render=False):
    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = policy.select_action(np.array(obs), noise=0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

def TrainRepAgent(agent_name, buffer, load=True):
    env = TrainEnv()
    env.pure_rep()

    state_dim = env.state_space
    action_dim = env.action_dim
    max_action = 1 # scale in env

    agent = TD3(state_dim, action_dim, max_action, agent_name)
    # agent.try_load(load)

    rewards, score = [], 0.0
    print_n = 100

    for n in range(1000):
        state = env.reset()

        a = agent.select_action(np.array(state), noise=0.1)
        s_prime, r, done, _ = env.step(a)
        # done_mask = 0.0 if done else 1.0
        done_mask = True
        buffer.add((state, s_prime, a, r, done_mask)) # never done
        agent.train(buffer, 2)
        # score += l
        score += r

        if n % print_n == 0 and n > 0:
            rewards.append(score)
            env.render()    
            mean = np.mean(rewards)
            b = buffer.ptr
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> Buf: {b}")
            score = 0
            lib.plot(rewards, figure_n=2)

            agent.save()
            # test_agent = TestRepDQN(12, 10, agent_name)
            # single_evaluation(test_agent, True)

    return rewards


def RunRepTD3Training(agent_name, start, n_runs, create=False):
    buffer = ReplayBufferTD3()
    total_rewards = []

    evals = []

    if create:
        collect_observations(buffer, 50)
        rewards = TrainRepAgent(agent_name, buffer, False)
        total_rewards += rewards
        lib.plot(total_rewards, figure_n=3)
        # agent = TestRepDQN(12, 10, agent_name)
        # s = single_evaluation(agent)
        # evals.append(s)

    for i in range(start, start + n_runs):
        print(f"Running batch: {i}")
        rewards = TrainRepAgent(agent_name, buffer, True)
        total_rewards += rewards

        lib.plot(total_rewards, figure_n=3)
        plt.figure(2).savefig("PNGs/Training_DQN_rep" + str(i))
        np.save('DataRecords/' + agent_name + '_rewards1.npy', total_rewards)
        # agent = TestRepDQN(12, 10, agent_name)
        # s = single_evaluation(agent)
        # evals.append(s)

    # try:
    #     print(evals)
    #     print(f"Max: {max(evals)}")
    # except:
    #     pass


if __name__ == "__main__":
    # test()

    agent_name = "TestingTD3"

    RunRepTD3Training(agent_name, 0, 5, True)
    # RunRepTD3Training(agent_name, 5, 5, False)


 
    