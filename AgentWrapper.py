import numpy as np


class AgentWrapper:
    # this agent is what will interface with the env.
    def __init__(self, classical, rl, env):
        # self.rl = Vanilla()
        self.rl = rl
        self.classic = classical 
        self.env = env

    def take_step(self, state):

        # rl_action = self.rl.get_action(state)
        rl_action = 1
        classical_action = self.classic.get_action(state)

        # action = self.add_actions(rl_action, classical_action)

        return classical_action

    def add_actions(self, rl, classic):
        action = rl + classic

        return action

        # theta_swerve = 0.8
        # # interpret action
        # # 0-m : left 
        # # m - straight
        # # m-n : right
        # agent_action += 1 # takes start from zero to 1

        # n_actions_side = (self.action_space -1)/2
        # m = n_actions_side + 1

        # if agent_action < m: # swerve left
        #     swerve = agent_action / n_actions_side * theta_swerve
        #     action = [con_action[0], con_action[1] -  swerve]
        #     dr = 1
        #     # print("Swerving left")
        # elif agent_action == m: # stay in the centre
        #     dr = 0
        #     swerve = 0
        #     action = con_action
        # elif agent_action > m: # swerve right
        #     swerve = (agent_action - m) / n_actions_side * theta_swerve
        #     action = [con_action[0], con_action[1] + swerve]
        #     dr = 1
        #     # print("Swerving right")
        # # print(swerve)
        # return action, dr
    
    def run_sim(self):
        env = self.env
        state, done, rewards = env.reset(), False, 0.0
        self.classic.reset()
        while not done:
            action = self.take_step(state)
            state, reward, done = env.step(action)
            rewards += reward
        print(f"EP complete, reward: {rewards}")

        return rewards



