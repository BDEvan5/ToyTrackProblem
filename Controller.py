import LocationState as ls
import numpy as np
import LibFunctions as f
import EpisodeMem as em
import TrackEnv1
import random


class Controller:
    def __init__(self, env, logger):
        self.env = env # at the moment the points are stored in the track - consider chanings
        self.logger = logger

        self.dt = 0.5 # controller frequency

        self.state = ls.CarState()
        self.control_sys = ControlSystem()
        self.agent_q = AgentQ(3, 4)

    def run_control(self):
        self.state = self.env.reset()
        ep_reward = 0
        for w_pt in self.env.track.route: # steps through way points
            i = 0

            ret, wp_reward = self.control_to_wp(w_pt)
            ep_reward += wp_reward

            if not ret:
                print("Episode finished : " + str(ep_reward))
                break # it crashed
        return ep_reward

    def run_standard_control(self):
        # this is with no rl
        self.state = self.env.reset()
        ep_reward = 0
        for w_pt in self.env.track.route: # steps through way points
            i = 0
            ret, wp_reward = self.standard_control(w_pt)
            ep_reward += wp_reward
            if not ret:
                print(ep_reward)
                break # it crashed or

    def standard_control(self, wp):
        # no rl for testing
        i = 0
        max_steps = 100

        while not self._check_if_at_target(wp) and i < max_steps: #keeps going till at each pt
            
            action = self.control_sys.get_controlled_action(self.state, wp)
            next_state, reward, done = self.env.step(action)

            obs = self.state.get_sense_observation()
            next_obs = next_state.get_sense_observation()

            if done:
                return False
                # break
            i+=1 
        return True # got to wp

    def control_to_wp(self, wp):
        i = 0
        max_steps = 100
        wp_reward = 0
        while not self._check_if_at_target(wp) and i < max_steps: #keeps going till at each pt
            
            action = self.control_sys.get_controlled_action(self.state, wp)
            agent_action = self.check_action_rl(action)
            action, dr = self.get_new_action(agent_action, action)

            next_state, reward, done = self.env.step(action)

            reward += dr
            wp_reward += reward
            obs = self.state.get_sense_observation()
            next_obs = next_state.get_sense_observation()
            self.agent_q.update_q_table(obs, agent_action, reward, next_obs)

            self.state = next_state
            if done:
                return False, wp_reward
            i+=1 
        return True, wp_reward # got to wp

    def _check_if_at_target(self, wp):
        way_point_dis = f.get_distance(self.state.x, wp.x)
        # print(way_point_dis)
        if way_point_dis < 2:
            # print("Way point reached: " + str(wp.x))
            return True
        return False
    
    def check_action_rl(self, action):
        observation = self.state.get_sense_observation()
        agent_action = self.agent_q.get_action(observation)
        return agent_action

    def get_new_action(self, agent_action, action):
        theta_swerve = 0.8
        # interpret action
        dr = -20
        if agent_action == 1: # stay in the centre
            dr = 0
            return action, dr 
        if agent_action == 0: # swerve left
            action = [action[0], action[1] - theta_swerve]
            # print("Swerving left")
            return action, dr
        if agent_action == 2: # swerve right
            action = [action[0], action[1] + theta_swerve]
            # print("Swerving right")
            return action, dr




