import numpy as np 
from ValueAgent import BufferVanilla
from PathTracker import ControlSystem 
import LibFunctions as f
from TrackMapInterface import render_track_ep, snap_track


class NewRunner:
    def __init__(self, env, model, path_obj, nsteps=64):
        self.env = env
        self.model = model
        self.path_obj = path_obj
        self.nsteps = nsteps

        self.ep_rewards = [0.0]
        self.state = self.env.reset()

        self.path = path_obj.route
        self.n_inds = len(self.path) -1

        self.pind = 0
        self.wp_done = 0
        self.control_system = ControlSystem()

    def run_batch(self, track): 
        print(f"Running batch")
        f_show = 20
        b, reward = BufferVanilla(), 0
        env, state = self.env, self.state
        reward = 0
        while len(b) <= self.nsteps:
            destination, nn_state = self.determine_destination(state)
            control_action = self.control_system(state[0:4], destination)
            nn_action, nn_value = self.model(nn_state)
            
            # ref_action = self.add_actions(control_action, nn_action)
            ref_action = control_action

            env.car_state.crash_chance = (nn_value) #UNnEAT
            next_state, reward, done = env.step(ref_action)

            self.store_outputs(b, nn_state, nn_action, nn_value, reward, done)         

            if done:
                f.plot(self.ep_rewards, figure_n=5)
                self.ep_rewards.append(0.0)
                print("Episode: %03d, Reward: %03d" % (len(self.ep_rewards) - 1, self.ep_rewards[-2]))

                if len(self.ep_rewards) % f_show == 1:
                    render_track_ep(track, self.path_obj, env.sim_mem, pause=False, dt=30)
                    # snap_track(track, self.path_obj, env.sim_mem)
                next_state = env.reset()
                self.pind = 0

            state = next_state

        self.state = next_state # remember for next batch
        _, b.last_q_val = self.model(nn_state)
        # f.plot_comp(b.values, b.rewards, figure_n=4)
        # render_track_ep(track, self.path_obj, env.sim_mem, pause=True)
        
        return b

    def run_training_batch(self, track): 
        print(f"Running batch")
        b, reward = BufferVanilla(), 0
        env, state = self.env, self.state
        reward = 0
        while len(b) <= self.nsteps:
            destination, nn_state = self.determine_destination(state)
            control_action = self.control_system(state[0:4], destination)
            nn_action, nn_value = self.model(nn_state)
            
            ref_action = self.add_actions(control_action, nn_action)
            ref_action = control_action

            env.car_state.crash_chance = (nn_value) #UNnEAT
            next_state, reward, done = env.step(ref_action)

            self.store_outputs(b, nn_state, nn_action, nn_value, reward, done)         

            if done:
                f.plot(self.ep_rewards, figure_n=5)
                self.ep_rewards.append(0.0)
                print("Episode: %03d, Reward: %03d" % (len(self.ep_rewards) - 1, self.ep_rewards[-2]))

                next_state = env.reset()
                self.pind = 0

            state = next_state

        self.state = next_state # remember for next batch
        _, b.last_q_val = self.model(nn_state)
        
        return b

    def run_test(self, track, tests=100, show_eps=False):
        f_show = 5
        env, state = self.env, self.state

        while len(self.ep_rewards) < tests:
            destination, nn_state = self.determine_destination(state)
            control_action = self.control_system(state[0:4], destination)
            nn_action, nn_value = self.model(nn_state)
            
            ref_action = self.add_actions(control_action, nn_action)

            env.car_state.crash_chance = (nn_value) #UNnEAT
            next_state, reward, done = env.step(ref_action)
            self.ep_rewards[-1] += (reward + self.wp_done)

            if done:
                f.plot(self.ep_rewards)
                self.ep_rewards.append(0.0)
                print("Episode: %03d, Reward: %03d" % (len(self.ep_rewards) - 1, self.ep_rewards[-2]))

                if len(self.ep_rewards) % f_show == 1 and show_eps:
                    render_track_ep(track, self.path_obj, env.sim_mem, pause=False, dt=40)
                    # snap_track(track, self.path_obj, env.sim_mem)
                next_state = env.reset()
                self.pind = 0

            state = next_state

        return np.mean(self.ep_rewards)



    def determine_destination(self, state):
        self.wp_done = 0
        car_dist = f.get_distance(self.path[self.pind].x, state[0:2])
        ds = f.get_distance(self.path[self.pind].x, self.path[self.pind+1].x)
        while car_dist > ds: # makes sure will keep going until reached
            if self.pind < (self.n_inds-1):
                self.pind += 1
            self.wp_done = 1
            car_dist = f.get_distance(self.path[self.pind].x, state[0:2])
            ds = f.get_distance(self.path[self.pind].x, self.path[self.pind+1].x)

        destination = self.path[self.pind+1] # next point

        nn_state = state[2::]
        relative_destination = destination.x - state[0:2]
        nn_state = np.append(nn_state, relative_destination)

        return destination, nn_state

    def store_outputs(self, b, nn_state, nn_action, nn_value, reward, done):
        action_cost = abs(nn_action - 1) * 0.05
        new_reward = reward + self.wp_done - action_cost
        self.env.env_state.agent_action = nn_action

        self.ep_rewards[-1] += new_reward
        new_done = done or bool(self.wp_done)
        b.add(nn_state, 1, nn_value, new_reward, new_done) # nn action = 1

    def add_actions(self, control_action, nn_action):
        m = 1
        theta_mod = 0.1
        if nn_action == m:
            return control_action
        elif nn_action < m:
            action = [control_action[0], control_action[1] - theta_mod]
            return action
        elif nn_action > m:
            action = [control_action[0], control_action[1] + theta_mod]
            return action



