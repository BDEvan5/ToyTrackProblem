import numpy as np 
from ValueAgent import BufferVanilla
from PathTracker import ControlSystem 
import LibFunctions as f
from TrackMapInterface import render_track_ep


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

    def run_batch(self, track): # temp track
        b, reward = BufferVanilla(), 0
        env, state = self.env, self.state
        reward = 0
        while len(b) <= self.nsteps:
            destination, nn_state = self.determine_destination(state)
            control_action = self.control_system(state[0:4], destination)
            nn_value, nn_action = self.model(nn_state), 1
            
            ref_action = self.add_actions(control_action, nn_action)

            env.car_state.crash_chance = (nn_value.numpy()[0]) #UNnEAT
            next_state, reward, done = env.step(ref_action)

            self.store_outputs(b, nn_state, nn_action, nn_value, reward, done)            

            if done:
                # f.plot(self.ep_rewards)
                self.ep_rewards.append(0.0)
                print(f"Last val: {b.values[-1]}")
                print(f"Last reward: {b.rewards[-1]}")
                print("Episode: %03d, Reward: %03d" % (len(self.ep_rewards) - 1, self.ep_rewards[-2]))

                # render_track_ep(track, self.path_obj, env.sim_mem, pause=True)
                next_state = env.reset()
                self.pind = 0

            state = next_state

        self.state = next_state
        _, q_val, _ = self.act(state)
        b.last_q_val = q_val
        # f.plot_comp(b.values, b.rewards, figure_n=4)

        # render_track_ep(track, self.path_obj, env.sim_mem, pause=True)
        
        return b


    def determine_destination(self, state):
        self.wp_done = 0
        car_dist = f.get_distance(self.path[self.pind].x, state[0:2])
        ds = f.get_distance(self.path[self.pind].x, self.path[self.pind+1].x)
        while car_dist > ds: # makes sure will keep going until reached
            if self.pind < (self.n_inds-1):
                self.pind += 1
            self.wp_done = 1

        destination = self.path[self.pind+1] # next point

        nn_state = state[2::]
        relative_destination = destination.x - state[0:2]
        nn_state = np.append(nn_state, relative_destination)

        return destination, nn_state

    def store_outputs(self, b, nn_state, nn_action, nn_value, reward, done):
        self.ep_rewards[-1] += reward
        new_reward = reward + self.wp_done

        new_done = done or bool(self.wp_done)
        b.add(nn_state, 1, nn_value, new_reward, new_done) # nn action = 1

    def add_actions(self, control_action, nn_action):
        if nn_action == 1:
            return control_action
        else:
            print("Unknown NN action")
            raise ValueError

    # def act(self, state):
    #     location = state[0:4]
    #     nn_state = state[2::]

    #     car_dist = f.get_distance(self.path[self.pind].x, state[0:2])
    #     ds = f.get_distance(self.path[self.pind].x, self.path[self.pind+1].x)
    #     if car_dist > ds: 
    #         if self.pind < (self.n_inds-1):
    #             self.pind += 1

    #     destination = self.path[self.pind+1] # next point

    #     relative_destination = destination.x - state[0:2]
    #     nn_state = np.append(nn_state, relative_destination)
    #     value = self.model.get_action_value(nn_state[None, :])

    #     control_action = self.control_system(location, destination)

    #     # add nn_action and control action
    #     ref_action = control_action # + nn_action
    #     # print(ref_action)

    #     return ref_action, value, nn_state

