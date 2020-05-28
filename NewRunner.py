import numpy as np 
from VanillaAgent import BufferVanilla
from PathTracker import ControlSystem 
import LibFunctions as f
from TrackMapInterface import render_track_ep


class NewRunner:
    def __init__(self, env, model, path_obj):
        self.env = env
        self.model = model
        self.path_obj = path_obj

        self.ep_rewards = [0.0]
        self.state = self.env.reset()

        self.path = path_obj.route
        self.n_inds = len(self.path) -1

        self.pind = 0
        self.control_system = ControlSystem()

    def run_batch(self, track): # temp track
        b = BufferVanilla()
        nsteps = 64
        env = self.env
        state = self.state
        while len(b) <= nsteps:
            ref_action, value = self.act(state)
            env.car_state.crash_chance = (1-value.numpy())
            next_state, reward, done = env.step(ref_action)
            new_reward = reward + self.check_wp_done(state[0:2])
            self.ep_rewards[-1] += reward

            b.add(state[2::], 1, value, new_reward, done) # nn action = 1

            if done:
                # f.plot(self.ep_rewards)
                self.ep_rewards.append(0.0)
                print("Episode: %03d, Reward: %03d" % (len(self.ep_rewards) - 1, self.ep_rewards[-2]))

                # env.sim_mem.print_ep()
                # render_track_ep(track, self.path_obj, env.sim_mem, pause=True)
                next_state = env.reset()
                self.pind = 0

            state = next_state

        self.state = next_state
        nn_state = state[2::]
        q_val = self.model.get_action_value(nn_state[None, :])
        b.last_q_val = q_val

        # render_track_ep(track, self.path_obj, env.sim_mem, pause=True)
        
        return b

    def check_wp_done(self, location):
        car_dist = f.get_distance(self.path[self.pind].x, location)
        ds = f.get_distance(self.path[self.pind].x, self.path[self.pind+1].x)
        if car_dist > ds: 
            return 1
        return 0



    def act(self, state):
        location = state[0:4]
        nn_state = state[2::]

        car_dist = f.get_distance(self.path[self.pind].x, state[0:2])
        ds = f.get_distance(self.path[self.pind].x, self.path[self.pind+1].x)
        if car_dist > ds: 
            if self.pind < (self.n_inds-1):
                self.pind += 1
            # else:
            #     self.env.car_state.done = True

        destination = self.path[self.pind+1] # next point
        # destination.print_point()
        # print(f"Location: {location}")

        value = self.model.get_action_value(nn_state[None, :])

        control_action = self.control_system(location, destination)

        # add nn_action and control action
        ref_action = control_action # + nn_action
        # print(ref_action)

        return ref_action, value

