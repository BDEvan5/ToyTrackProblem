import numpy as np 
from ValueAgent import BufferVanilla
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
        reward = 0
        while len(b) <= nsteps:
            check_wp = self.check_wp_done(state[0:2])
            ref_action, value, nn_state = self.act(state)
            value_store = value.numpy()[0]

            env.car_state.crash_chance = (value_store)
            next_state, reward, done = env.step(ref_action)

            self.ep_rewards[-1] += reward
            new_reward = reward + check_wp * 0.5
            new_done = done or bool(check_wp)
            b.add(nn_state, 1, value, new_reward, new_done) # nn action = 1

            if done:
                # f.plot(self.ep_rewards)
                self.ep_rewards.append(0.0)
                print(f"Last val: {b.values[-1]}")
                print(f"Last reward: {b.rewards[-1]}")
                print("Episode: %03d, Reward: %03d" % (len(self.ep_rewards) - 1, self.ep_rewards[-2]))

                
                # env.sim_mem.print_ep()
                # render_track_ep(track, self.path_obj, env.sim_mem, pause=True)
                next_state = env.reset()
                self.pind = 0

            state = next_state

        self.state = next_state
        _, q_val, _ = self.act(state)
        # nn_state = state[2::] + 
        # q_val = self.model.get_action_value(nn_state[None, :])
        b.last_q_val = q_val
        f.plot_comp(b.values, b.rewards, figure_n=4)
        # b.print_batch()

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

        relative_destination = destination.x - state[0:2]

        nn_state = np.append(nn_state, relative_destination)

        value = self.model.get_action_value(nn_state[None, :])

        control_action = self.control_system(location, destination)

        # add nn_action and control action
        ref_action = control_action # + nn_action
        # print(ref_action)

        return ref_action, value, nn_state

