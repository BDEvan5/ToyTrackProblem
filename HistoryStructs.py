import LibFunctions as lib
import os, shutil


class TrainHistory():
    def __init__(self, agent_name) -> None:
        self.agent_name = agent_name

        # training data
        self.lengths = []
        self.rewards = [] 
        self.t_counter = 0 # total steps
        
        # espisode data
        self.ep_counter = 0 # ep steps
        self.ep_reward = 0

        self.init_file_struct()

    def init_file_struct(self):
        path = 'Vehicles/' + self.agent_name 

        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)

    def add_step_data(self, new_r):
        self.ep_reward += new_r
        self.ep_counter += 1
        self.t_counter += 1 

    def lap_done(self):
        self.lengths.append(self.ep_counter)
        self.rewards.append(self.ep_reward)

        self.ep_counter = 0
        self.ep_reward = 0

    def print_update(self):
        mean = np.mean(self.rewards)
        score = self.rewards[-1]
        print(f"Run: {self.t_counter} --> Score: {score:.2f} --> Mean: {mean:.2f} --> ")
        
        lib.plot(self.rewards, figure_n=2)
