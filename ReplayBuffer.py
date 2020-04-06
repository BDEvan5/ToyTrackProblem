import random

class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer = []

    def save_step(self, step):
        if len(self.buffer) > self.config.memory_size:
            self.buffer.pop(0)
        self.buffer.append(step)

    def add_ep(self, ep):
        for step in ep.steps: # passes a sim mem object
            self.save_step(step)

    def sample_batch(self):
        if len(self.buffer) > self.config.batch_size:
            sample_batch = random.sample(self.buffer, self.config.batch_size)
            return sample_batch
        sample_batch = random.sample(self.buffer, len(self.buffer)-1)
        return sample_batch

