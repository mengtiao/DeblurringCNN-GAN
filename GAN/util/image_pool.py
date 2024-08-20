import random
import numpy as np
import torch
from torch.autograd import Variable
from collections import deque

class SampleBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.current_size = 0
        if self.max_size > 0:
            self.buffer = deque()

    def store(self, new_samples):
        if self.max_size == 0:
            return new_samples
        for sample in new_samples.data:
            sample = torch.unsqueeze(sample, 0)
            if self.current_size < self.max_size:
                self.current_size += 1
                self.buffer.append(sample)
            else:
                self.buffer.popleft()
                self.buffer.append(sample)

    def fetch(self):
        sample_count = min(len(self.buffer), self.max_size)
        if sample_count > 0:
            selected_samples = random.sample(self.buffer, sample_count)
        else:
            selected_samples = list(self.buffer)
        return torch.cat(selected_samples, 0)
