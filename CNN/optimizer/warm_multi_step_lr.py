import math
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler

# MultiStep learning rate scheduler with warm restart
class ProgressiveStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, decay_rate=0.1, last_epoch=-1, expansion_factor=1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                f'Milestones should be a list of increasing integers. Got {milestones}'
            )

        self.milestones = milestones
        self.decay_rate = decay_rate
        self.expansion_factor = expansion_factor

        self.initial_warmup = 5
        self.incremental_gain = (self.expansion_factor - 1) / self.initial_warmup
        super(ProgressiveStepLR, self).__init__(optimizer, last_epoch)

    def compute_learning_rates(self):
        if self.last_epoch < self.initial_warmup:
            return [
                base_lr * (1 + self.last_epoch * self.incremental_gain) / self.expansion_factor
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * self.decay_rate ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]
