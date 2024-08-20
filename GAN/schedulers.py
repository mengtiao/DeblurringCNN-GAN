import math
from torch.optim import lr_scheduler

class CosineAnnealingRestart(lr_scheduler.CosineAnnealingLR):
    """Implement SGDR with Cosine Annealing and periodic restarts.

    Adjusts learning rate using a cosine annealing schedule between epochs, 
    automatically resetting after a specified number of epochs (`T_max`).
    """

    def __init__(self, optimizer, max_epochs=30, expansion_factor=1, min_lr=0, last_epoch=-1):
        """Initialize with cosine annealing strategy.

        Args:
            max_epochs (int): Max epochs before a reset.
            expansion_factor (int): Expands max_epochs after each reset.
            min_lr (int): Minimum learning rate.
            last_epoch (int): Last epoch index.
        """
        self.expansion_factor = expansion_factor
        super().__init__(optimizer, max_epochs, min_lr, last_epoch)

    def get_lr(self):
        """Calculate learning rate using cosine annealing."""
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.expansion_factor
        return [(base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 + self.eta_min
                for base_lr in self.base_lrs]

class LinearLRDecay(lr_scheduler._LRScheduler):
    """Linearly decays the learning rate to a minimum value over a set number of epochs."""

    def __init__(self, optimizer, total_epochs, start_decay_at=0, final_lr=0, last_epoch=-1):
        """Initialize the linear decay scheduler.

        Args:
            total_epochs (int): Total epochs for the decay.
            start_decay_at (int): Start decay after this epoch.
            final_lr (int): Final minimum learning rate.
        """
        self.total_epochs = total_epochs
        self.start_decay_at = start_decay_at
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the new learning rates for each parameter group."""
        if self.last_epoch < self.start_decay_at:
            return self.base_lrs
        return [(base_lr - (base_lr - self.final_lr) / self.total_epochs * (self.last_epoch - self.start_decay_at))
                for base_lr in self.base_lrs]
