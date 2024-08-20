import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist

class DistributedValidationSampler(Sampler):
    """
    A sampler for distributed evaluation without extra padding of samples.
    This sampler should only be used for evaluation as it does not shuffle indices like the training samplers.
    It is useful in conjunction with torch.nn.parallel.DistributedDataParallel where each process can
    access only its subset of the dataset.

    Note:
        Assumes the dataset size is constant.

    Args:
        dataset: Dataset for sampling.
        num_replicas (optional, int): Number of distributed processes. Retrieved from the current group by default.
        rank (optional, int): Rank within the distributed group. Retrieved by default.
        shuffle (optional, bool): If True, indices will be shuffled. Default is False.
        seed (optional, int): Seed for random shuffling. Must be identical across all processes. Default is 0.

    Warning:
        When using distributed mode, ensure to call set_epoch at the start of each epoch to properly shuffle data.

    Example:
        >>> sampler = DistributedValidationSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(not is_distributed), sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     validate(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is required")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is required")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.total_size = len(self.dataset)
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Set the epoch number for the sampler. If shuffling is enabled, this ensures all replicas
        use a unique random order each epoch.

        Args:
            epoch (int): Epoch number to set.
        """
        self.epoch = epoch
