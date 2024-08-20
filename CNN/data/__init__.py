"""通用数据集加载器"""

from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from .sampler import DistributedEvalSampler

class DatasetLoader():
    def __init__(self, config):

        self.operating_modes = ['train', 'val', 'test', 'demo']

        self.process = {
            'train': config.do_train,
            'val':  config.do_validate,
            'test': config.do_test,
            'demo': config.demo
        }

        self.dataset_config = {
            'train': config.data_train,
            'val': config.data_val,
            'test': config.data_test,
            'demo': 'Demo'
        }

        self.config = config

        def create_data_loader(mode='train'):
            data_name = self.dataset_config[mode]
            dataset_module = import_module('data.' + data_name.lower())
            dataset_class = getattr(dataset_module, data_name)(config, mode)

            if mode == 'train':
                if config.distributed:
                    batch_size = int(config.batch_size / config.n_GPUs)
                    sampler = DistributedSampler(dataset_class, shuffle=True, num_replicas=config.world_size, rank=config.rank)
                    num_workers = int((config.num_workers + config.n_GPUs - 1) / config.n_GPUs)
                else:
                    batch_size = config.batch_size
                    sampler = RandomSampler(dataset_class, replacement=False)
                    num_workers = config.num_workers
                drop_last = True

            elif mode in ('val', 'test', 'demo'):
                if config.distributed:
                    batch_size = 1
                    sampler = DistributedEvalSampler(dataset_class, shuffle=False, num_replicas=config.world_size, rank=config.rank)
                    num_workers = int((config.num_workers + config.n_GPUs - 1) / config.n_GPUs)
                else:
                    batch_size = config.n_GPUs
                    sampler = SequentialSampler(dataset_class)
                    num_workers = config.num_workers
                drop_last = False

            loader = DataLoader(
                dataset=dataset_class,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=drop_last,
            )

            return loader

        self.dataset_loaders = {}
        for mode in self.operating_modes:
            if self.process[mode]:
                self.dataset_loaders[mode] = create_data_loader(mode)
                print('===> Loading {} dataset: {}'.format(mode, self.dataset_config[mode]))
            else:
                self.dataset_loaders[mode] = None

    def get_loader(self):
        return self.dataset_loaders
