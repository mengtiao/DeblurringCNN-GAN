import os
import re
from importlib import import_module

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .discriminator import Discriminator

from utils import interact

class CustomModel(nn.Module):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.config = config
        self.device = config.device
        self.gpu_count = config.n_GPUs
        self.models_directory = os.path.join(config.save_dir, 'models')
        os.makedirs(self.models_directory, exist_ok=True)

        model_module = import_module('model.' + config.model)

        self.networks = nn.ModuleDict()
        self.networks.generator = model_module.build_model(config)
        if 'adv' in config.loss.lower():
            self.networks.discriminator = Discriminator(config)
        else:
            self.networks.discriminator = None

        self.to(config.device, dtype=config.dtype, non_blocking=True)
        self.load_model(config.load_epoch, path=config.pretrained)

    def setup_parallel(self):
        if self.config.device_type == 'cuda':
            ParallelClass = DistributedDataParallel if self.config.distributed else DataParallel
            parallel_args = {
                "device_ids": [self.config.rank] if self.config.distributed else list(range(self.gpu_count)),
                "output_device": self.config.rank
            }

            for key, model in self.networks.items():
                if model is not None:
                    self.networks[key] = ParallelClass(model, **parallel_args)

    def forward(self, inputs):
        return self.networks.generator(inputs)

    def compute_model_path(self, epoch):
        filename = f'model-{epoch}.pt'
        return os.path.join(self.models_directory, filename)

    def save_model(self, epoch):
        torch.save(self.get_state_dict(), self.compute_model_path(epoch))

    def load_model(self, epoch=None, path=None):
        model_path = path or self.compute_model_path(epoch if epoch >= 0 else self.find_latest_epoch())
        print(f'Loading model from {model_path}')
        state = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state)

    def synchronize_models(self):
        if self.config.distributed:
            model_params = parameters_to_vector(self.parameters())
            dist.broadcast(model_params, 0)
            if self.config.rank != 0:
                vector_to_parameters(model_params, self.parameters())
            del model_params

    def find_latest_epoch(self):
        files = sorted(os.listdir(self.models_directory))
        last_epoch = int(re.findall('\\d+', files[-1])[0]) if files else 0
        return last_epoch

    def display(self):
        print(self.networks)
