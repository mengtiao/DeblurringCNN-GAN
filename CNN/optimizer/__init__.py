import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import os
from collections import Counter

from model import Model
from utils import interact, Map

class TrainingOptimizer(object):
    def __init__(self, config, model_instance):
        self.config = config

        self.optim_dir = os.path.join(self.config.save_dir, 'optimizer')
        os.makedirs(self.optim_dir, exist_ok=True)

        model_to_optimize = model_instance.model if isinstance(model_instance, Model) else model_instance

        # Base arguments for optimizer
        optimizer_settings = {
            'lr': config.lr,
            'weight_decay': config.weight_decay
        }

            # 根据配置选择并设置不同的优化器类型

        if config.optimizer_type == 'SGD':
            optimizer_class = optim.SGD
            optimizer_settings['momentum'] = config.momentum
        elif config.optimizer_type == 'ADAM':
            optimizer_class = optim.Adam
            optimizer_settings.update({'betas': config.betas, 'eps': config.epsilon})
        elif config.optimizer_type == 'RMSPROP':
            optimizer_class = optim.RMSprop
            optimizer_settings['eps'] = config.epsilon

        # Scheduler configuration
        scheduler_config, scheduler_class = self.configure_scheduler(config)

        self.create_optimizer = lambda model: self.build_optimizer(model, optimizer_class, optimizer_settings, scheduler_class, scheduler_config)

        self.primary = self.create_optimizer(model_to_optimize.G)
        self.secondary = self.create_optimizer(model_to_optimize.D) if hasattr(model_to_optimize, 'D') and model_to_optimize.D is not None else None

        self.load(config.load_epoch)

    def configure_scheduler(self, config):
        scheduler_kwargs = {}
        if config.scheduler_type == 'step':
            scheduler_class = lrs.MultiStepLR
            scheduler_kwargs = {
                'milestones': config.milestones,
                'gamma': config.gamma,
            }
        elif config.scheduler_type == 'plateau':
            scheduler_class = lrs.ReduceLROnPlateau
            scheduler_kwargs = {
                'mode': 'min',
                'factor': config.gamma,
                'patience': 10,
                'verbose': True,
                'threshold': 0.0001,
                'threshold_mode': 'abs',
                'cooldown': 10,
            }
        return scheduler_kwargs, scheduler_class

    def build_optimizer(self, model, optimizer_class, optimizer_settings, scheduler_class, scheduler_config):
        class CustomOptimizer(optimizer_class):
            def __init__(self, model_parameters):
                super(CustomOptimizer, self).__init__(model_parameters, **optimizer_settings)
                self.scheduler = scheduler_class(self, **scheduler_config)

            def adjust_learning_rate(self, metrics=None):  #调整学习率
                if isinstance(self.scheduler, lrs.ReduceLROnPlateau):
                    self.scheduler.step(metrics)
                else:
                    self.scheduler.step()

            def current_learning_rate(self):
                return self.scheduler.get_last_lr()[0]

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        return CustomOptimizer(trainable_params)

    def zero_grad(self):
        self.primary.zero_grad()
        if self.secondary:
            self.secondary.zero_grad()

    def step(self):
        self.primary.step()
        if self.secondary:
            self.secondary.step()

    def adjust_lr(self, metrics=None):
        self.primary.adjust_learning_rate(metrics)
        if self.secondary:
            self.secondary.adjust_learning_rate(metrics)

    def save(self, epoch=None):
        state_dict = {
            'primary': self.primary.state_dict(),
            'secondary': self.secondary.state_dict() if self.secondary else None
        }
        path = os.path.join(self.optim_dir, f'optimizer_state_{epoch}.pt')
        torch.save(state_dict, path)

    def load(self, epoch):
        if epoch <= 0:
            return
        path = os.path.join(self.optim_dir, f'optimizer_state_{epoch}.pt')
        state = torch.load(path)
        self.primary.load_state_dict(state['primary'])
        if self.secondary and 'secondary' in state:
            self.secondary.load_state_dict(state['secondary'])
