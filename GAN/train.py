import logging
import os
from functools import partial

import cv2
import torch
import torch.optim as optim
import tqdm
import yaml
from torch.utils.data import DataLoader

from adversarial_trainer import GANManager
from data_manager import ImagePairLoader
from performance_tracker import PerformanceLogger
from loss_functions import define_loss
from architecture import fetch_model, fetch_networks
from lr_schedulers import GradualDecay, CosineAnnealingWithRestarts
from fire import Fire

cv2.setNumThreads(0)


class NeuralTrainer:
    def __init__(self, configuration, training_loader: DataLoader, validation_loader: DataLoader):
        self.configuration = configuration
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.adversarial_weight = configuration['model']['adv_lambda']
        self.performance_logger = PerformanceLogger(configuration['experiment_desc'])
        self.initial_epochs = configuration['warmup_num']

        # 初始化训练器

    def execute_training(self):
        self.prepare_network()
        for epoch in range(self.configuration['num_epochs']):
            if epoch == self.initial_epochs and self.initial_epochs != 0:
                self.netG.module.unfreeze_layers()
                self.optim_G = self.define_optimizer(self.netG.parameters())
                self.lr_scheduler_G = self.select_scheduler(self.optim_G)
            self.process_epoch(epoch)
            self.evaluate_model(epoch)
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()

            if self.performance_logger.record_best_model():
                torch.save({
                    'model': self.netG.state_dict()
                }, 'optimal_{}.h5'.format(self.configuration['experiment_desc']))
            torch.save({
                'model': self.netG.state_dict()
            }, 'current_{}.h5'.format(self.configuration['experiment_desc']))
            print(self.performance_logger.report_losses())
            logging.debug("Experiment: %s, Epoch: %d, Metrics: %s" % (
                self.configuration['experiment_desc'], epoch, self.performance_logger.report_losses()))

    def process_epoch(self, epoch_index):
        self.performance_logger.reset_metrics()
        for param_group in self.optim_G.param_groups:
            current_lr = param_group['lr']

        total_batches = self.configuration.get('train_batches_per_epoch', len(self.training_loader))
        progress_bar = tqdm.tqdm(self.training_loader, total=total_batches)
        progress_bar.set_description('Epoch {}, LR {}'.format(epoch_index, current_lr))
        
        # 执行单个训练周期

        for i, data in enumerate(progress_bar):
            inputs, targets = self.model.prepare_inputs(data)
            predictions = self.netG(inputs)
            discriminator_loss = self.update_discriminator(predictions, targets)
            self.optim_G.zero_grad()
            content_loss = self.criterionG(predictions, targets)
            adversarial_loss = self.gan_trainer.compute_generator_loss(predictions, targets)
            total_loss = content_loss + self.adversarial_weight * adversarial_loss
            total_loss.backward()
            self.optim_G.step()
            self.performance_logger.log_losses(total_loss.item(), content_loss.item(), discriminator_loss)
            psnr, ssim, img_snapshot = self.model.capture_metrics(inputs, predictions, targets)
            self.performance_logger.log_metrics(psnr, ssim)
            progress_bar.set_postfix(loss=self.performance_logger.report_losses())
            if i == 0:
                self.performance_logger.store_image(img_snapshot, tag='train')
            if i >= total_batches:
                break
        progress_bar.close()
        self.performance_logger.sync_with_tensorboard(epoch_index)

    def evaluate_model(self, epoch_index):
        self.performance_logger.reset_metrics()
        total_val_batches = self.configuration.get('val_batches_per_epoch', len(self.validation_loader))
        validation_progress = tqdm.tqdm(self.validation_loader, total=total_val_batches)
        validation_progress.set_description('Validation')
        
        # 评估当前模型的性能

        for i, data in enumerate(validation_progress):
            inputs, targets = self.model.prepare_inputs(data)
            with torch.no_grad():
                predictions = self.netG(inputs)
                content_loss = self.criterionG(predictions, targets)
                adversarial_loss = self.gan_trainer.compute_generator_loss(predictions, targets)
            generator_loss = content_loss + self.adversarial_weight * adversarial_loss
            self.performance_logger.log_losses(generator_loss.item(), content_loss.item())
            psnr, ssim, img_snapshot = self.model.capture_metrics(inputs, predictions, targets)
            self.performance_logger.log_metrics(psnr, ssim)
            if i == 0:
                self.performance_logger.store_image(img_snapshot, tag='val')
            if i >= total_val_batches:
                break
        validation_progress.close()
        self.performance_logger.sync_with_tensorboard(epoch_index, validation=True)

    def update_discriminator(self, outputs, targets):
        if self.configuration['model']['d_name'] == 'no_gan':
            return 0
        self.optim_D.zero_grad()
        discriminator_loss = self.adversarial_weight * self.gan_trainer.compute_discriminator_loss(outputs, targets)
        discriminator_loss.backward(retain_graph=True)
        self.optim_D.step()
        return discriminator_loss.item()

    def define_optimizer(self, params):
        optimizer_name = self.configuration['optimizer']['name']
        optimizers = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adadelta': optim.Adadelta
        }
        if optimizer_name not in optimizers:
            raise ValueError(f"Optimizer [{optimizer_name}] not recognized.")
        return optimizers[optimizer_name](params, lr=self.configuration['optimizer']['lr'])

    def select_scheduler(self, optimizer):
        scheduler_name = self.configuration['scheduler']['name']
        schedulers = {
            'plateau': optim.lr_scheduler.ReduceLROnPlateau,
            'sgdr': CosineAnnealingWithRestarts,
            'linear': GradualDecay
        }
        if scheduler_name not in schedulers:
            raise ValueError(f"Scheduler [{scheduler_name}] not recognized.")
        return schedulers[scheduler_name](optimizer, **self.configuration['scheduler'])

    def prepare_network(self):
        loss_funcs = define_loss(self.configuration['model'])
        self.criterionG, criterionD = loss_funcs
        self.netG, netD = fetch_networks(self.configuration['model'])
        self.netG.cuda()
        self.gan_trainer = GANManager.select_trainer(self.configuration['model']['d_name'], netD, criterionD)
        self.model = fetch_model(self.configuration['model'])
        self.optim_G = self.define_optimizer(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.optim_D = self.define_optimizer(self.gan_trainer.get_params())
        self.lr_scheduler_G = self.select_scheduler(self.optim_G)
        self.lr_scheduler_D = self.select_scheduler(self.optim_D)


def main(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    batch_size = config.pop('batch_size')
    DataLoaderConstructor = partial(DataLoader, batch_size=batch_size, shuffle=True, drop_last=True)

    dataset_paths = map(config.pop, ('train', 'val'))
    datasets = map(ImagePairLoader.from_config, dataset_paths)
    training_loader, validation_loader = map(DataLoaderConstructor, datasets)
    trainer = NeuralTrainer(config, train=training_loader, val=validation_loader)
    trainer.execute_training()


if __name__ == '__main__':
    Fire(main)
