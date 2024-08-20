import logging
from collections import defaultdict
import numpy as np
from tensorboardX import SummaryWriter

# Constants
WINDOW_SIZE = 100

class PerformanceTracker:
    def __init__(self, experiment_name):
        self.writer = SummaryWriter(experiment_name)
        logging.basicConfig(filename=f'{experiment_name}.log', level=logging.DEBUG)
        self.metrics = defaultdict(list)
        self.image_collections = defaultdict(list)
        self.best_psnr = 0

    def add_image(self, image: np.ndarray, tag: str):
        self.image_collections[tag].append(image)

    def reset(self):
        self.metrics = defaultdict(list)
        self.image_collections = defaultdict(list)

    def log_losses(self, generator_loss, content_loss, discriminator_loss=0):
        loss_values = {
            'G_loss': generator_loss,
            'G_loss_content': content_loss,
            'G_loss_adv': generator_loss - content_loss,
            'D_loss': discriminator_loss
        }
        for name, value in loss_values.items():
            self.metrics[name].append(value)

    def log_quality_metrics(self, psnr, ssim):
        quality_metrics = {'PSNR': psnr, 'SSIM': ssim}
        for name, value in quality_metrics.items():
            self.metrics[name].append(value)

    def compose_loss_report(self):
        recent_metrics = {key: np.mean(self.metrics[key][-WINDOW_SIZE:]) for key in ('G_loss', 'PSNR', 'SSIM') if key in self.metrics}
        return '; '.join(f'{key}={value:.4f}' for key, value in recent_metrics.items())

    def record_to_tensorboard(self, epoch, is_validation=False):
        prefix = 'Validation' if is_validation else 'Train'
        for metric_name in self.metrics:
            average_value = np.mean(self.metrics[metric_name])
            self.writer.add_scalar(f'{prefix}_{metric_name}', average_value, global_step=epoch)
        self.record_images(epoch)

    def record_images(self, epoch):
        for tag, images in self.image_collections.items():
            if images:
                image_stack = np.stack(images, axis=0)
                self.writer.add_images(tag, image_stack[:, :, :, ::-1].astype(np.float32) / 255.0,
                                       dataformats='NHWC', global_step=epoch)
                self.image_collections[tag] = []

    def evaluate_model_improvement(self):
        current_psnr = np.mean(self.metrics['PSNR'])
        if self.best_psnr < current_psnr:
            self.best_psnr = current_psnr
            return True
        return False
