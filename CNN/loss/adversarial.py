import torch
import torch.nn as nn

from utils import interact
import torch.cuda.amp as amp

class Adversarial(nn.Module):
    def __init__(self, args, model, optimizer):
        super(Adversarial, self).__init__()
        self.args = args
        self.model = model.model  # Assuming 'model' is a wrapper that contains 'D' (discriminator)
        self.optimizer = optimizer
        self.scaler = amp.GradScaler(init_scale=args.init_scale, enabled=args.amp)
        self.BCELoss = nn.BCEWithLogitsLoss()

    def update_discriminator(self, fake, real):
        fake_detach = fake.detach()
        self.optimizer.D.zero_grad()
        with amp.autocast(self.args.amp):
            loss_d = self.calculate_discriminator_loss(fake_detach, real)
        self.scaler.scale(loss_d).backward()
        self.scaler.step(self.optimizer.D)
        self.scaler.update()

    def calculate_discriminator_loss(self, fake, real):
        d_fake = self.model.D(fake)
        d_real = self.model.D(real)
        label_fake = torch.zeros_like(d_fake)
        label_real = torch.ones_like(d_real)
        return self.BCELoss(d_fake, label_fake) + self.BCELoss(d_real, label_real)

    def calculate_generator_loss(self, fake, real):
        d_real = self.model.D(real)
        label_real = torch.ones_like(d_real)
        d_fake_bp = self.model.D(fake)
        return self.BCELoss(d_fake_bp, label_real)

    def forward(self, fake, real, training=False):
        if training:
            self.update_discriminator(fake, real)
        loss_g = self.calculate_generator_loss(fake, real)
        return loss_g
