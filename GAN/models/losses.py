import torch
import torch.nn as nn
from torchvision.models import vgg19
from torch.autograd import Variable
from util.image_pool import ImagePool

# Function to initialize VGG19 model up to the 14th layer
def get_content_model():
    model = vgg19(pretrained=True).features[:15].cuda().eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

# Content loss comparing features extracted by VGG19
class ContentLoss(nn.Module):
    def __init__(self, layer=14):
        super().__init__()
        self.model = get_content_model()
        self.layer = layer
        self.loss = nn.MSELoss()
        
    def forward(self, fake, real):
        fake_features = self.model(fake)
        real_features = self.model(real).detach()
        return self.loss(fake_features, real_features)

# GAN loss for discrimination
class GANLoss(nn.Module):
    def __init__(self, use_l1=True):
        super().__init__()
        self.loss = nn.L1Loss() if use_l1 else nn.BCEWithLogitsLoss()
        
    def forward(self, input, target_is_real):
        target_tensor = torch.full_like(input, fill_value=(1.0 if target_is_real else 0.0))
        return self.loss(input, target_tensor)

# Main loss function wrapper
class CombinedLoss(nn.Module):
    def __init__(self, disc_type='wgan-gp'):
        super().__init__()
        self.content_loss = ContentLoss()
        self.gan_loss = GANLoss(use_l1=(disc_type=='lsgan'))
        self.image_pool = ImagePool(50)  # For storing previously generated images
        
    def forward(self, netD, fake, real):
        # Calculate GAN loss for fake and real
        fake_loss = self.gan_loss(netD(fake), True)
        real_loss = self.gan_loss(netD(real), False)
        # Calculate content loss
        content_loss = self.content_loss(fake, real)
        return fake_loss, real_loss, content_loss

# Example of how to use these classes
def setup_losses():
    model = {
        'content_loss': 'perceptual',
        'disc_loss': 'wgan-gp'
    }
    content_loss, disc_loss = get_loss(model)
    return content_loss, disc_loss

def get_loss(model):
    if model['content_loss'] == 'perceptual':
        content_loss = ContentLoss()
    else:
        content_loss = ContentLoss()  # Default to using MSE if not specified
    if model['disc_loss'] == 'wgan-gp':
        disc_loss = GANLoss(use_l1=False)  # Example adjustment for WGAN-GP
    else:
        disc_loss = GANLoss(use_l1=True)  # Default to LSGAN if not specified
    return content_loss, disc_loss
