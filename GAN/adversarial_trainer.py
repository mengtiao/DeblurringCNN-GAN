import torch
import copy

class ModelFactoryRegistry:
    registry = {}

    @staticmethod
    def register_factory(model_id, factory_instance):
        ModelFactoryRegistry.registry[model_id] = factory_instance

    @staticmethod
    def get_model(model_id, discriminator=None, loss_fn=None):
        if model_id not in ModelFactoryRegistry.registry:
            raise ValueError(f"Model ID {model_id} not recognized.")
        return ModelFactoryRegistry.registry[model_id].create(discriminator, loss_fn)

class GANBase:
    def __init__(self, discriminator, loss_fn):
        self.discriminator = discriminator
        self.loss_fn = loss_fn

    def compute_d_loss(self, predictions, targets):
        raise NotImplementedError

    def compute_g_loss(self, predictions, targets):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

class SimpleGAN(GANBase):
    def __init__(self, discriminator, loss_fn):
        super().__init__(discriminator.cuda(), loss_fn)

    def compute_d_loss(self, predictions, targets):
        return self.loss_fn(self.discriminator, predictions, targets)

    def compute_g_loss(self, predictions, targets):
        return self.loss_fn.get_generator_loss(self.discriminator, predictions, targets)

    def parameters(self):
        return self.discriminator.parameters()

    class Factory:
        @staticmethod
        def create(discriminator, loss_fn):
            return SimpleGAN(discriminator, loss_fn)

class MultiDiscriminatorGAN(GANBase):
    def __init__(self, discriminator, loss_fn):
        super().__init__(discriminator, loss_fn)
        self.patch_discriminator = discriminator['patch'].cuda()
        self.full_discriminator = discriminator['full'].cuda()
        self.full_loss_fn = copy.deepcopy(loss_fn)

    def compute_d_loss(self, predictions, targets):
        patch_loss = self.loss_fn(self.patch_discriminator, predictions, targets)
        full_loss = self.full_loss_fn(self.full_discriminator, predictions, targets)
        return (patch_loss + full_loss) / 2

    def compute_g_loss(self, predictions, targets):
        patch_g_loss = self.loss_fn.get_generator_loss(self.patch_discriminator, predictions, targets)
        full_g_loss = self.full_loss_fn.get_generator_loss(self.full_discriminator, predictions, targets)
        return (patch_g_loss + full_g_loss) / 2

    def parameters(self):
        return list(self.patch_discriminator.parameters()) + list(self.full_discriminator.parameters())

    class Factory:
        @staticmethod
        def create(discriminator, loss_fn):
            return MultiDiscriminatorGAN(discriminator, loss_fn)

# Example of registration and usage
ModelFactoryRegistry.register_factory('SimpleGAN', SimpleGAN.Factory())
ModelFactoryRegistry.register_factory('MultiDiscriminatorGAN', MultiDiscriminatorGAN.Factory())
