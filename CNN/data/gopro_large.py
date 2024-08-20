from data.dataset import ImageDataset

from utils import interact

class GOPROLarge(ImageDataset):
    """Handles GOPRO Large dataset for training and testing."""
    def __init__(self, config, mode='train'):
        super(GOPROLarge, self).__init__(config, mode)

    def define_modes(self):
        self.valid_modes = ('train', 'test')

    def configure_keys(self):
        super(GOPROLarge, self).configure_keys()
        self.blur_tag = 'blur_gamma'
        # Uncomment or add the sharp tag as necessary
        # self.sharp_tag = 'sharp'

    def __getitem__(self, index):
        blur_image, sharp_image, padding_width, idx, relative_path = super(GOPROLarge, self).__getitem__(index)
        relative_path = relative_path.replace(f'{self.blur_tag}/', '')

        return blur_image, sharp_image, padding_width, idx, relative_path
