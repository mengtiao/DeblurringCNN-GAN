import os
import random
import imageio
import numpy as np
import torch.utils.data as data

from data import common
from utils import interact

class ImageDataset(data.Dataset):
    """Basic dataset loader class"""
    def __init__(self, config, usage_mode='train'):
        super(ImageDataset, self).__init__()
        self.config = config
        self.usage_mode = usage_mode

        self.valid_modes = ()
        self.define_modes()
        self.verify_mode()

        self.configure_keys()

        if self.usage_mode == 'train':
            dataset_dir = config.data_train
        elif self.usage_mode == 'val':
            dataset_dir = config.data_val
        elif self.usage_mode == 'test':
            dataset_dir = config.data_test
        elif self.usage_mode == 'demo':
            pass
        else:
            raise NotImplementedError(f'Unsupported mode: {self.usage_mode}!')

        if self.usage_mode == 'demo':
            self.data_root = config.demo_input_dir
        else:
            self.data_root = os.path.join(config.data_root, dataset_dir, self.usage_mode)

        self.blurred_images = []
        self.clear_images = []

        self.scan_directory()

    def define_modes(self):
        self.valid_modes = ('train', 'val', 'test', 'demo')

    def verify_mode(self):
        if self.usage_mode not in self.valid_modes:
            raise NotImplementedError(f'Invalid mode: {self.usage_mode}')

    def configure_keys(self):
        self.blur_tag = 'blur'  # to be overridden in subclass
        self.sharp_tag = 'sharp'  # to be overridden in subclass

        self.exclude_blur_tags = []
        self.exclude_sharp_tags = []

    def scan_directory(self, directory=None):
        if directory is None:
            directory = self.data_root

        def key_filter(path, true_tag, excluded_tags):
            path = os.path.join(path, '')
            if path.find(true_tag) >= 0:
                for tag in excluded_tags:
                    if path.find(tag) >= 0:
                        return False
                return True
            return False

        def collect_files_by_tag(root, true_tag, excluded_tags):
            files = []
            for subdir, dirs, file_names in os.walk(root):
                if not dirs:
                    path_files = [os.path.join(subdir, fname) for fname in file_names]
                    if key_filter(subdir, true_tag, excluded_tags):
                        files += path_files

            files.sort()
            return files

        def normalize_tags():
            self.blur_tag = os.path.join(self.blur_tag, '')
            self.exclude_blur_tags = [os.path.join(tag, '') for tag in self.exclude_blur_tags]
            self.sharp_tag = os.path.join(self.sharp_tag, '')
            self.exclude_sharp_tags = [os.path.join(tag, '') for tag in self.exclude_sharp_tags]

        normalize_tags()

        self.blurred_images = collect_files_by_tag(directory, self.blur_tag, self.exclude_blur_tags)
        self.clear_images = collect_files_by_tag(directory, self.sharp_tag, self.exclude_sharp_tags)

        if self.clear_images:
            assert(len(self.blurred_images) == len(self.clear_images))

    def __getitem__(self, index):
        blur_image = imageio.imread(self.blurred_images[index], pilmode='RGB')
        if self.clear_images:
            clear_image = imageio.imread(self.clear_images[index], pilmode='RGB')
            images = [blur_image, clear_image]
        else:
            images = [blur_image]

        pad_width = 0
        if self.usage_mode == 'train':
            images = common.crop(*images, ps=self.config.patch_size)
            if self.config.augment:
                images = common.augment(*images, hflip=True, rot=True, shuffle=True, change_saturation=True, rgb_range=self.config.rgb_range)
                images[0] = common.add_noise(images[0], sigma_sigma=2, rgb_range=self.config.rgb_range)
        elif self.usage_mode == 'demo':
            images[0], pad_width = common.pad(images[0], divisor=2**(self.config.n_scales-1))
        else:
            pass

        if self.config.gaussian_pyramid:
            images = common.generate_pyramid(*images, n_scales=self.config.n_scales)

        images = common.np2tensor(*images)
        relpath = os.path.relpath(self.blurred_images[index], self.data_root)

        blur = images[0]
        sharp = images[1] if len(images) > 1 else None

        return blur, sharp, pad_width, index, relpath

    def __len__(self):
        return len(self.blurred_images)
