import random
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import pyramid_gaussian

import torch

def apply_function(func, item):
    if isinstance(item, (list, tuple)):
        return [apply_function(func, sub_item) for sub_item in item]
    elif isinstance(item, dict):
        return {k: apply_function(func, v) for k, v in item.items()}
    else:
        return func(item)

def random_crop(*inputs, patch_size=256):
    def find_shape(input):
        if isinstance(input, (list, tuple)):
            return find_shape(input[0])
        elif isinstance(input, dict):
            return find_shape(list(input.values())[0])
        else:
            return input.shape

    height, width, _ = find_shape(inputs)

    start_y = random.randrange(0, height - patch_size + 1)
    start_x = random.randrange(0, width - patch_size + 1)

    def crop_image(image):
        if image.ndim == 2:
            return image[start_y:start_y+patch_size, start_x:start_x+patch_size, np.newaxis]
        else:
            return image[start_y:start_y+patch_size, start_x:start_x+patch_size]

    return apply_function(crop_image, inputs)

def insert_noise(*inputs, sigma_factor=2, max_rgb=255):
    if len(inputs) == 1:
        inputs = inputs[0]

    noise_level = np.random.normal() * sigma_factor * max_rgb / 255

    def add_noise(image):
        noise = np.random.randn(*image.shape).astype(np.float32) * noise_level
        return np.clip(image + noise, 0, max_rgb)

    return apply_function(add_noise, inputs)

def enhance_image(*inputs, flip_horizontally=True, rotate=True, do_shuffle=True, adjust_saturation=True, rgb_max=255):
    actions = (False, True)

    do_flip = flip_horizontally and random.choice(actions)
    do_vertical_flip = rotate and random.choice(actions)
    do_rotate = rotate and random.choice(actions)

    if do_shuffle:
        color_order = list(range(3))
        random.shuffle(color_order)
        if color_order == list(range(3)):
            do_shuffle = False

    if adjust_saturation:
        saturation_factor = np.random.uniform(0.5, 1.5)

    def augment_image(image):
        if do_flip: image = image[:, ::-1]
        if do_vertical_flip: image = image[::-1]
        if do_rotate: image = np.transpose(image, (1, 0, 2))
        if do_shuffle and image.ndim > 2 and image.shape[-1] == 3:
            image = image[..., color_order]

        if adjust_saturation:
            hsv = rgb2hsv(image)
            hsv[..., 1] *= saturation_factor
            image = hsv2rgb(hsv) * rgb_max

        return image.astype(np.float32)

    return apply_function(augment_image, inputs)

def pad_image(img, alignment=4, padding=None, is_negative=False):
    if isinstance(img, np.ndarray):
        return pad_numpy(img, alignment, padding, is_negative)
    elif isinstance(img, torch.Tensor):
        return pad_tensor(img, alignment, padding, is_negative)

def pad_numpy(image, align=4, padding=None, negative=False):
    if padding is None:
        height, width, _ = image.shape
        padding_height = -height % align
        padding_width = -width % align
        padding = ((0, padding_height), (0, padding_width), (0, 0))

    image = np.pad(image, padding, mode='edge')
    return image, padding

def pad_tensor(image, align=4, padding=None, negative=False):
    n, c, h, w = image.shape
    if padding is None:
        padding_height = -h % align
        padding_width = -w % align
        padding = (0, padding_width, 0, padding_height)
    else:
        padding_height, padding_width = padding
        if isinstance(padding_height, torch.Tensor):
            padding_height = padding_height.item()
        if isinstance(padding_width, torch.Tensor):
            padding_width = padding_width.item()
        padding = (0, padding_width, 0, padding_height)

    if negative:
        padding = [-p for p in padding]

    image = torch.nn.functional.pad(image, padding, 'reflect')
    return image, padding

def create_image_pyramid(*items, levels):
    def build_pyramid(image):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        return list(pyramid_gaussian(image, max_layer=levels - 1, multichannel=True))

    return apply_function(build_pyramid, items)

def convert_to_tensor(*items):
    def numpy_to_tensor(x):
        np_transposed = np.ascontiguousarray(x.transpose(2, 0, 1))
        tensor = torch.from_numpy(np_transposed)
        return tensor

    return apply_function(numpy_to_tensor, items)

def convert_type(*items, device=None, data_type=torch.float):
    def to_device_dtype(x):
        return x.to(device=device, dtype=data_type, non_blocking=True, copy=False)

    return apply_function(to_device_dtype, items)
