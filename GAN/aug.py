from typing import List, Tuple
import albumentations as A

def create_augmentation_pipeline(size: int, augmentation_type: str = 'geometric', crop_type='random'):
    augmentation_options = {
        'weak': A.Compose([A.HorizontalFlip()]),
        'geometric': A.OneOf([
            A.HorizontalFlip(always_apply=True),
            A.ShiftScaleRotate(always_apply=True),
            A.Transpose(always_apply=True),
            A.OpticalDistortion(always_apply=True),
            A.ElasticTransform(always_apply=True),
        ])
    }

    augmentation_sequence = augmentation_options[augmentation_type]
    crop_methods = {
        'random': A.RandomCrop(size, size, always_apply=True),
        'center': A.CenterCrop(size, size, always_apply=True)
    }
    padding = A.PadIfNeeded(size, size)

    augmentation_pipeline = A.Compose([
        augmentation_sequence,
        padding,
        crop_methods[crop_type]
    ], additional_targets={'target': 'image'})

    def apply_augmentations(image, target):
        result = augmentation_pipeline(image=image, target=target)
        return result['image'], result['target']

    return apply_augmentations


def create_normalization_pipeline():
    normalization = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    composed_normalization = A.Compose([normalization], additional_targets={'target': 'image'})

    def apply_normalization(image, target):
        result = composed_normalization(image=image, target=target)
        return result['image'], result['target']

    return apply_normalization


def resolve_augmentation_function(name: str):
    augmentation_mapping = {
        'cutout': A.Cutout,
        'rgb_shift': A.RGBShift,
        'hsv_shift': A.HueSaturationValue,
        'motion_blur': A.MotionBlur,
        'median_blur': A.MedianBlur,
        'snow': A.RandomSnow,
        'shadow': A.RandomShadow,
        'fog': A.RandomFog,
        'brightness_contrast': A.RandomBrightnessContrast,
        'gamma': A.RandomGamma,
        'sun_flare': A.RandomSunFlare,
        'sharpen': A.Sharpen,
        'jpeg': A.ImageCompression,
        'gray': A.ToGray,
        'pixelize': A.Downscale,
    }
    return augmentation_mapping[name]


def construct_augmentation_process(config: List[dict]):
    augmentation_list = []
    for aug_info in config:
        augmentation_name = aug_info.pop('name')
        augmentation_class = resolve_augmentation_function(augmentation_name)
        probability = aug_info.pop('prob', 0.5)
        augmentation_list.append(augmentation_class(p=probability, **aug_info))

    combined_augmentations = A.OneOf(augmentation_list)

    def apply_augmentation(image):
        return combined_augmentations(image=image)['image']

    return apply_augmentation
