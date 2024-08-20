import os
from copy import deepcopy
from functools import partial
from glob import glob
from hashlib import sha1
from typing import Callable, Iterable, Optional, Tuple

import cv2
import numpy as np
from glog import logger
from joblib import Parallel, cpu_count, delayed
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm

import aug

def bucket_subsample(dataset: Iterable, range_bounds: Tuple[float, float], hash_function: Callable, num_buckets=100, salt='', verbose=True):
    dataset_list = list(dataset)
    bucket_indices = assign_buckets(dataset_list, num_buckets=num_buckets, salt=salt, hash_function=hash_function)

    lower_bound, upper_bound = [x * num_buckets for x in range_bounds]
    logging_message = f'Subsampling buckets from {lower_bound} to {upper_bound}, total buckets number is {num_buckets}'
    if salt:
        logging_message += f'; salt is {salt}'
    if verbose:
        logger.info(logging_message)
    return np.array([item for bucket, item in zip(bucket_indices, dataset_list) if lower_bound <= bucket < upper_bound])

def compute_hash(paths: Tuple[str, str], salt: str = '') -> str:
    file_paths = ''.join(map(os.path.basename, paths))
    return sha1(f'{file_paths}_{salt}'.encode()).hexdigest()

def assign_buckets(items: Iterable, num_buckets: int, hash_function: Callable, salt=''):
    hashes = map(partial(hash_function, salt=salt), items)
    return np.array([int(h, 16) % num_buckets for h in hashes])

def read_image(file_path: str):
    image = cv2.imread(file_path)
    if image is None:
        logger.warning(f'Failed to read image {file_path} using OpenCV, trying with skimage')
        image = imread(file_path)[:, :, ::-1]  # Convert BGR to RGB
    return image

class ImagePairsDataset(Dataset):
    def __init__(self,
                 files_a: Tuple[str],
                 files_b: Tuple[str],
                 transform_fn: Callable,
                 normalize_fn: Callable,
                 corruption_fn: Optional[Callable] = None,
                 should_preload: bool = True,
                 preload_dim: Optional[int] = 0,
                 verbose=True):

        assert len(files_a) == len(files_b)
        self.should_preload = should_preload
        self.data_a = files_a
        self.data_b = files_b
        self.verbose = verbose
        self.corruption_fn = corruption_fn
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        logger.info(f'Initializing dataset with {len(self.data_a)} pairs')

        if should_preload:
            preload = partial(self.bulk_preload, preload_dim=preload_dim)
            self.data_a, self.data_b = map(preload, (files_a, files_b)) if files_a != files_b else (preload(files_a), preload(files_a))

    def bulk_preload(self, paths: Iterable[str], preload_dim: int):
        preload_tasks = [delayed(read_image)(path) for path in paths]
        if self.verbose:
            preload_tasks = tqdm(preload_tasks, desc='Preloading images')
        return Parallel(n_jobs=cpu_count(), backend='threading')(preload_tasks)

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        a, b = self.data_a[idx], self.data_b[idx]
        a, b = self.transform_fn(a, b)
        if self.corruption_fn:
            a = self.corruption_fn(a)
        a, b = self.normalize_fn(a), self.normalize_fn(b)
        return {'a': a, 'b': b}

    @staticmethod
    def from_config(config):
        config = deepcopy(config)
        files_a, files_b = map(lambda x: sorted(glob(config[x], recursive=True)), ('files_a', 'files_b'))
        transform_fn = aug.get_transforms(size=config['size'], scope=config['scope'], crop=config['crop'])
        normalize_fn = aug.get_normalize()
        corruption_fn = aug.get_corrupt_function(config['corrupt'])

        subsampled_data = bucket_subsample(data=zip(files_a, files_b),
                                           range_bounds=config.get('bounds', (0, 1)),
                                           hash_function=compute_hash,
                                           verbose=config.get('verbose', True))

        files_a, files_b = map(list, zip(*subsampled_data))
        return ImagePairsDataset(files_a=files_a, files_b=files_b, transform_fn=transform_fn, normalize_fn=normalize_fn, corruption_fn=corruption_fn, verbose=config['verbose'])

