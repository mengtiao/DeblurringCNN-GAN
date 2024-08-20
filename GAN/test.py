import os
import unittest
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from albumentations import Compose, CenterCrop, PadIfNeeded
from tqdm import tqdm
from dataset import PairedDataset
from models.networks import get_generator
from util.metrics import PSNR
from ssim.ssimlib import SSIM

# 用于测试图像增强的测试类

class TestImageAugmentations(unittest.TestCase):
    def setUp(self):
        self.img = (np.random.rand(100, 100, 3) * 255).astype('uint8')
    
    def test_augmentation_consistency(self):
        from aug import get_transforms
        for scope in ('strong', 'weak'):
            for crop in ('random', 'center'):
                aug_pipeline = get_transforms(80, scope=scope, crop=crop)
                img_a, img_b = self.img.copy(), self.img.copy()
                img_a, img_b = aug_pipeline(img_a, img_b)
                np.testing.assert_allclose(img_a, img_b)

# 用于测试数据集加载和处理的测试类

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = os.path.join(os.getcwd(), 'temp_data')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.raw_dir = os.path.join(self.tmp_dir, 'raw')
        self.gt_dir = os.path.join(self.tmp_dir, 'gt')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.gt_dir, exist_ok=True)
        for i in range(5):
            img = make_img()
            cv2.imwrite(os.path.join(self.raw_dir, f'{i}.png'), img)
            cv2.imwrite(os.path.join(self.gt_dir, f'{i}.png'), img)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir)
    
    def test_dataset_loading(self):
        dataset = PairedDataset.from_config({
            'files_a': os.path.join(self.raw_dir, '*.png'),
            'files_b': os.path.join(self.gt_dir, '*.png'),
            'size': 32,
            'bounds': [0, 1],
            'scope': 'strong',
            'crop': 'center',
            'preload': 1,
            'preload_size': 64,
            'verbose': False
        })
        dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, drop_last=True)
        for batch in dataloader:
            a, b = batch['a'], batch['b']
            self.assertEqual(a.shape[0], 2) 
            np.testing.assert_allclose(a.numpy(), b.numpy()) 

# 测试图像质量指标（PSNR和SSIM）

def test_image_quality_metrics():
    args = get_args()
    with open('config/config.yaml') as cfg:
        config = yaml.load(cfg, Loader=yaml.SafeLoader)
    model = get_generator(config['model'])
    model.load_state_dict(torch.load(args.weights_path)['model']).cuda()
    filenames = sorted(glob.glob(args.img_folder + '/test' + '/blur/**/*.png', recursive=True))
    psnr, ssim = test_metrics(model, filenames)
    print(f"Average PSNR: {psnr}, Average SSIM: {ssim}")

if __name__ == '__main__':
    unittest.main()
