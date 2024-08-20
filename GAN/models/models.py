import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as SSIM
from util.metrics import PSNR

class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        inputs = data['a'].cuda()
        targets = data['b'].cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = ((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(imtype)
        return image_numpy

    def get_images_and_metrics(self, inp, output, target):
        inp_img = self.tensor2im(inp[0])
        fake_img = self.tensor2im(output.data[0])
        real_img = self.tensor2im(target.data[0])
        psnr = PSNR(fake_img, real_img)
        ssim = SSIM(fake_img, real_img, multichannel=True)
        vis_img = np.hstack((inp_img, fake_img, real_img))
        return psnr, ssim, vis_img

def get_model(model_config):
    return DeblurModel()
