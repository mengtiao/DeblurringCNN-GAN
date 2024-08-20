import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def generate_gaussian_kernel(size, sigma):
    gaussian_kernel = torch.tensor([math.exp(-((x - size // 2) ** 2) / (2 * sigma ** 2)) for x in range(size)])
    return gaussian_kernel / gaussian_kernel.sum()

def initialize_ssim_window(size, channels):
    one_dimensional_kernel = generate_gaussian_kernel(size, 1.5).unsqueeze(1)
    two_dimensional_kernel = one_dimensional_kernel.mm(one_dimensional_kernel.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(two_dimensional_kernel.expand(channels, 1, size, size).contiguous())
    return window

def calculate_SSIM(index1, index2):
    channel_count = index1.size(1)
    window_size = 11
    ssim_window = initialize_ssim_window(window_size, channel_count)

    if index1.is_cuda:
        ssim_window = ssim_window.cuda(index1.get_device())
    ssim_window = ssim_window.type_as(index1)

    mean1 = F.conv2d(index1, ssim_window, padding=window_size // 2, groups=channel_count)
    mean2 = F.conv2d(index2, ssim_window, padding=window_size // 2, groups=channel_count)

    mean1_sq = mean1.pow(2)
    mean2_sq = mean2.pow(2)
    mean1_mean2 = mean1 * mean2

    variance1 = F.conv2d(index1 * index1, ssim_window, padding=window_size // 2, groups=channel_count) - mean1_sq
    variance2 = F.conv2d(index2 * index2, ssim_window, padding=window_size // 2, groups=channel_count) - mean2_sq
    covariance = F.conv2d(index1 * index2, ssim_window, padding=window_size // 2, groups=channel_count) - mean1_mean2

    constant1 = 0.01 ** 2
    constant2 = 0.03 ** 2

    ssim_index = ((2 * mean1_mean2 + constant1) * (2 * covariance + constant2)) / \
                 ((mean1_sq + mean2_sq + constant1) * (variance1 + variance2 + constant2))
    return ssim_index.mean()

def calculate_PSNR(reference, distorted):
    mse = torch.mean((reference / 255.0 - distorted / 255.0) ** 2)
    if mse == 0:
        return 100
    max_pixel_value = 1
    return 20 * math.log10(max_pixel_value / math.sqrt(mse))
