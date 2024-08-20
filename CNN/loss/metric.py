# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch import nn

def extend_dimensions(tensor):
    if tensor.ndim < 4:
        tensor = tensor.expand([1] * (4-tensor.ndim) + list(tensor.shape))
    return tensor

class PeakSignalNoiseRatio(nn.Module):
    def __init__(self):
        super(PeakSignalNoiseRatio, self).__init__()

    def forward(self, image1, image2, max_value=None):
        if max_value is None:
            max_value = 255 if image1.max() > 1 else 1

        squared_error = (image1 - image2) ** 2
        squared_error = extend_dimensions(squared_error)

        mean_squared_error = squared_error.mean(dim=list(range(1, squared_error.ndim)))
        psnr_value = 10 * (max_value ** 2 / mean_squared_error).log10().mean()

        return psnr_value

class StructuralSimilarityIndex(nn.Module):
    def __init__(self, device_type='cpu', precision=torch.float32):
        super(StructuralSimilarityIndex, self).__init__()

        self.device_type = device_type
        self.precision = precision
        self.filter_weight = self.calculate_ssim_weights().to(device_type, dtype=precision, non_blocking=True)

    def calculate_ssim_weights(self):
        sigma = 1.5
        radius = int(3.5 * sigma + 0.5)
        window_size = 2 * radius + 1
        num_channels = 3

        weights_1d = torch.Tensor([-(x - window_size // 2) ** 2 / (2 * sigma ** 2) for x in range(window_size)]).exp()
        weights_1d = weights_1d.unsqueeze(1)
        weights_2d = weights_1d.mm(weights_1d.t())
        weights_2d /= weights_2d.sum()
        window_weights = weights_2d.repeat(num_channels, 1, 1, 1)

        return window_weights

    def forward(self, image1, image2, max_value=None):
        image1 = image1.to(self.device_type, dtype=self.precision)
        image2 = image2.to(self.device_type, dtype=self.precision)

        if max_value is None:
            max_value = 255 if image1.max() > 1 else 1

        def filter_images(images):
            num_channels = images.shape[1]
            return nn.functional.conv2d(images, self.filter_weight, groups=num_channels)

        image1 = extend_dimensions(image1)
        image2 = extend_dimensions(image2)

        ux = filter_images(image1)
        uy = filter_images(image2)

        ux_squared = filter_images(image1 * image1)
        uy_squared = filter_images(image2 * image2)
        uxy = filter_images(image1 * image2)

        variance_x = ux_squared - ux * ux
        variance_y = uy_squared - uy * uy
        covariance_xy = uxy - ux * uy

        C1 = (0.01 * max_value) ** 2
        C2 = (0.03 * max_value) ** 2

        numerator1 = 2 * ux * uy + C1
        numerator2 = 2 * covariance_xy + C2
        denominator1 = ux ** 2 + uy ** 2 + C1
        denominator2 = variance_x + variance_y + C2

        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
        ssim_index = ssim_map.mean()

        return ssim_index
