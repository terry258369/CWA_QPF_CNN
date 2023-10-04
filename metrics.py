import torch
import torch.nn as nn
import numpy as np
from scipy.signal import convolve2d


class WeightMSE(nn.Module):
    def __init__(self, weight):
        super(WeightMSE, self).__init__()
        self.weight = weight

    def forward(self, output, label):
        # label = label.to('mps')
        label = label.float()

        error = label - output
        error_weight = torch.where((label < 80), torch.pow(error, 2) * self.weight[0], error)
        error_weight = torch.where((label >= 80) & (label < 200), torch.pow(error, 2) * self.weight[1], error_weight)
        error_weight = torch.where((label >= 200) & (label < 350), torch.pow(error, 2) * self.weight[2], error_weight)
        error_weight = torch.where((label >= 350) & (label < 500), torch.pow(error, 2) * self.weight[3], error_weight)
        error_weight = torch.where((label >= 500), torch.pow(error, 2) * self.weight[4], error_weight)

        # error_weight = torch.where((label < 15), torch.pow(error, 2) * self.weight[0], error)
        # error_weight = torch.where((label >= 15) & (label < 40), torch.pow(error, 2) * self.weight[1], error_weight)
        # error_weight = torch.where((label >= 40) & (label < 70), torch.pow(error, 2) * self.weight[2], error_weight)
        # error_weight = torch.where((label >= 70) & (label < 130), torch.pow(error, 2) * self.weight[3], error_weight)
        # error_weight = torch.where((label >= 130), torch.pow(error, 2) * self.weight[4], error_weight)

        error_weight_mean = torch.mean(error_weight)
        error_weight_mean = torch.sqrt(error_weight_mean)

        return error_weight_mean


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, output, label):
        output = output[0:270, 86:244]
        label = label[0:270, 86:244]

        error = label - output
        error_score = np.sqrt(np.mean(error ** 2))

        return error_score


def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.k1 = 0.01
        self.k2 = 0.02
        self.win_size = 3
        self.L = 255

    def forward(self, output, label):
        im1 = output[0:270, 86:244]
        im2 = label[0:270, 86:244]

        if not im1.shape == im2.shape:
            raise ValueError("Input Images must have the same dimensions")
        if len(im1.shape) > 2:
            raise ValueError("Please input the images with 1 channel")

        M, N = im1.shape
        C1 = (self.k1 * self.L) ** 2
        C2 = (self.k2 * self.L) ** 2
        window = matlab_style_gauss2d(shape=(self.win_size, self.win_size), sigma=1.5)
        window = window / np.sum(np.sum(window))

        if im1.dtype == np.uint8:
            im1 = np.double(im1)
        if im2.dtype == np.uint8:
            im2 = np.double(im2)

        mu1 = filter2(im1, window, 'valid')
        mu2 = filter2(im2, window, 'valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
        sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
        sigma_l2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma_l2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return np.mean(np.mean(ssim_map))
