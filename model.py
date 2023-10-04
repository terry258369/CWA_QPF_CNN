import torch
import torch.nn.functional as F
from torch import nn
from layer import CBAM, BasicConv2d, OutConv, SelfAttention
from torchvision.models.inception import BasicConv2d, Inception3
from GAM import GAM


class CNN_Attention(nn.Module):
    def __init__(self, in_channels=20, out_channels=1, kernel_size=3, device='cuda'):
        super(CNN_Attention, self).__init__()
        self.device = device
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        self.conv1 = BasicConv2d(in_channels, 4, kernel_size=kernel_size, padding=padding)
        self.conv2 = BasicConv2d(4, 8, kernel_size=kernel_size, padding=padding)
        self.conv3 = BasicConv2d(8, 16, kernel_size=kernel_size, padding=padding)
        self.conv4 = BasicConv2d(16, 32, kernel_size=kernel_size, padding=padding)
        # self.conv5 = BasicConv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.CBAM = CBAM(32, ratio=10)
        # self.self_attention = SelfAttention(in_dim=1, activation=nn.ReLU())
        self.OutConv = OutConv(32, out_channels)

    def forward(self, x):
        x = x.to(self.device)
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        # outputs = self.conv5(outputs)
        outputs = self.CBAM(outputs)
        outputs = self.OutConv(outputs)
        # outputs = self.self_attention(outputs)
        outputs = F.adaptive_avg_pool2d(outputs, (312, 312))

        return outputs


class CNN(nn.Module):
    def __init__(self, in_channels=21, out_channels=1, kernel_size=3):
        super(CNN, self).__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        self.conv1 = BasicConv2d(in_channels, 4, kernel_size=kernel_size, padding=padding)
        self.conv2 = BasicConv2d(4, 8, kernel_size=kernel_size, padding=padding)
        self.conv3 = BasicConv2d(8, 16, kernel_size=kernel_size, padding=padding)
        self.conv4 = BasicConv2d(16, 32, kernel_size=kernel_size, padding=padding)
        # self.conv5 = BasicConv2d(16, 32, kernel_size=kernel_size, padding=padding)
        self.OutConv = OutConv(32, 1)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        # outputs = self.conv5(outputs)
        outputs = self.OutConv(outputs)
        outputs = F.adaptive_avg_pool2d(outputs, (312, 312))
        return outputs


class Inception(nn.Module):
    def __init__(self, in_channels=20, out_channels=1, kernel_size=3):
        self.inplanes = 41
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 8, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(8, 16, kernel_size=3, padding=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(8, 16, kernel_size=5, padding=2)

        self.branch_pool = BasicConv2d(in_channels, 1, kernel_size=1)
        self.OutConv = OutConv(41, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        outputs = torch.cat(outputs, 1)
        outputs = self.OutConv(outputs)
        outputs = F.adaptive_avg_pool2d(outputs, (312, 312))

        return outputs


class InceptionExperiment(nn.Module):
    def __init__(self, in_channels=20, out_channels=1, kernel_size=3, device='mps'):
        super(InceptionExperiment, self).__init__()
        self.device = device
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        self.conv1 = BasicConv2d(in_channels, 4, kernel_size=kernel_size, padding=padding)
        self.conv2 = BasicConv2d(4, 8, kernel_size=kernel_size, padding=padding)
        self.conv3 = BasicConv2d(8, 16, kernel_size=kernel_size, padding=padding)
        self.conv4 = BasicConv2d(16, 32, kernel_size=kernel_size, padding=padding)
        # self.conv5 = BasicConv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.CBAM = CBAM(32, ratio=4)

        # Inception Module
        self.branch1x1 = BasicConv2d(32, 8, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(8, 16, kernel_size=3, padding=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(8, 16, kernel_size=5, padding=2)

        self.branch_pool = BasicConv2d(in_channels, 1, kernel_size=1)
        self.OutConv = OutConv(41, 1)

    def forward(self, x):
        x = x.to(self.device)
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        # outputs = self.conv5(outputs)
        outputs = self.CBAM(outputs)

        # Inception Module
        branch1x1 = self.branch1x1(outputs)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        outputs = torch.cat(outputs, 1)
        outputs = self.OutConv(outputs)
        outputs = F.adaptive_avg_pool2d(outputs, (312, 312))

        return outputs
