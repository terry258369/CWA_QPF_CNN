import torch
from torch import nn
import torch.nn.functional as F
from typing import Union
from types import FunctionType


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // ratio),
            nn.ReLU(),
            nn.Linear(input_channels // ratio, input_channels)
        )

    def forward(self, x):
        """input shape: [batch_size, 20, 312, 312]"""

        """
        Take the input and apply average and max pooling
        avg_pool與max_pool是對輸入進行降維操作，準備用於後面的MLP模塊
        
        avg_values shape: [batch_size, 20, 1, 1]
        max_values shape: [batch_size, 20, 1, 1]
        """
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        # print("avg_values shape: ", avg_values.shape)
        # print("max_values shape: ", max_values.shape)

        """
        avg_pool與max_pool分別進入MLP模塊進行計算權重
        使用nn.Linear模塊中的神經元作為權重進行計算
        計算完成後，加總在一起
        
        out shape: [batch_size, 20]
        """
        out = self.MLP(avg_values) + self.MLP(max_values)
        # print("out shape: ", out.shape)

        """
        先用sigmoid做activation
        接著用unsqueeze在第2、3軸生出1維（為了後面的expand_as）
        expand_as會將out的shape延展成與x相同的shape，這樣才可以進行相乘操作
        
        sigmoid out shape: [batch_size, 20]
        unsqueeze shape: [batch_size, 20, 1, 1]
        expand_as shape: [batch_size, 20, 312, 312]
        scale shape: [batch_size, 20, 312, 312]
        """
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        # print("sigmoid shape: ", torch.sigmoid(out).shape)
        # print("unsqueeze shape: ", torch.sigmoid(out).unsqueeze(2).unsqueeze(3).shape)
        # print("expand_as shape: ", torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x).shape)
        # print("scale shape: ", scale.shape)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        """ input shape: [batch_size, 20, 312, 312] """

        """
        對每個channel相同位置取平均與最大值
        avg_out shape: [batch_size, 1, 312, 312]
        max_out shape: [batch_size, 1, 312, 312]
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print("avg_out shape: ", avg_out.shape)

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print("max_out shape: ", max_out.shape)

        """
        將平均與最大值於channel維度上concat在一起
        cat shape: [batch_size, 2, 312, 312]
        """
        out = torch.cat([avg_out, max_out], dim=1)
        # print("cat shape: ", out.shape)

        """
        進Conv2D卷積（in_channels=2, out_channels=1）
        conv shape: [batch_size, 1, 312, 312]
        """
        out = self.conv(out)
        # print("conv shape: ", out.shape)

        """
        Batch Normalization
        bn shape: [batch_size, 1, 312, 312]
        """
        out = self.bn(out)
        # print("bn shape: ", out.shape)

        """
        Sigmoid Activation
        out shape: [batch_size, 20, 312, 312]
        """
        scale = x * torch.sigmoid(out)
        # print("scale shape: ", scale.shape)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        # shape: [batch_size, channel, height, width]
        """ input shape: [batch_size, 20, 312, 312] """

        """
        Channel Attention
        out shape: [batch_size, 20, 312, 312]
        """
        out = self.channel_att(x)
        # print("channel att shape: ", out.shape)

        """
        Spatial Attention
        out shape: [batch_size, 20, 312, 312]
        """
        out = self.spatial_att(out)
        # print("spatial att shape: ", out.shape)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def init_conv(conv: nn.Conv2d, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SelfAttention(nn.Module):
    def __init__(
            self,
            in_dim: int,
            activation: nn.Module = nn.ReLU()
    ):
        super(SelfAttention, self).__init__()
        self.in_channels = in_dim
        self.activation = activation

        self.query = nn.Conv2d(in_channels=in_dim, out_channels=max(in_dim // 8, 1), kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=max(in_dim // 8, 1), kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.query)
        init_conv(self.key)
        init_conv(self.value)

    def forward(self, x):
        B, C, W, H = x.size()

        query = self.query(x).view(B, -1, W * H)
        key = self.key(x).view(B, -1, W * H)
        value = self.value(x).view(B, -1, W * H)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention)

        self_attention = torch.bmm(value, attention)
        self_attention = self_attention.view(B, C, W, H)

        return self.gamma * self_attention + x
