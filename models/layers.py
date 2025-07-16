"""
Custom layer implementations for FORTRESS model
Including depthwise separable convolutions and spatial attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution: Depthwise + Pointwise convolutions
    
    This implementation provides significant parameter reduction compared to standard convolutions
    while maintaining similar representational capacity.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        # Depthwise convolution - applies one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            groups=in_channels, bias=False
        )
        # Pointwise convolution - 1x1 conv to combine features
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module using depthwise separable convolutions
    
    Computes spatial attention weights by aggregating channel information
    through average and max pooling operations.
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Use depthwise separable conv for spatial attention
        self.conv = DepthwiseSeparableConv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        sa_map = self.sigmoid(self.conv(concat))
        return x * sa_map


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth) regularization
    
    Randomly drops entire paths during training to improve generalization.
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class DoubleConv(nn.Module):
    """
    Traditional DoubleConv block with depthwise separable convolutions
    
    Applies two consecutive convolution operations with batch normalization
    and ReLU activation, enhanced with DropBlock regularization.
    """
    
    def __init__(self, in_channels, out_channels, block_size=7, keep_prob=0.9):
        super().__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size
        
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        if self.training:
            x = self.dropblock(x)
        return x
    
    def dropblock(self, x):
        """DropBlock regularization implementation"""
        if not self.training or self.keep_prob == 1:
            return x
            
        gamma = (1. - self.keep_prob) / self.block_size**2
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        mask = 1 - mask
        
        out = x * mask * (mask.numel() / mask.sum())
        return out

