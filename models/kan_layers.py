"""
Kolmogorov-Arnold Network (KAN) layer implementations for FORTRESS
These layers implement function composition-based learning paradigms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import tikan for KAN
try:
    from tikan import KANLinear
    TIKAN_AVAILABLE = True
except ImportError:
    TIKAN_AVAILABLE = False

from .layers import DropPath


def to_2tuple(x):
    """Convert input to 2-tuple format"""
    if isinstance(x, (tuple, list)):
        return x
    return (x, x)


class DW_bn_relu(nn.Module):
    """
    Depthwise convolution + BatchNorm + ReLU used inside KAN layers
    
    This module applies spatial processing within the KAN framework
    while maintaining the token-based representation.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class KANLayer(nn.Module):
    """
    KAN layer with depthwise convolution integration
    
    Implements function composition-based transformations using either
    real KANLinear layers (if tikan is available) or fallback linear layers.
    """
    
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        grid_size=3,
        spline_order=2,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1,1],
        drop=0.,
        no_kan=False
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if not no_kan and TIKAN_AVAILABLE:
            # Use real KANLinear layers
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        else:
            # Fall back to normal linear layers if tikan not available
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # First transformation
        x = self.fc1(x.reshape(B*N, C))
        x = x.reshape(B, N, -1)
        x = self.dwconv_1(x, H, W)
        
        # Second transformation
        B, N, C2 = x.shape
        x = self.fc2(x.reshape(B*N, C2))
        x = x.reshape(B, N, -1)
        x = self.dwconv_2(x, H, W)
        return x


class KANBlock(nn.Module):
    """
    Complete KAN block with normalization and residual connection
    
    Combines layer normalization, KAN transformation, and residual connection
    with optional drop path regularization.
    """
    
    def __init__(self, dim, drop=0., drop_path=0., no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.layer = KANLayer(dim, dim, dim, no_kan=no_kan)

    def forward(self, x, H, W):
        shortcut = x
        x = self.norm(x)
        x = self.layer(x, H, W)
        return shortcut + self.drop_path(x)


class PatchEmbed(nn.Module):
    """
    Feature to Patch Embedding for KAN processing
    
    Converts spatial feature maps into token sequences suitable for
    KAN-based processing while preserving spatial relationships.
    """
    
    def __init__(
        self,
        img_size=8,
        patch_size=1,
        stride=1,
        in_chans=256,
        embed_dim=256
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
            bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.proj(x)
        _, embed_dim, H_new, W_new = out.shape
        # Flatten into tokens
        out = out.flatten(2).transpose(1, 2)
        out = self.norm(out)
        return out, H_new, W_new

