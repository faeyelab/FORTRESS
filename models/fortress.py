"""
FORTRESS: Function-composition Optimized Real-Time Resilient Structural Segmentation
Main model implementation with KAN integration and depthwise separable convolutions.
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

from .layers import DepthwiseSeparableConv2d, SpatialAttention
from .kan_layers import KANBlock, PatchEmbed


class EnhancedSAUNetWithKAN(nn.Module):
    """
    FORTRESS: Enhanced SAUNet with KAN Integration and Depthwise Separable Convolutions
    
    This model combines:
    - Depthwise separable convolutions for parameter efficiency
    - Kolmogorov-Arnold Network (KAN) integration for enhanced feature learning
    - Multi-scale spatial attention mechanisms
    - Deep supervision for improved training convergence
    """
    
    def __init__(self, in_channels=3, num_classes=7, block_size=7, keep_prob=0.9, start_filters=32, use_kan=True):
        super(EnhancedSAUNetWithKAN, self).__init__()
        self.num_classes = num_classes
        self.use_kan = use_kan and TIKAN_AVAILABLE
        
        # Encoder with KAN enhancement and depthwise separable convolutions
        self.enc1 = KANDoubleConv(in_channels, start_filters, block_size, keep_prob, use_kan)
        self.enc2 = KANDoubleConv(start_filters, start_filters*2, block_size, keep_prob, use_kan)
        self.enc3 = KANDoubleConv(start_filters*2, start_filters*4, block_size, keep_prob, use_kan)
        self.enc4 = KANDoubleConv(start_filters*4, start_filters*8, block_size, keep_prob, use_kan)
        self.enc5 = KANDoubleConv(start_filters*8, start_filters*16, block_size, keep_prob, use_kan)
        
        # Enhanced Spatial Attention with different kernel sizes
        self.spatial_attention1 = SpatialAttention(kernel_size=7)
        self.spatial_attention2 = SpatialAttention(kernel_size=5)
        self.spatial_attention3 = SpatialAttention(kernel_size=3)
        
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(start_filters*16, start_filters*16 // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(start_filters*16 // 16, start_filters*16, 1),
            nn.Sigmoid()
        )
        
        # Enhanced Decoder with Deep Supervision and KAN
        self.up4 = nn.ConvTranspose2d(start_filters*16, start_filters*8, kernel_size=2, stride=2)
        self.dec4 = KANDoubleConv(start_filters*16, start_filters*8, block_size, keep_prob, use_kan)
        self.ds4 = nn.Conv2d(start_filters*8, num_classes, kernel_size=1)
        
        self.up3 = nn.ConvTranspose2d(start_filters*8, start_filters*4, kernel_size=2, stride=2)
        self.dec3 = KANDoubleConv(start_filters*8, start_filters*4, block_size, keep_prob, use_kan)
        self.ds3 = nn.Conv2d(start_filters*4, num_classes, kernel_size=1)
        
        self.up2 = nn.ConvTranspose2d(start_filters*4, start_filters*2, kernel_size=2, stride=2)
        self.dec2 = KANDoubleConv(start_filters*4, start_filters*2, block_size, keep_prob, use_kan)
        self.ds2 = nn.Conv2d(start_filters*2, num_classes, kernel_size=1)
        
        self.up1 = nn.ConvTranspose2d(start_filters*2, start_filters, kernel_size=2, stride=2)
        self.dec1 = KANDoubleConv(start_filters*2, start_filters, block_size, keep_prob, use_kan)
        
        # Final convolution with depthwise separable conv
        self.final_conv = nn.Sequential(
            DepthwiseSeparableConv2d(start_filters, start_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(start_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(start_filters, num_classes, kernel_size=1)
        )
        
        # KAN enhancement for bottleneck
        if self.use_kan:
            self.bottleneck_patch_embed = PatchEmbed(
                img_size=8,
                patch_size=1,
                in_chans=start_filters*16,
                embed_dim=start_filters*16
            )
            self.bottleneck_kan = KANBlock(dim=start_filters*16, drop=0.0, drop_path=0.1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Residual connections
        self.res1 = nn.Conv2d(in_channels, start_filters, kernel_size=1)
        self.res2 = nn.Conv2d(start_filters, start_filters*2, kernel_size=1)
        self.res3 = nn.Conv2d(start_filters*2, start_filters*4, kernel_size=1)
        self.res4 = nn.Conv2d(start_filters*4, start_filters*8, kernel_size=1)

    def forward(self, x):
        # Encoder with residual connections
        e1 = self.enc1(x) + self.res1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1) + self.res2(self.pool(e1))
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2) + self.res3(self.pool(e2))
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3) + self.res4(self.pool(e3))
        p4 = self.pool(e4)
        
        # Deeper bottleneck with multi-scale attention and KAN
        e5 = self.enc5(p4)
        e5 = self.spatial_attention1(e5)
        e5 = e5 * self.channel_attention(e5)
        
        # KAN enhancement for bottleneck
        if self.use_kan and hasattr(self, 'bottleneck_kan'):
            B, C, H, W = e5.shape
            if H * W <= 256:  # Only for reasonable sizes
                tokens, newH, newW = self.bottleneck_patch_embed(e5)
                tokens = self.bottleneck_kan(tokens, newH, newW)
                kan_e5 = tokens.reshape(B, newH, newW, C).permute(0,3,1,2).contiguous()
                e5 = e5 + kan_e5 * 0.2  # Residual connection
        
        # Decoder with deep supervision and multi-scale attention
        d4 = self.up4(e5)
        d4 = torch.cat([d4, self.spatial_attention2(e4)], dim=1)
        d4 = self.dec4(d4)
        ds4_out = self.ds4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.spatial_attention3(e3)], dim=1)
        d3 = self.dec3(d3)
        ds3_out = self.ds3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        ds2_out = self.ds2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        
        if self.training:
            # Return main output and deep supervision outputs
            return (out, 
                   F.interpolate(ds4_out, size=out.shape[2:], mode='bilinear', align_corners=False),
                   F.interpolate(ds3_out, size=out.shape[2:], mode='bilinear', align_corners=False),
                   F.interpolate(ds2_out, size=out.shape[2:], mode='bilinear', align_corners=False))
        
        return out


class KANDoubleConv(nn.Module):
    """DoubleConv block enhanced with KAN processing and depthwise separable convolutions"""
    
    def __init__(self, in_channels, out_channels, block_size=7, keep_prob=0.9, use_kan=True):
        super().__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.use_kan = use_kan and TIKAN_AVAILABLE
        
        # Depthwise separable convolution path
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # KAN enhancement for feature refinement
        if self.use_kan:
            self.patch_embed = PatchEmbed(
                img_size=32,  # Adaptive based on input
                patch_size=1,
                in_chans=out_channels,
                embed_dim=out_channels
            )
            self.kan_block = KANBlock(dim=out_channels, drop=0.0, drop_path=0.0)
            self.reproject = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Depthwise separable convolution
        conv_out = self.double_conv(x)
        
        # KAN enhancement
        if self.use_kan:
            B, C, H, W = conv_out.shape
            # Only apply KAN if spatial dimensions are reasonable
            if H * W <= 1024:  # Prevent memory issues
                tokens, new_H, new_W = self.patch_embed(conv_out)
                tokens = self.kan_block(tokens, new_H, new_W)
                # Reshape back to spatial
                kan_out = tokens.reshape(B, new_H, new_W, C).permute(0, 3, 1, 2).contiguous()
                kan_out = self.reproject(kan_out)
                # Residual connection
                if kan_out.shape == conv_out.shape:
                    conv_out = conv_out + kan_out * 0.1  # Small residual weight
        
        if self.training:
            conv_out = self.dropblock(conv_out)
        return conv_out
    
    def dropblock(self, x):
        """DropBlock regularization"""
        if not self.training or self.keep_prob == 1:
            return x
            
        gamma = (1. - self.keep_prob) / self.block_size**2
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        mask = 1 - mask
        
        out = x * mask * (mask.numel() / mask.sum())
        return out

