# Enhanced SAUNet with KAN Integration, Depthwise Separable Convolutions for s2DS Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, jaccard_score, matthews_corrcoef
import os
import time
import random
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import torch.backends.cudnn as cudnn

# Import tikan for KAN
try:
    from tikan import KANLinear
    TIKAN_AVAILABLE = True
    print("tikan library found - using KAN layers")
except ImportError:
    print("tikan library not found - using regular linear layers as fallback")
    TIKAN_AVAILABLE = False

# Enable cuDNN autotuner for faster performance with fixed-size inputs
cudnn.benchmark = True

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cuda.matmul.allow_tf32 = False

random_generator = torch.Generator()
random_generator.manual_seed(seed)

# s2DS Dataset Configuration
NUMBER_CLASSES = 7

# Color mapping for s2DS visualization
COLORS = [
    [0, 0, 0],       # 0: Background - Black
    [255, 0, 0],     # 1: Crack - Red
    [0, 255, 0],     # 2: Spalling - Green  
    [0, 0, 255],     # 3: Corrosion - Blue
    [255, 255, 0],   # 4: Efflorescence - Yellow
    [255, 0, 255],   # 5: Vegetation - Magenta
    [0, 255, 255],   # 6: Control Point - Cyan
]

CLASS_NAMES = [
    'Background', 'Crack', 'Spalling', 'Corrosion', 
    'Efflorescence', 'Vegetation', 'Control Point'
]

# s2DS Dataset Class
class DefectDataset(Dataset):
    def __init__(self, folder_path, transform=None, size=(256, 256)):
        self.folder_path = folder_path
        self.transform = transform
        self.size = size
        
        # Original and remapped class values for s2DS
        self.original_values = np.array([0, 29, 76, 149, 178, 225, 255], dtype=np.uint8)
        self.remapped_values = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.uint8)
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(folder_path) 
            if f.endswith('.png') and not f.endswith('_lab.png')
        ])
        
        # Validate and get corresponding mask files
        self.mask_files = []
        for img_file in self.image_files:
            mask_file = img_file.replace('.png', '_lab.png')
            mask_path = os.path.join(folder_path, mask_file)
            if not os.path.exists(mask_path):
                raise ValueError(f"Missing mask file for image: {img_file}")
            self.mask_files.append(mask_file)

    def __len__(self):
        return len(self.image_files)
    
    def remap_mask(self, mask):
        remapped = np.zeros_like(mask, dtype=np.uint8)
        for orig, remap in zip(self.original_values, self.remapped_values):
            remapped[mask == orig] = remap
        return remapped

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.folder_path, self.mask_files[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Resize
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        
        # Remap mask values
        mask = self.remap_mask(mask)
        
        # Convert to tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).long()
        
        if self.transform:
            image = self.transform(image)
            
        return image, mask

# DropPath implementation
class DropPath(nn.Module):
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

# Depthwise Separable Convolution
class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise Separable Convolution: Depthwise + Pointwise convolutions"""
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

# KAN-related components
def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return x
    return (x, x)

class DW_bn_relu(nn.Module):
    """Depthwise + BN + ReLU used inside KAN layer"""
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
    """KAN layer with depthwise convolution"""
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
            # Use real KANLinear
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
            # Fall back to normal linear if tikan not available
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # fc1
        x = self.fc1(x.reshape(B*N, C))
        x = x.reshape(B, N, -1)
        x = self.dwconv_1(x, H, W)
        # fc2
        B, N, C2 = x.shape
        x = self.fc2(x.reshape(B*N, C2))
        x = x.reshape(B, N, -1)
        x = self.dwconv_2(x, H, W)
        return x

class KANBlock(nn.Module):
    """A KAN block that normalizes + uses KANLayer + residual"""
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
    """Feature to Patch Embedding for KAN processing"""
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

# Original components from EnhancedSAUNet
class SpatialAttention(nn.Module):
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

class DoubleConv(nn.Module):
    """Traditional DoubleConv with depthwise separable convolutions"""
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
        if not self.training or self.keep_prob == 1:
            return x
            
        gamma = (1. - self.keep_prob) / self.block_size**2
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        mask = 1 - mask
        
        out = x * mask * (mask.numel() / mask.sum())
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
        if not self.training or self.keep_prob == 1:
            return x
            
        gamma = (1. - self.keep_prob) / self.block_size**2
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        mask = 1 - mask
        
        out = x * mask * (mask.numel() / mask.sum())
        return out

class EnhancedSAUNetWithKAN(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, block_size=7, keep_prob=0.9, start_filters=32, use_kan=True):
        super(EnhancedSAUNetWithKAN, self).__init__()
        self.num_classes = num_classes
        self.use_kan = use_kan
        
        # Encoder with KAN enhancement and depthwise separable convolutions
        self.enc1 = KANDoubleConv(in_channels, start_filters, block_size, keep_prob, use_kan)
        self.enc2 = KANDoubleConv(start_filters, start_filters*2, block_size, keep_prob, use_kan)
        self.enc3 = KANDoubleConv(start_filters*2, start_filters*4, block_size, keep_prob, use_kan)
        self.enc4 = KANDoubleConv(start_filters*4, start_filters*8, block_size, keep_prob, use_kan)
        self.enc5 = KANDoubleConv(start_filters*8, start_filters*16, block_size, keep_prob, use_kan)
        
        # Enhanced Spatial Attention (now with depthwise separable convs)
        self.spatial_attention1 = SpatialAttention(kernel_size=7)
        self.spatial_attention2 = SpatialAttention(kernel_size=5)
        self.spatial_attention3 = SpatialAttention(kernel_size=3)
        
        # Channel Attention (using depthwise separable conv where applicable)
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
        if use_kan and TIKAN_AVAILABLE:
            self.bottleneck_patch_embed = PatchEmbed(
                img_size=8,
                patch_size=1,
                in_chans=start_filters*16,
                embed_dim=start_filters*16
            )
            self.bottleneck_kan = KANBlock(dim=start_filters*16, drop=0.0, drop_path=0.1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Residual connections (1x1 convs don't benefit much from depthwise separable)
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
        if self.use_kan and TIKAN_AVAILABLE and hasattr(self, 'bottleneck_kan'):
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

# Visualization Functions
def mask_to_rgb(mask, colors=COLORS):
    """Convert segmentation mask to RGB visualization"""
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        rgb_mask[mask == class_id] = color
    return rgb_mask

def plot_segmentation_results(images, ground_truths, predictions, save_path, num_samples=8):
    """Plot segmentation results with original image, ground truth, and prediction"""
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = images[i]
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # Ground truth
        gt = ground_truths[i]
        if torch.is_tensor(gt):
            gt = gt.cpu().numpy()
        gt_rgb = mask_to_rgb(gt.astype(np.uint8))
        
        # Prediction
        pred = predictions[i]
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        pred_rgb = mask_to_rgb(pred.astype(np.uint8))
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image {i+1}', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title(f'Ground Truth {i+1}', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title(f'Prediction {i+1}', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
    
    # Create legend
    legend_elements = [Patch(facecolor=np.array(color)/255.0, label=CLASS_NAMES[i]) 
                      for i, color in enumerate(COLORS)]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
               ncol=len(CLASS_NAMES), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Segmentation results saved to: {save_path}")

def plot_class_distribution(y_data, title, save_path):
    """Plot class distribution in the dataset"""
    unique, counts = np.unique(y_data.flatten(), return_counts=True)
    
    plt.figure(figsize=(12, 6))
    colors_norm = [np.array(COLORS[int(cls)])/255.0 for cls in unique]
    bars = plt.bar(range(len(unique)), counts, color=colors_norm)
    
    plt.xlabel('Class ID', fontsize=12, fontweight='bold')
    plt.ylabel('Pixel Count', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique)), [f'{int(cls)}\n{CLASS_NAMES[int(cls)]}' for cls in unique], 
               rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to: {save_path}")

# Utility functions
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def calculate_fps(model, input_size=(1, 3, 256, 256), num_iterations=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure time
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()
    
    fps = num_iterations / (end_time - start_time)
    return fps

def calculate_flops(model, input_size=(1, 3, 256, 256)):
    try:
        from thop import profile
        dummy_input = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(dummy_input,))
        return flops / 1e9  # Convert to GFLOPs
    except ImportError:
        print("thop not available for FLOP calculation")
        return 0.0

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_parameter_reduction():
    """Calculate theoretical parameter reduction from depthwise separable convolutions"""
    channel_configs = [32, 64, 128, 256, 512]  # Common channel sizes in the network
    reductions = []
    
    for c_out in channel_configs:
        standard_params = 9 * c_out * c_out  # Assuming same in/out channels for simplicity
        depthwise_params = 9 * c_out + c_out * c_out
        reduction_factor = standard_params / depthwise_params
        reductions.append(reduction_factor)
    
    avg_reduction = np.mean(reductions)
    return avg_reduction

# Loss Functions and Metrics
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_true * y_pred, dim=[2, 3])
        union = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3])
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# s2DS class weights
class_weights = torch.tensor([0.25, 2.5, 2.5, 2.0, 1.5, 1.5, 1.0]).to(device)

class CombinedLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.3, 0.2]):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.weights = weights

    def forward(self, inputs, targets):
        if isinstance(inputs, tuple):
            inputs = inputs[0]  # Take main output for single-output loss calculation
        if inputs.shape[2:] != targets.shape[1:]:
            inputs = F.interpolate(inputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
        return (self.weights[0] * self.ce_loss(inputs, targets).mean() +
                self.weights[1] * self.dice_loss(inputs, targets) +
                self.weights[2] * self.focal_loss(inputs, targets))

class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_criterion, ds_weights=[1.0, 0.4, 0.3, 0.2]):
        super(DeepSupervisionLoss, self).__init__()
        self.base_criterion = base_criterion
        self.ds_weights = ds_weights
        
    def forward(self, outputs, targets):
        if not isinstance(outputs, tuple):
            return self.base_criterion(outputs, targets)
            
        loss = 0
        # Main output loss
        loss += self.ds_weights[0] * self.base_criterion(outputs[0], targets)
        
        # Deep supervision losses
        for idx, output in enumerate(outputs[1:], 1):
            loss += self.ds_weights[idx] * self.base_criterion(output, targets)
            
        return loss

# Evaluation Metrics
def iou_score(y_pred, y_true, threshold=0.5):
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]  # Use main output for evaluation
    y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = (y_pred > threshold).float()
    y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

    intersection = torch.sum(y_true * y_pred, dim=[2, 3])
    union = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3]) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def f_score(y_pred, y_true, threshold=0.5, beta=1):
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]  # Use main output for evaluation
    y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = (y_pred > threshold).float()
    y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

    tp = torch.sum(y_true * y_pred, dim=[2, 3])
    fp = torch.sum((1 - y_true) * y_pred, dim=[2, 3])
    fn = torch.sum(y_true * (1 - y_pred), dim=[2, 3])

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-6)
    return f_score.mean()

def calculate_fwiou(predicted_mask, ground_truth_mask, class_frequencies):
    unique_classes = np.unique(ground_truth_mask)
    intersection_sum = 0
    union_sum = 0
    for class_val in unique_classes:
        intersection = np.sum((predicted_mask == class_val) & (ground_truth_mask == class_val))
        union = np.sum((predicted_mask == class_val) | (ground_truth_mask == class_val))
        frequency = class_frequencies.get(class_val, 1.0)
        intersection_sum += frequency * intersection
        union_sum += frequency * union
    fwiou = intersection_sum / union_sum if union_sum != 0 else 0
    return fwiou

# s2DS class frequencies
class_frequencies = {
    0: 0.2,  # Background
    1: 1.0,  # Crack
    2: 1.0,  # Spalling
    3: 1.0,  # Corrosion
    4: 0.7100,  # Efflorescence
    5: 0.6419,  # Vegetation
    6: 0.3518   # Control Point
}

def main():
    # Data paths for s2DS dataset
    TRAIN_FOLDER = '/home/mferdaus/EFPN_mefta/s2DS-data/s2ds/train'  
    VAL_FOLDER = '/home/mferdaus/EFPN_mefta/s2DS-data/s2ds/val'
    TEST_FOLDER = '/home/mferdaus/EFPN_mefta/s2DS-data/s2ds/test'

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # Create datasets with 256x256 size
    try:
        print("Loading s2DS dataset...")
        train_dataset = DefectDataset(TRAIN_FOLDER, size=(256, 256))
        val_dataset = DefectDataset(VAL_FOLDER, size=(256, 256))
        test_dataset = DefectDataset(TEST_FOLDER, size=(256, 256))
        
        print(f"✓ s2DS data loaded successfully!")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples") 
        print(f"  Test: {len(test_dataset)} samples")
        
    except Exception as e:
        print(f"❌ ERROR: s2DS data files not found: {e}")
        print("Please check the data paths and ensure the s2DS dataset is available.")
        return

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True)

    # Plot class distributions for s2DS
    print("Analyzing class distributions...")
    sample_masks_train = [train_dataset[i][1].numpy() for i in range(min(100, len(train_dataset)))]
    sample_masks_val = [val_dataset[i][1].numpy() for i in range(min(50, len(val_dataset)))]
    sample_masks_test = [test_dataset[i][1].numpy() for i in range(min(50, len(test_dataset)))]
    
    plot_class_distribution(np.array(sample_masks_train), 
                           's2DS Training Set Class Distribution', 
                           'results/plots/s2ds_train_class_distribution.png')
    plot_class_distribution(np.array(sample_masks_val), 
                           's2DS Validation Set Class Distribution', 
                           'results/plots/s2ds_val_class_distribution.png')
    plot_class_distribution(np.array(sample_masks_test), 
                           's2DS Test Set Class Distribution', 
                           'results/plots/s2ds_test_class_distribution.png')

    # Model initialization for s2DS (7 classes)
    model = EnhancedSAUNetWithKAN(
        in_channels=3,
        num_classes=7,  # s2DS has 7 classes
        keep_prob=0.9,
        start_filters=32,
        use_kan=True
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    theoretical_reduction = calculate_parameter_reduction()
    
    print(f"Total number of parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Theoretical depthwise separable reduction: ~{theoretical_reduction:.1f}x")
    print(f"KAN enabled: {TIKAN_AVAILABLE and model.use_kan}")
    print(f"Depthwise Separable Convolutions: ENABLED")
    
    model_size = get_model_size(model)
    fps = calculate_fps(model)
    flops = calculate_flops(model)
    
    print(f"Model size: {model_size:.2f} MB")
    print(f"FPS: {fps:.2f}")
    print(f"GFLOPs: {flops:.2f}")

    # Training setup
    base_criterion = CombinedLoss().to(device)
    criterion = DeepSupervisionLoss(base_criterion).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Result filename
    kan_suffix = "kan" if (TIKAN_AVAILABLE and model.use_kan) else "nokan"
    result_filename = f'results/enhanced-saunet-{kan_suffix}-depthwise-s2ds-with-plots.txt'
    
    # Save initial model info
    with open(result_filename, 'w') as f:
        f.write(f"Enhanced SAUNet with KAN Integration and Depthwise Separable Convolutions - s2DS Dataset\n")
        f.write(f"KAN enabled: {TIKAN_AVAILABLE and model.use_kan}\n")
        f.write(f"Depthwise Separable Convolutions: ENABLED\n")
        f.write(f"Total number of parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Theoretical parameter reduction: ~{theoretical_reduction:.1f}x\n")
        f.write(f"Model size: {model_size:.2f} MB\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"GFLOPs: {flops:.2f}\n\n")

    # Training Loop
    num_epochs = 50
    best_val_iou = 0.0

    print(f"Starting training for {num_epochs} epochs on s2DS dataset...")
    
    # Store training metrics for plotting
    train_losses = []
    train_ious = []
    train_f1s = []
    val_losses = []
    val_ious = []
    val_f1s = []
    
    with open(result_filename, 'a') as f:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            running_loss = 0.0
            running_iou = 0.0
            running_f_score = 0.0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                running_iou += iou_score(outputs, labels).item()
                running_f_score += f_score(outputs, labels).item()
                
                # Print progress every 20 batches
                if (batch_idx + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} "
                          f"- Loss: {loss.item():.4f}")

            # Average training metrics
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_iou = running_iou / len(train_loader)
            epoch_train_f1 = running_f_score / len(train_loader)
            
            train_losses.append(epoch_train_loss)
            train_ious.append(epoch_train_iou)
            train_f1s.append(epoch_train_f1)

            # Validation
            val_start_time = time.time()
            val_loss = 0.0
            val_iou = 0.0
            val_f_score = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_iou += iou_score(outputs, labels).item()
                    val_f_score += f_score(outputs, labels).item()
            
            val_time = time.time() - val_start_time
            epoch_time = time.time() - epoch_start_time

            # Average validation metrics
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_iou = val_iou / len(val_loader)
            epoch_val_f1 = val_f_score / len(val_loader)
            
            val_losses.append(epoch_val_loss)
            val_ious.append(epoch_val_iou)
            val_f1s.append(epoch_val_f1)

            scheduler.step()
            avg_val_iou = val_iou / len(val_loader)
            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                torch.save(model.state_dict(), 'best_model_enhanced_saunet_kan_depthwise_s2ds.pth')
                print(f"  🎯 New best model saved! IoU: {avg_val_iou:.4f}")

            log_line = f"Epoch {epoch + 1}/{num_epochs} (⏱️ {epoch_time:.1f}s) - " \
                       f"Loss: {epoch_train_loss:.4f}, " \
                       f"IOU: {epoch_train_iou:.4f}, " \
                       f"F-Score: {epoch_train_f1:.4f}, " \
                       f"Val Loss: {epoch_val_loss:.4f}, " \
                       f"Val IOU: {epoch_val_iou:.4f}, " \
                       f"Val F-Score: {epoch_val_f1:.4f}"
            print(f"✅ {log_line}")
            f.write(log_line + '\n')
            
            print("-" * 80)

    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_ious, label='Training IoU', color='blue')
    plt.plot(val_ious, label='Validation IoU', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.title('Training and Validation IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='Training F1', color='blue')
    plt.plot(val_f1s, label='Validation F1', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/s2ds_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training curves saved to: results/plots/s2ds_training_curves.png")

    # Load best model for evaluation
    try:
        model.load_state_dict(torch.load('best_model_enhanced_saunet_kan_depthwise_s2ds.pth'))
        print("Loaded best model for final evaluation")
    except:
        print("Using current model for evaluation")

    # Final Evaluation with Segmentation Plotting
    print("Starting final evaluation with segmentation plotting...")
    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    test_f_score = 0.0
    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Handle tuple output (deep supervision)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_iou += iou_score(outputs, labels).item()
            test_f_score += f_score(outputs, labels).item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_images.append(inputs.cpu().numpy())

    test_results = f"Test Loss: {test_loss/len(test_loader):.4f}, " \
                   f"Test IOU: {test_iou/len(test_loader):.4f}, " \
                   f"Test F-Score: {test_f_score/len(test_loader):.4f}"
    print(test_results)

    with open(result_filename, 'a') as f:
        f.write(test_results + '\n')

    # Convert predictions and ground truth to numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = np.concatenate(all_images, axis=0)

    # Create segmentation plots
    print("Creating segmentation result plots...")
    
    # Plot results for different samples
    num_plot_samples = min(8, len(all_images))
    indices_to_plot = np.linspace(0, len(all_images)-1, num_plot_samples, dtype=int)
    
    plot_images = [all_images[i].transpose(1, 2, 0) for i in indices_to_plot]
    plot_labels = [all_labels[i] for i in indices_to_plot]
    plot_preds = [all_preds[i] for i in indices_to_plot]
    
    plot_segmentation_results(
        plot_images, 
        plot_labels, 
        plot_preds, 
        'results/plots/enhanced_saunet_s2ds_segmentation_results.png',
        num_samples=num_plot_samples
    )

    # Calculate per-class metrics
    y_true_flat = all_labels.flatten()
    y_pred_flat = all_preds.flatten()
    unique_classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))

    # F1 score for each class
    f1_scores = {}
    iou_scores = {}
    for class_value in unique_classes:
        binary_y_true = (y_true_flat == class_value)
        binary_y_pred = (y_pred_flat == class_value)
        f1_scores[class_value] = f1_score(binary_y_true, binary_y_pred, zero_division=1)
        
        # IoU for each class
        intersection = np.sum(binary_y_true & binary_y_pred)
        union = np.sum(binary_y_true | binary_y_pred)
        iou_scores[class_value] = intersection / union if union != 0 else 0

    # Print and save F1 and IoU for each class
    with open(result_filename, 'a') as f:
        for class_value in unique_classes:
            f1_score_value = f1_scores[class_value]
            iou_score_value = iou_scores[class_value]
            class_result_line = f"Class {class_value} ({CLASS_NAMES[class_value]}): F1 Score = {f1_score_value:.4f}, IoU = {iou_score_value:.4f}"
            print(class_result_line)
            f.write(class_result_line + '\n')

        # Calculate averages
        average_F1_all = np.mean(list(f1_scores.values()))
        average_iou_all = np.mean(list(iou_scores.values()))
        
        # Average excluding class 0 (background)
        F1_scores_without_class_0 = {k: v for k, v in f1_scores.items() if k != 0}
        average_F1_without_class_0 = np.mean(list(F1_scores_without_class_0.values()))
        iou_scores_without_class_0 = {k: v for k, v in iou_scores.items() if k != 0}
        average_iou_without_class_0 = np.mean(list(iou_scores_without_class_0.values()))

        # Calculate additional metrics
        fwiou = calculate_fwiou(all_preds, all_labels, class_frequencies)
        balanced_accuracy = np.mean([np.sum((y_true_flat == c) & (y_pred_flat == c)) / np.sum(y_true_flat == c)
                                  for c in unique_classes if np.sum(y_true_flat == c) > 0])
        
        # MCC for each class
        mcc_scores = []
        for class_idx in range(7):
            class_true = (y_true_flat == class_idx)
            class_pred = (y_pred_flat == class_idx)
            mcc = matthews_corrcoef(class_true, class_pred)
            mcc_scores.append(mcc)
        mean_mcc = np.mean(mcc_scores)

        # Write final results
        f.write(f"\nFinal Results Summary:\n")
        f.write(f"Average F1 Score (including class 0): {average_F1_all:.4f}\n")
        f.write(f"Average F1 Score (excluding class 0): {average_F1_without_class_0:.4f}\n")
        f.write(f"Average IoU (including class 0): {average_iou_all:.4f}\n")
        f.write(f"Average IoU (excluding class 0): {average_iou_without_class_0:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"Mean MCC: {mean_mcc:.4f}\n")
        f.write(f"FWIoU: {fwiou:.4f}\n")
        f.write(f"\nModel Efficiency:\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Theoretical Parameter Reduction: ~{theoretical_reduction:.1f}x\n")
        f.write(f"Model Size: {model_size:.2f} MB\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"GFLOPs: {flops:.2f}\n")

    # Create performance bar plot
    plt.figure(figsize=(12, 8))
    
    classes = list(f1_scores.keys())
    f1_values = [f1_scores[cls] for cls in classes]
    iou_values = [iou_scores[cls] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, f1_values, width, label='F1 Score', alpha=0.8)
    plt.bar(x + width/2, iou_values, width, label='IoU Score', alpha=0.8)
    
    plt.xlabel('Class ID')
    plt.ylabel('Score')
    plt.title('Per-Class Performance: Enhanced SAUNet with KAN on s2DS')
    plt.xticks(x, [f'{int(cls)}\n{CLASS_NAMES[int(cls)]}' for cls in classes], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/s2ds_per_class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Per-class performance plot saved to: results/plots/s2ds_per_class_performance.png")

    print("\n" + "="*60)
    print("🚀 ENHANCED SAUNET ON s2DS DATASET - FINAL RESULTS:")
    print("="*60)
    print(f"✓ Enhanced SAUNet with KAN & Depthwise Separable Convs - Final Performance:")
    print(f"  - Average F1 (excl. class 0): {average_F1_without_class_0:.4f}")
    print(f"  - Average IoU (excl. class 0): {average_iou_without_class_0:.4f}")
    print(f"  - Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"  - FWIoU: {fwiou:.4f}")
    print(f"  - Parameter Efficiency: ~{theoretical_reduction:.1f}x reduction")
    print("="*60)
    print(f"\n📊 Results saved in: {result_filename}")
    print("📈 All visualization plots saved in: results/plots/")
    print("="*60)

if __name__ == "__main__":
    main()

