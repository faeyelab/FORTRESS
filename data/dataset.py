"""
Dataset loading utilities for structural defect segmentation
Supports s2DS dataset format with proper class remapping.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DefectDataset(Dataset):
    """
    Dataset class for structural defect segmentation
    
    Supports the s2DS (Structural Defects Dataset) format with automatic
    class value remapping and proper image/mask loading.
    """
    
    def __init__(self, folder_path, transform=None, size=(256, 256)):
        """
        Initialize the dataset
        
        Args:
            folder_path (str): Path to the dataset folder
            transform (callable, optional): Optional transform to be applied on images
            size (tuple): Target size for resizing images and masks
        """
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
        """
        Remap mask values from original s2DS values to consecutive class indices
        
        Args:
            mask (np.ndarray): Original mask with s2DS class values
            
        Returns:
            np.ndarray: Remapped mask with consecutive class indices
        """
        remapped = np.zeros_like(mask, dtype=np.uint8)
        for orig, remap in zip(self.original_values, self.remapped_values):
            remapped[mask == orig] = remap
        return remapped

    def __getitem__(self, idx):
        """
        Get a single item from the dataset
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: (image, mask) where image is a tensor and mask is a long tensor
        """
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


# Dataset configuration constants
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

# s2DS class frequencies for weighted loss computation
CLASS_FREQUENCIES = {
    0: 0.2,      # Background
    1: 1.0,      # Crack
    2: 1.0,      # Spalling
    3: 1.0,      # Corrosion
    4: 0.7100,   # Efflorescence
    5: 0.6419,   # Vegetation
    6: 0.3518    # Control Point
}

