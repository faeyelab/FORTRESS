"""
Loss function implementations for FORTRESS model training
Includes Dice loss, Focal loss, and combined loss functions with deep supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    
    Computes the Dice coefficient between predicted and ground truth masks,
    particularly effective for handling class imbalance in segmentation.
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Ensure spatial dimensions match
        y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_true * y_pred, dim=[2, 3])
        union = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3])
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focuses learning on hard examples by down-weighting easy examples,
    particularly useful for segmentation tasks with severe class imbalance.
    """
    
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Ensure spatial dimensions match
        y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function using CrossEntropy, Dice, and Focal losses
    
    Combines multiple loss functions with configurable weights to leverage
    the benefits of each loss type for robust training.
    """
    
    def __init__(self, class_weights=None, weights=[0.5, 0.3, 0.2]):
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
    """
    Deep supervision loss for multi-output training
    
    Applies loss computation to multiple intermediate outputs with different weights,
    improving gradient flow and training convergence.
    """
    
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
            if idx < len(self.ds_weights):
                loss += self.ds_weights[idx] * self.base_criterion(output, targets)
            
        return loss


# Default class weights for s2DS dataset
def get_s2ds_class_weights(device='cuda'):
    """
    Get class weights for s2DS dataset to handle class imbalance
    
    Args:
        device (str): Device to place the tensor on
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    class_weights = torch.tensor([0.25, 2.5, 2.5, 2.0, 1.5, 1.5, 1.0])
    return class_weights.to(device)

