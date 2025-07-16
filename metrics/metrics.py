"""
Evaluation metrics for segmentation tasks
Includes IoU, F-score, and frequency-weighted IoU calculations.
"""

import torch
import torch.nn.functional as F
import numpy as np


def iou_score(y_pred, y_true, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) score
    
    Args:
        y_pred (torch.Tensor): Predicted segmentation logits
        y_true (torch.Tensor): Ground truth segmentation masks
        threshold (float): Threshold for converting probabilities to binary predictions
        
    Returns:
        torch.Tensor: Mean IoU score across all classes
    """
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]  # Use main output for evaluation
    
    # Ensure spatial dimensions match
    y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = (y_pred > threshold).float()
    y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

    intersection = torch.sum(y_true * y_pred, dim=[2, 3])
    union = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3]) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def f_score(y_pred, y_true, threshold=0.5, beta=1):
    """
    Calculate F-score (F1-score when beta=1)
    
    Args:
        y_pred (torch.Tensor): Predicted segmentation logits
        y_true (torch.Tensor): Ground truth segmentation masks
        threshold (float): Threshold for converting probabilities to binary predictions
        beta (float): Beta parameter for F-beta score
        
    Returns:
        torch.Tensor: Mean F-score across all classes
    """
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]  # Use main output for evaluation
    
    # Ensure spatial dimensions match
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
    """
    Calculate Frequency Weighted Intersection over Union (FWIoU)
    
    Args:
        predicted_mask (np.ndarray): Predicted segmentation mask
        ground_truth_mask (np.ndarray): Ground truth segmentation mask
        class_frequencies (dict): Dictionary mapping class indices to their frequencies
        
    Returns:
        float: Frequency weighted IoU score
    """
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


def calculate_per_class_metrics(y_true_flat, y_pred_flat, num_classes):
    """
    Calculate per-class IoU and F1 scores
    
    Args:
        y_true_flat (np.ndarray): Flattened ground truth labels
        y_pred_flat (np.ndarray): Flattened predicted labels
        num_classes (int): Number of classes
        
    Returns:
        tuple: (f1_scores_dict, iou_scores_dict)
    """
    from sklearn.metrics import f1_score as sklearn_f1
    
    unique_classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
    f1_scores = {}
    iou_scores = {}
    
    for class_value in unique_classes:
        if class_value < num_classes:
            binary_y_true = (y_true_flat == class_value)
            binary_y_pred = (y_pred_flat == class_value)
            
            # F1 score
            f1_scores[class_value] = sklearn_f1(binary_y_true, binary_y_pred, zero_division=1)
            
            # IoU score
            intersection = np.sum(binary_y_true & binary_y_pred)
            union = np.sum(binary_y_true | binary_y_pred)
            iou_scores[class_value] = intersection / union if union != 0 else 0
    
    return f1_scores, iou_scores


def calculate_mean_metrics(f1_scores, iou_scores, exclude_background=True):
    """
    Calculate mean metrics with option to exclude background class
    
    Args:
        f1_scores (dict): Per-class F1 scores
        iou_scores (dict): Per-class IoU scores
        exclude_background (bool): Whether to exclude class 0 (background)
        
    Returns:
        dict: Dictionary containing mean metrics
    """
    if exclude_background:
        f1_values = [v for k, v in f1_scores.items() if k != 0]
        iou_values = [v for k, v in iou_scores.items() if k != 0]
    else:
        f1_values = list(f1_scores.values())
        iou_values = list(iou_scores.values())
    
    return {
        'mean_f1': np.mean(f1_values) if f1_values else 0.0,
        'mean_iou': np.mean(iou_values) if iou_values else 0.0,
        'mean_f1_with_bg': np.mean(list(f1_scores.values())),
        'mean_iou_with_bg': np.mean(list(iou_scores.values()))
    }

