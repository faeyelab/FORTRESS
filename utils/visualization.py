"""
Visualization utilities for segmentation results and dataset analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import torch

from ..data.dataset import COLORS, CLASS_NAMES


def mask_to_rgb(mask, colors=COLORS):
    """
    Convert segmentation mask to RGB visualization
    
    Args:
        mask (np.ndarray): Segmentation mask with class indices
        colors (list): List of RGB colors for each class
        
    Returns:
        np.ndarray: RGB visualization of the mask
    """
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        rgb_mask[mask == class_id] = color
    return rgb_mask


def plot_segmentation_results(images, ground_truths, predictions, save_path, num_samples=8):
    """
    Plot segmentation results with original image, ground truth, and prediction
    
    Args:
        images (list): List of input images
        ground_truths (list): List of ground truth masks
        predictions (list): List of predicted masks
        save_path (str): Path to save the plot
        num_samples (int): Number of samples to plot
    """
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
    """
    Plot class distribution in the dataset
    
    Args:
        y_data (np.ndarray): Array of mask data
        title (str): Plot title
        save_path (str): Path to save the plot
    """
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


def plot_training_curves(train_losses, val_losses, train_ious, val_ious, train_f1s, val_f1s, save_path):
    """
    Plot training curves for loss, IoU, and F1 score
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_ious (list): Training IoU scores
        val_ious (list): Validation IoU scores
        train_f1s (list): Training F1 scores
        val_f1s (list): Validation F1 scores
        save_path (str): Path to save the plot
    """
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_per_class_performance(f1_scores, iou_scores, save_path):
    """
    Plot per-class performance metrics
    
    Args:
        f1_scores (dict): Per-class F1 scores
        iou_scores (dict): Per-class IoU scores
        save_path (str): Path to save the plot
    """
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
    plt.title('Per-Class Performance: FORTRESS Model')
    plt.xticks(x, [f'{int(cls)}\n{CLASS_NAMES[int(cls)]}' for cls in classes], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class performance plot saved to: {save_path}")


def create_comparison_table(results_dict, save_path=None):
    """
    Create a comparison table of different models
    
    Args:
        results_dict (dict): Dictionary with model names as keys and metrics as values
        save_path (str, optional): Path to save the table as image
    """
    import pandas as pd
    
    df = pd.DataFrame(results_dict).T
    
    if save_path:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        rowLabels=df.index, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison table saved to: {save_path}")
    
    return df

