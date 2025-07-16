"""
General utility functions for model analysis and performance measurement
"""

import torch
import time
import numpy as np


def get_model_size(model):
    """
    Calculate model size in MB
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        float: Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def calculate_fps(model, input_size=(1, 3, 256, 256), num_iterations=100, device='cuda'):
    """
    Calculate model inference speed in FPS
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_size (tuple): Input tensor size
        num_iterations (int): Number of iterations for timing
        device (str): Device to run inference on
        
    Returns:
        float: Frames per second
    """
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure time
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    end_time = time.time()
    fps = num_iterations / (end_time - start_time)
    return fps


def calculate_flops(model, input_size=(1, 3, 256, 256), device='cuda'):
    """
    Calculate model FLOPs using thop library
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_size (tuple): Input tensor size
        device (str): Device to run calculation on
        
    Returns:
        float: GFLOPs (Giga floating point operations)
    """
    try:
        from thop import profile
        dummy_input = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(dummy_input,))
        return flops / 1e9  # Convert to GFLOPs
    except ImportError:
        print("thop not available for FLOP calculation")
        return 0.0


def count_parameters(model):
    """
    Count total and trainable parameters in a model
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_parameter_reduction():
    """
    Calculate theoretical parameter reduction from depthwise separable convolutions
    
    Returns:
        float: Average reduction factor
    """
    channel_configs = [32, 64, 128, 256, 512]  # Common channel sizes
    reductions = []
    
    for c_out in channel_configs:
        # Assuming same in/out channels for simplicity
        standard_params = 9 * c_out * c_out  # 3x3 conv
        depthwise_params = 9 * c_out + c_out * c_out  # 3x3 depthwise + 1x1 pointwise
        reduction_factor = standard_params / depthwise_params
        reductions.append(reduction_factor)
    
    avg_reduction = np.mean(reductions)
    return avg_reduction


def setup_reproducibility(seed=42):
    """
    Setup reproducible training environment
    
    Args:
        seed (int): Random seed
    """
    import random
    import os
    
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


def print_model_summary(model, input_size=(1, 3, 256, 256), device='cuda'):
    """
    Print comprehensive model summary
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_size (tuple): Input tensor size
        device (str): Device for calculations
    """
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)
    fps = calculate_fps(model, input_size, device=device)
    flops = calculate_flops(model, input_size, device=device)
    reduction = calculate_parameter_reduction()
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Inference Speed: {fps:.2f} FPS")
    print(f"Computational Cost: {flops:.2f} GFLOPs")
    print(f"Theoretical Parameter Reduction: ~{reduction:.1f}x")
    print("=" * 60)

