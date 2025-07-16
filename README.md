# FORTRESS: Function-composition Optimized Real-Time Resilient Structural Segmentation

FORTRESS (Function-composition Optimized Real-Time Resilient Structural Segmentation) is a highly efficient semantic segmentation framework designed specifically for structural defect detection in civil infrastructure. It combines depthwise separable convolutions with adaptive Kolmogorov-Arnold Network integration to achieve remarkable efficiency gains while maintaining superior segmentation performance.

Official implementation of the paper "FORTRESS: Function-composition Optimized Real-Time Resilient Structural Segmentation via Kolmogorov-Arnold Enhanced Spatial Attention Networks" submitted to Journal of LaTeX Class Files.

## Key Innovations

* **Systematic depthwise separable convolution framework**: Achieves 3.6× parameter reduction per layer while maintaining feature representation quality
* **Adaptive TiKAN integration**: Selectively applies function composition transformations only when computationally beneficial
* **Multi-scale attention fusion**: Combines spatial, channel, and KAN-enhanced features across decoder levels for comprehensive defect analysis

## Performance Highlights

* **High Accuracy**: F1-score of 0.771 and mean IoU of 0.677 on benchmark infrastructure datasets
* **Parameter Efficiency**: 91% parameter reduction (31M → 2.9M parameters)
* **Computational Efficiency**: 91% computational complexity reduction (13.7 → 1.17 GFLOPs)
* **Speed Improvement**: 3× inference speed improvement for real-time deployment

## Model Architecture

FORTRESS integrates Kolmogorov-Arnold modules into an enhanced U-Net architecture with spatial attention:

* Enhanced encoder with depthwise separable convolutions and KAN integration
* Multi-scale spatial attention mechanisms at multiple decoder levels
* Adaptive KAN processing for feature refinement
* Deep supervision for improved training convergence
* Residual connections for gradient flow optimization

## Installation

```shell
# Clone the repository
git clone https://github.com/yourusername/fortress-paper-code.git
cd fortress-paper-code

# Create conda environment (optional)
conda create -n fortress python=3.8
conda activate fortress

# Install dependencies
pip install -r requirements.txt

# Install tikan for KAN functionality (optional)
pip install tikan
```

## Usage

### Training

```shell
python train.py --config config/default.yaml --dataset s2ds
```

### Inference

```python
import torch
from models.fortress import EnhancedSAUNetWithKAN

# Load model
model = EnhancedSAUNetWithKAN(
    in_channels=3,
    num_classes=7,  # For s2DS dataset
    use_kan=True
)
model.load_state_dict(torch.load('path/to/weights.pth'))
model.eval()

# Inference
with torch.no_grad():
    input_tensor = torch.randn(1, 3, 256, 256)  # Example input
    output = model(input_tensor)
    if isinstance(output, tuple):
        output = output[0]  # Main output for inference
    prediction = torch.argmax(output, dim=1)
```

## Datasets

FORTRESS is evaluated on challenging structural defect datasets:

### Structural Defects Dataset (S2DS)

Contains 743 high-resolution images of concrete surfaces with pixel-wise annotations across seven distinct classes:

* **Class 0**: Background
* **Class 1**: Crack (linear fractures)
* **Class 2**: Spalling (surface detachment)
* **Class 3**: Corrosion (rust stains)
* **Class 4**: Efflorescence (chemical deposits)
* **Class 5**: Vegetation (plant growth)
* **Class 6**: Control Point (fiducial markers)

#### Dataset Organization

After downloading, organize the dataset with the following structure:

```
data/
├── s2ds/
    ├── train/
    │   ├── image1.png
    │   ├── image1_lab.png
    │   └── ...
    ├── val/
    │   ├── image1.png
    │   ├── image1_lab.png
    │   └── ...
    └── test/
        ├── image1.png
        ├── image1_lab.png
        └── ...
```

## Project Structure

```
fortress-paper-code/
├── README.md
├── requirements.txt
├── LICENSE
├── train.py                    # Main training script
├── config/
│   └── default.yaml           # Configuration file
├── data/
│   └── dataset.py             # Dataset loading utilities
├── models/
│   ├── __init__.py
│   ├── fortress.py            # Main FORTRESS model
│   ├── layers.py              # Custom layer implementations
│   └── kan_layers.py          # KAN-specific components
├── losses/
│   ├── __init__.py
│   └── losses.py              # Loss function implementations
├── metrics/
│   ├── __init__.py
│   └── metrics.py             # Evaluation metrics
├── utils/
│   ├── __init__.py
│   ├── utils.py               # General utilities
│   └── visualization.py      # Plotting and visualization
└── results/
    └── plots/                 # Generated plots and visualizations
```

## Key Features

### Depthwise Separable Convolutions

FORTRESS employs depthwise separable convolutions throughout the architecture to achieve significant parameter reduction:

```python
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
```

### Adaptive KAN Integration

The model selectively applies KAN transformations based on computational constraints:

```python
# KAN enhancement is applied only when beneficial
if self.use_kan and spatial_size <= threshold:
    features = self.kan_block(features, H, W)
```

### Multi-Scale Attention

Spatial attention mechanisms are applied at multiple scales for comprehensive feature enhancement:

```python
# Multi-scale spatial attention
e5 = self.spatial_attention1(e5)  # 7x7 kernel
e4 = self.spatial_attention2(e4)  # 5x5 kernel  
e3 = self.spatial_attention3(e3)  # 3x3 kernel
```

## Training Configuration

The model supports various training configurations:

* **Optimizer**: AdamW with weight decay
* **Learning Rate**: Cosine annealing scheduler
* **Loss Function**: Combined loss (CrossEntropy + Dice + Focal)
* **Deep Supervision**: Multi-level supervision for improved convergence
* **Data Augmentation**: Albumentations-based augmentation pipeline

## Results

### Quantitative Results

| Method | Parameters | GFLOPs | F1-Score | mIoU | FPS |
|--------|------------|--------|----------|------|-----|
| U-Net | 31.0M | 13.7 | 0.652 | 0.589 | 45.2 |
| SA-UNet | 28.5M | 12.1 | 0.698 | 0.634 | 52.1 |
| U-KAN | 25.2M | 11.8 | 0.721 | 0.651 | 38.7 |
| **FORTRESS** | **2.9M** | **1.17** | **0.771** | **0.677** | **135.6** |

### Efficiency Analysis

* **Parameter Reduction**: 91% reduction compared to baseline U-Net
* **Computational Reduction**: 91% GFLOPs reduction
* **Speed Improvement**: 3× faster inference
* **Memory Efficiency**: Significantly reduced memory footprint

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{fortress2025,
  title={FORTRESS: Function-composition Optimized Real-Time Resilient Structural Segmentation via Kolmogorov-Arnold Enhanced Spatial Attention Networks},
  author={Thrainer, Christina and Ferdaus, Md Meftahul and Abdelguerfi, Mahdi and Guetl, Christian and Sloan, Steven and Niles, Kendall N. and Pathak, Ken},
  journal={Submitted to IEEE Transactions on Cybernetics},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* The authors acknowledge the support of the US Army Corps of Engineers, Engineer Research and Development Center
* Special thanks to the Canizaro Livingston Gulf States Center for Environmental Informatics at the University of New Orleans
* Graz University of Technology for collaborative research support

## Contact

For questions and collaborations, please contact:
- Md Meftahul Ferdaus: mferdaus@uno.edu


## About

FORTRESS represents a significant advancement in automated structural defect segmentation, combining theoretical foundations from the Kolmogorov-Arnold representation theorem with practical engineering requirements for real-time infrastructure monitoring. The framework addresses critical challenges in civil infrastructure inspection by providing both high accuracy and computational efficiency necessary for deployment in resource-constrained field environments.

## Languages

* Python 100.0%

