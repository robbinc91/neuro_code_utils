# Neuro Code Utils

A collection of Python utilities for deep learning applications in neuroscience, providing tools for medical image processing, model conversion, and data augmentation specifically designed for neuroimaging workflows.

## Overview

This repository contains practical utilities that bridge the gap between deep learning frameworks and neuroscience research, with a focus on MRI data processing and model interoperability.

## Repository Structure

```
neuro_code_utils/
├── src/
│   ├── InstanceNormalization.py
│   ├── InstanceNormalizationTF.py
│   ├── keras2torch.py
│   ├── MRIDataAugmenter.py
│   ├── n4_and_registration.py
│   ├── na_and_reg_better.py
│   ├── onnx2keras.py
│   └── torch2onnx.py
└── README.md
```

## File Descriptions

### 🧠 Image Processing & Registration

#### `n4_and_registration.py`
**Medical Image Preprocessing Pipeline**
- Performs N4 bias field correction for intensity inhomogeneity
- Brain extraction using PyRobex
- Rigid registration to template space using ANTs
- Handles complete preprocessing workflow for MRI data

#### `na_and_reg_better.py` 
**Enhanced Registration Utilities**
- Modular functions for MRI loading and N4 correction
- Brain mask extraction and template alignment
- Bidirectional transformation support (forward + inverse)
- Save/load transformation parameters for reproducible processing

#### `MRIDataAugmenter.py`
**MRI-Specific Data Augmentation**
- 3D spatial transformations for volumetric data
- Translation, rotation, and flipping augmentations
- Configurable probability settings for each transformation
- NIfTI file format support with nibabel integration

### 🔄 Model Conversion & Interoperability

#### `torch2onnx.py`
**PyTorch to ONNX Export**
- Converts PyTorch models to ONNX format
- Example using ResNet18 from torchvision
- Preserves model architecture and weights for cross-framework deployment

#### `onnx2keras.py`
**ONNX to Keras Conversion**
- Imports ONNX models into Keras/TensorFlow
- Maintains layer structure and parameters
- Enables framework switching for trained models

#### `keras2torch.py`
**Keras to PyTorch Weight Transfer**
- Transfers weights from Keras to PyTorch models
- Supports Conv2D, Linear, and BatchNorm layers
- Handles parameter transposition between frameworks

### 🧩 Normalization Layers

#### `InstanceNormalization.py`
**PyTorch Implementation**
- Custom Instance Normalization layer
- Computes mean and variance across spatial dimensions
- Essential for style transfer and generative models

#### `InstanceNormalizationTF.py`
**TensorFlow Implementation**
- Trainable Instance Normalization with scale and offset parameters
- Integrated as Keras custom layer
- Compatible with TF2.x and eager execution

## Installation & Dependencies

```bash
# Core deep learning frameworks
pip install torch torchvision tensorflow

# Medical image processing
pip install nibabel ants-py pyrobex

# Additional utilities
pip install onnx onnx2keras scipy
```

## Usage Examples

### Medical Image Preprocessing
```python
from src.na_and_reg_better import n4_correction, rigid_registration

# Correct intensity inhomogeneity
corrected_mri = n4_correction(original_mri)

# Register to template space
transform = rigid_registration(template_mask, brain_mask, "transform.mat")
```

### Model Conversion
```python
# Convert PyTorch model to ONNX
from src.torch2onnx import export_model

# Use converted model in Keras
from src.onnx2keras import load_keras_model
```

### Data Augmentation
```python
from src.MRIDataAugmenter import MRIDataAugmenter

augmenter = MRIDataAugmenter()
augmented_scan = augmenter.augment_image("scan.nii.gz")
```

## Applications

- **Neuroimaging Research**: Preprocess MRI data for deep learning
- **Multi-framework Development**: Convert models between PyTorch, TensorFlow, and ONNX
- **Data Augmentation**: Generate synthetic training data for medical images
- **Model Deployment**: Export models to standardized formats for inference

## Contributing

Contributions are welcome! Please feel free to submit pull requests for:
- Additional normalization layers
- More medical image processing utilities
- Enhanced model conversion capabilities
- Bug fixes and performance improvements

## License

This project is licensed under the MIT License.

## Citation

If you use these utilities in your research, please consider citing:

```bibtex
@software{neuro_code_utils,
  title = {Neuro Code Utils: Deep Learning Utilities for Neuroscience},
  author = {Robin Cabeza Ruiz},
  year = {2024},
  url = {https://github.com/yourusername/neuro_code_utils}
}
```

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.