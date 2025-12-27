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

#### `optimal_registration.py`
**Optimal 3D Brain MRI Registration**
- High-level 3-stage registration pipeline: Rigid -> Affine -> SyN (non-linear)
- Optional N4 bias-field correction and skull-stripping (PyRobex)
- Applies final transforms to the full MRI and saves warped output
- Usage: `optimal_brain_registration(mri_path, template_path, output_dir, do_n4=True)`

#### `simple_registration.py`
**SimpleITK-based Registration (no ANTs / no PyRobex)**
- Lightweight alternative using `SimpleITK` only
- Optional N4 bias-field correction (SimpleITK), Otsu-based skull-stripping, and a 3-stage registration: Rigid -> Affine -> BSpline
- Usage: `simple_brain_registration(mri_path, template_path, output_dir, do_n4=True)`

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

#### `batch_augment.py`
**Batch Augmentation CLI**
- Batch-augment NIfTI files using `MRIDataAugmenter`.
- Preserves original affine and header when saving augmented volumes.
- Saves outputs as `<original>_aug<N>.nii.gz`.

CLI example:
```bash
python src/batch_augment.py input_folder output_folder -n 5 -r
```
Programmatic usage:
```python
from src.batch_augment import batch_augment
batch_augment("input_folder", "output_folder", n_augment=5, recursive=True)
```

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

#### `keras2tfjs.py`
**Keras -> TensorFlow.js Conversion**
- Convert an in-memory `tf.keras.Model` or saved Keras model (`.h5` / SavedModel)
- Uses the `tensorflowjs` Python API when available, or falls back to the `tensorflowjs_converter` CLI
- Example: `convert_keras_model_to_tfjs(model_or_path, output_dir)`

#### `torch_to_tfjs.py`
**PyTorch -> TensorFlow.js Conversion (via ONNX)**
- Exports a PyTorch model to ONNX, converts ONNX -> Keras with `onnx2keras`, then writes TFJS layers format using the Keras -> TFJS helper
- Convenience function: `convert_torch_model_to_tfjs(torch_model, example_input, output_dir)`

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

### 🧠 File utilities & Visualization

#### `file_utils.py`
**NIfTI helpers**
- `list_nifti_files`, `load_nifti`, `save_nifti` helpers for common IO tasks

#### `visualize_segmentation.py`
**MRI + Segmentation Visualization**
- Create PNG overlays of MRI slices with multi-label segmentation colorized
- Iterate through axes (0,1,2) and save `n_slices` per axis with legends for label values
- Example: `visualize_segmentation(mri_path, seg_path, out_dir, n_slices=8)`


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