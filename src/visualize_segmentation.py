import os
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def _normalize_image(slice_arr: np.ndarray) -> np.ndarray:
    # Scale image to 0-1 using 1-99 percentiles to reduce outlier impact
    lo, hi = np.percentile(slice_arr, (1, 99))
    if hi - lo <= 0:
        return np.clip(slice_arr, 0, 1)
    norm = (slice_arr - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def visualize_segmentation(
    mri_path: str,
    seg_path: str,
    out_dir: str,
    n_slices: int = 8,
    alpha: float = 0.5,
    figsize: tuple = (6, 6),
):
    """Create overlay visualizations of a 3D MRI with segmentation labels.

    Saves PNG images into `out_dir`, iterating through the three axes (0,1,2)
    and creating `n_slices` slices per axis. Multi-label segmentations are
    colored with distinct colors; background label 0 is not shown.

    Args:
        mri_path: Path to the MRI NIfTI file.
        seg_path: Path to the segmentation NIfTI file (integer labels).
        out_dir: Directory to save the PNG visualizations.
        n_slices: Number of slices to generate per axis.
        alpha: Alpha blending for the label overlay.
        figsize: Figure size for saved images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mri_img = nib.load(mri_path)
    seg_img = nib.load(seg_path)

    mri = mri_img.get_fdata()
    seg = seg_img.get_fdata()

    if mri.shape != seg.shape:
        raise ValueError("MRI and segmentation volumes must have the same shape")

    # iterate axes
    for axis in range(3):
        dim = mri.shape[axis]
        if n_slices <= 0:
            continue
        indices = np.linspace(0, dim - 1, num=n_slices, dtype=int)
        for i, idx in enumerate(indices):
            if axis == 0:
                m_slice = mri[idx, :, :]
                s_slice = seg[idx, :, :]
            elif axis == 1:
                m_slice = mri[:, idx, :]
                s_slice = seg[:, idx, :]
            else:
                m_slice = mri[:, :, idx]
                s_slice = seg[:, :, idx]

            m_norm = _normalize_image(m_slice)

            plt.figure(figsize=figsize)
            plt.imshow(m_norm.T, cmap='gray', origin='lower')

            unique_labels = np.unique(s_slice.astype(int))
            unique_labels = unique_labels[unique_labels != 0]

            legend_patches = []
            if unique_labels.size > 0:
                cmap_base = plt.get_cmap('tab20')
                colors = cmap_base(np.linspace(0, 1, unique_labels.size))
                cmap = ListedColormap(colors)

                # map labels to 1..N for display
                mapped = np.zeros_like(s_slice, dtype=int)
                for idx_map, lab in enumerate(unique_labels, start=1):
                    mapped[s_slice == lab] = idx_map
                    legend_patches.append(mpatches.Patch(color=colors[idx_map - 1], label=str(int(lab))))

                masked = np.ma.masked_where(mapped == 0, mapped)
                plt.imshow(masked.T, cmap=cmap, alpha=alpha, origin='lower')

            plt.axis('off')
            title = f'Axis {axis} slice {int(idx)}'
            plt.title(title)
            if legend_patches:
                plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

            out_path = out_dir / f'axis{axis}_slice{int(idx)}.png'
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

    return str(out_dir)
