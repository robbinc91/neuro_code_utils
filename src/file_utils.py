from pathlib import Path
from typing import List, Tuple
import nibabel as nib
import numpy as np


def list_nifti_files(root_dir: str, recursive: bool = False) -> List[str]:
    p = Path(root_dir)
    if recursive:
        patterns = ("**/*.nii", "**/*.nii.gz")
    else:
        patterns = ("*.nii", "*.nii.gz")
    files = []
    for pat in patterns:
        files.extend([str(x) for x in p.glob(pat)])
    return sorted(files)


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, object]:
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header


def save_nifti(data: np.ndarray, affine: np.ndarray, header: object, out_path: str) -> None:
    out_img = nib.Nifti1Image(data, affine, header)
    nib.save(out_img, out_path)
