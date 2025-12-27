import os
from pathlib import Path
import numpy as np
import nibabel as nib

from src.batch_augment import batch_augment


def make_dummy_nifti(path: str, shape=(16, 16, 8)):
    data = np.zeros(shape, dtype=np.float32)
    data[4:12, 4:12, 2:6] = 1.0
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, path)


def test_batch_augment_creates_outputs(tmp_path):
    inp = tmp_path / "input"
    out = tmp_path / "output"
    inp.mkdir()
    out.mkdir()

    nifti_path = inp / "test.nii.gz"
    make_dummy_nifti(str(nifti_path))

    created = batch_augment(str(inp), str(out), n_augment=3, recursive=False)

    assert created == 3
    files = list(out.glob("*.nii.gz"))
    assert len(files) == 3
