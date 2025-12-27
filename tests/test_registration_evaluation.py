import numpy as np
import nibabel as nib
import tempfile
import os
from scipy.ndimage import shift

from src.registration_evaluation import evaluate_registration


def _make_blob(shape=(64, 64, 48), center=None, sigma=6.0):
    if center is None:
        center = np.array(shape) / 2.0
    grid = np.indices(shape).astype(np.float32)
    coords = np.stack([grid[i] for i in range(len(shape))], axis=-1)
    diff = coords - center
    rad2 = np.sum(diff ** 2, axis=-1)
    blob = np.exp(-rad2 / (2 * sigma * sigma))
    return blob


def test_evaluate_registration_translation(tmp_path):
    fixed = _make_blob()
    # warp by small translation
    moved = shift(fixed, shift=(2, -1, 0), order=1)

    fpath = tmp_path / 'fixed.nii.gz'
    mpath = tmp_path / 'moved.nii.gz'
    nib.save(nib.Nifti1Image(fixed.astype(np.float32), np.eye(4)), str(fpath))
    nib.save(nib.Nifti1Image(moved.astype(np.float32), np.eye(4)), str(mpath))

    res = evaluate_registration(str(fpath), str(mpath))
    assert 'metrics' in res
    assert 0.0 <= res['final_score'] <= 1.0
    # for small translation expect moderate-high score
    assert res['final_score'] > 0.4
