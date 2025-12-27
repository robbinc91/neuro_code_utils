"""Registration evaluation utilities.

Provides `evaluate_registration` which computes five metrics between a fixed
image and a warped-moving image and returns a combined score in [0,1].

Metrics computed:
 - Dice coefficient between automatic brain masks (Otsu)
 - Normalized Mutual Information (NMI)
 - Normalized Cross-Correlation (NCC)
 - Gradient magnitude Pearson correlation
 - Histogram intersection

All metrics are mapped to [0,1] and averaged equally for a final score.
"""
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import stats
from scipy import ndimage
import json


def _otsu_threshold(data: np.ndarray, nbins: int = 256) -> float:
    # Otsu's threshold implementation
    data = data[np.isfinite(data)]
    data = data[data > 0]  # ignore zeros
    if data.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(data.ravel(), bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2, 1))[::-1]
    var_between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(var_between)
    return bin_centers[idx]


def _dice_from_thresholds(fixed: np.ndarray, moving: np.ndarray, nbins: int = 256) -> float:
    t1 = _otsu_threshold(fixed, nbins=nbins)
    t2 = _otsu_threshold(moving, nbins=nbins)
    m1 = (fixed > t1).astype(np.uint8)
    m2 = (moving > t2).astype(np.uint8)
    inter = np.sum((m1 & m2).astype(np.float64))
    denom = np.sum(m1) + np.sum(m2)
    if denom == 0:
        return 0.0
    return 2.0 * inter / denom


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    # compute joint histogram
    x = x.ravel()
    y = y.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return 0.0
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    pxy = hist / np.sum(hist)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / (px[:, None][nzs] * py[None, :][nzs])))
    # normalize by average entropy
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    if hx + hy == 0:
        return 0.0
    nmi = 2.0 * mi / (hx + hy)
    return float(np.clip(nmi, 0.0, 1.0))


def _ncc(x: np.ndarray, y: np.ndarray) -> float:
    x = x.ravel()
    y = y.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (np.sqrt(np.sum(xm * xm)) * np.sqrt(np.sum(ym * ym)))
    if denom == 0:
        return 0.0
    return float(np.sum(xm * ym) / denom)


def _gradient_correlation(x: np.ndarray, y: np.ndarray) -> float:
    # compute gradient magnitude and Pearson correlation
    gx = np.sqrt(np.sum(np.stack(np.gradient(x), axis=0) ** 2, axis=0))
    gy = np.sqrt(np.sum(np.stack(np.gradient(y), axis=0) ** 2, axis=0))
    gx = gx.ravel()
    gy = gy.ravel()
    mask = np.isfinite(gx) & np.isfinite(gy)
    if np.sum(mask) == 0:
        return 0.0
    r, _ = stats.pearsonr(gx[mask], gy[mask])
    if np.isnan(r):
        return 0.0
    return float(r)


def _histogram_intersection(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    x = x.ravel()
    y = y.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return 0.0
    h1, _ = np.histogram(x, bins=bins, density=True)
    h2, _ = np.histogram(y, bins=bins, density=True)
    # histogram intersection
    inter = np.sum(np.minimum(h1, h2))
    # since histograms are density, intersection in [0,1]
    return float(np.clip(inter, 0.0, 1.0))


def evaluate_registration(
    fixed_path: str,
    warped_moving_path: str,
    mask_fixed_path: Optional[str] = None,
    mask_moving_path: Optional[str] = None,
    bins: int = 64,
    deformation_field_path: Optional[str] = None,
    landmarks_path: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Evaluate registration quality between a fixed image and a warped moving image.

    Returns a dictionary containing per-metric scores and a final combined score
    in range [0,1].
    """
    fixed_nii = nib.load(fixed_path)
    warped_nii = nib.load(warped_moving_path)
    fixed = fixed_nii.get_fdata()
    warped = warped_nii.get_fdata()

    # Use provided masks or compute Otsu masks
    if mask_fixed_path is not None:
        mask_fixed = nib.load(mask_fixed_path).get_fdata() > 0
    else:
        t = _otsu_threshold(fixed)
        mask_fixed = fixed > t

    if mask_moving_path is not None:
        mask_moving = nib.load(mask_moving_path).get_fdata() > 0
    else:
        t = _otsu_threshold(warped)
        mask_moving = warped > t

    # restrict evaluation to union of masks to avoid background issues
    eval_mask = (mask_fixed | mask_moving)
    if np.sum(eval_mask) == 0:
        raise ValueError("Empty evaluation mask — check inputs or thresholds.")

    fixed_eval = fixed[eval_mask]
    warped_eval = warped[eval_mask]

    # Metric 1: Dice on brain masks
    dice = float(_dice_from_thresholds(fixed * eval_mask, warped * eval_mask))

    # Metric 2: Normalized Mutual Information
    nmi = _mutual_information(fixed_eval, warped_eval, bins=bins)

    # Metric 3: Normalized Cross-Correlation -> map to 0..1
    ncc_val = _ncc(fixed_eval, warped_eval)
    ncc = float(np.clip((ncc_val + 1.0) / 2.0, 0.0, 1.0))

    # Metric 4: Gradient correlation -> map to 0..1
    grad_r = _gradient_correlation(fixed * eval_mask, warped * eval_mask)
    grad_corr = float(np.clip((grad_r + 1.0) / 2.0, 0.0, 1.0))

    # Metric 5: Histogram intersection
    hist_inter = _histogram_intersection(fixed_eval, warped_eval, bins=bins)

    # Aggregate (equal weights)
    metrics = {
        'dice': dice,
        'nmi': nmi,
        'ncc': ncc,
        'gradient_correlation': grad_corr,
        'histogram_intersection': hist_inter,
    }

    # include optional additional metrics
    extra = {}
    if deformation_field_path is not None:
        try:
            def_nii = nib.load(deformation_field_path)
            def_data = def_nii.get_fdata()
            jac_stats = _compute_jacobian_stats(def_data)
            # map to 0..1: high score if folding fraction low and mean det ~1
            folding = jac_stats.get('folding_fraction', 1.0)
            mean_det = jac_stats.get('mean', 0.0)
            jac_score = float(np.clip(np.exp(-abs(mean_det - 1.0)) * (1.0 - folding), 0.0, 1.0))
            extra['jacobian'] = jac_score
            metrics['jacobian'] = jac_score
        except Exception:
            extra['jacobian'] = None

    if landmarks_path is not None:
        try:
            lp = np.load(landmarks_path)
            fixed_lm = lp['fixed']
            moving_lm = lp['moving']
            tre_val = _compute_tre(fixed_lm, moving_lm)
            # map tre to score (smaller better), assume mm units, sigma=5mm
            tre_score = float(np.clip(np.exp(-tre_val / 5.0), 0.0, 1.0))
            extra['tre'] = {'mm': float(tre_val), 'score': tre_score}
            metrics['tre'] = tre_score
        except Exception:
            extra['tre'] = None

    # If custom weights supplied, compute weighted average
    if weights:
        # normalize weights to sum 1 across present metrics
        keys = [k for k in metrics.keys() if metrics[k] is not None]
        wsum = sum(weights.get(k, 0.0) for k in keys)
        if wsum <= 0:
            final_score = float(np.mean(list(metrics.values())))
        else:
            final_score = float(sum(metrics[k] * weights.get(k, 0.0) for k in keys) / wsum)
    else:
        final_score = float(np.mean(list(metrics.values())))

    return {
        'metrics': metrics,
        'final_score': final_score,
        'extra': extra,
    }


def _compute_jacobian_stats(def_field: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics of Jacobian determinant from a displacement field.

    def_field expected shape: (..., ndim) where last dim are vector components.
    Returns mean, std, and folding fraction (fraction of voxels with det<=0).
    """
    if def_field.ndim < 4:
        raise ValueError("deformation field must have last dimension equal to vector components")
    # def_field: shape (Z,Y,X,3) or similar
    comps = [def_field[..., i] for i in range(def_field.shape[-1])]
    grads = []
    for comp in comps:
        g = np.stack(np.gradient(comp), axis=0)  # shape (ndim,...)
        grads.append(g)

    # Build Jacobian: I + grad(u)
    # For each voxel, create a matrix I + du/dx
    dims = def_field.shape[-1]
    shape = def_field.shape[:-1]
    dets = np.zeros(shape)
    it = np.nditer(dets, flags=['multi_index'], op_flags=['writeonly'])
    for idx, _ in np.ndenumerate(dets):
        J = np.eye(dims)
        for i in range(dims):
            for j in range(dims):
                # derivative of component i w.r.t axis j at voxel idx
                J[i, j] += grads[i][j][idx]
        dets[idx] = np.linalg.det(J)

    mean = float(np.mean(dets))
    std = float(np.std(dets))
    folding_fraction = float(np.mean(dets <= 0.0))
    return {'mean': mean, 'std': std, 'folding_fraction': folding_fraction}


def _compute_tre(fixed_lm: np.ndarray, moving_lm: np.ndarray) -> float:
    """Compute Target Registration Error (mean Euclidean distance) between paired landmarks.

    `fixed_lm` and `moving_lm` are expected shape (N,3).
    Returns mean distance in same units as the coordinates.
    """
    fixed_lm = np.asarray(fixed_lm)
    moving_lm = np.asarray(moving_lm)
    if fixed_lm.shape != moving_lm.shape:
        raise ValueError("Landmark arrays must have the same shape")
    if fixed_lm.ndim != 2 or fixed_lm.shape[1] != 3:
        raise ValueError("Landmarks must be shape (N,3)")
    dists = np.linalg.norm(fixed_lm - moving_lm, axis=1)
    return float(np.mean(dists))
