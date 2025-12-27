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

    final_score = float(np.mean(list(metrics.values())))

    return {
        'metrics': metrics,
        'final_score': final_score,
    }
