from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
from PIL import Image
from scipy import fftpack
from scipy.ndimage import convolve

# Lightweight, engineered features intended for classical baselines.
# They are not meant to be state-of-the-art steganalysis features (e.g., SRM),
# but they are reproducible and provide reasonable signal for anomaly baselines.

@dataclass(frozen=True)
class FeatureConfig:
    resize: Optional[Tuple[int, int]] = (256, 256)
    grayscale: bool = True
    hist_bins: int = 64
    diff_hist_bins: int = 64
    dct_hist_bins: int = 64
    dct_block: int = 8

def _load_image(path: Path, cfg: FeatureConfig) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if cfg.resize is not None:
        img = img.resize(cfg.resize, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if cfg.grayscale:
        # ITU-R BT.601 luma approximation
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return arr

def _moments(x: np.ndarray) -> Tuple[float, float, float, float]:
    x = x.astype(np.float64).ravel()
    mu = float(np.mean(x))
    var = float(np.var(x))
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0.0:
        return mu, var, 0.0, 0.0
    z = (x - mu) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0
    return mu, var, skew, kurt

import math

def _histogram(x: np.ndarray, bins: int, rng: Tuple[float, float]) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, range=rng, density=True)
    return h.astype(np.float32)

def _adjacent_diffs(img: np.ndarray) -> np.ndarray:
    # Horizontal and vertical differences
    dh = img[:, 1:] - img[:, :-1]
    dv = img[1:, :] - img[:-1, :]
    return np.concatenate([dh.ravel(), dv.ravel()]).astype(np.float32)

def _highpass_residual(img: np.ndarray) -> np.ndarray:
    # Simple high-pass filter (Laplacian-like)
    k = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]], dtype=np.float32)
    res = convolve(img.astype(np.float32), k, mode="reflect")
    return res

def _blockwise_dct_hist(img: np.ndarray, block: int, bins: int) -> np.ndarray:
    H, W = img.shape
    Hc = (H // block) * block
    Wc = (W // block) * block
    x = img[:Hc, :Wc].astype(np.float32)

    mags: List[np.ndarray] = []
    for i in range(0, Hc, block):
        for j in range(0, Wc, block):
            b = x[i:i+block, j:j+block]
            # 2D DCT-II
            d = fftpack.dct(fftpack.dct(b.T, norm="ortho").T, norm="ortho")
            # Ignore DC; take magnitudes of AC coefficients
            ac = d.copy()
            ac[0, 0] = 0.0
            mags.append(np.abs(ac).ravel())
    mags_all = np.concatenate(mags) if mags else np.array([], dtype=np.float32)
    if mags_all.size == 0:
        return np.zeros((bins,), dtype=np.float32)

    # Use log scale to reduce heavy tails
    v = np.log1p(mags_all)
    vmax = float(np.quantile(v, 0.995)) if v.size > 100 else float(np.max(v) + 1e-6)
    vmax = max(vmax, 1e-6)
    return _histogram(v, bins=bins, rng=(0.0, vmax))

def extract_features(path: Path, cfg: FeatureConfig = FeatureConfig()) -> Dict[str, float]:
    img = _load_image(path, cfg)

    feats: Dict[str, float] = {}
    # Pixel intensity moments
    mu, var, skew, kurt = _moments(img)
    feats.update({
        "pix_mean": mu,
        "pix_var": var,
        "pix_skew": skew,
        "pix_kurt": kurt,
    })

    # Intensity histogram
    h = _histogram(img, bins=cfg.hist_bins, rng=(0.0, 255.0))
    for i, v in enumerate(h):
        feats[f"hist_{i:03d}"] = float(v)

    # Adjacent difference statistics + histogram (LSB embedding often perturbs local differences)
    diffs = _adjacent_diffs(img)
    dmu, dvar, dskew, dkurt = _moments(diffs)
    feats.update({
        "diff_mean": dmu,
        "diff_var": dvar,
        "diff_skew": dskew,
        "diff_kurt": dkurt,
    })
    # Differences roughly in [-255, 255]
    dh = _histogram(diffs, bins=cfg.diff_hist_bins, rng=(-255.0, 255.0))
    for i, v in enumerate(dh):
        feats[f"diffhist_{i:03d}"] = float(v)

    # High-pass residual moments (captures noise/residual artifacts)
    res = _highpass_residual(img)
    rmu, rvar, rskew, rkurt = _moments(res)
    feats.update({
        "hp_mean": rmu,
        "hp_var": rvar,
        "hp_skew": rskew,
        "hp_kurt": rkurt,
    })

    # Block DCT magnitude histogram (frequency-domain artifacts)
    dcth = _blockwise_dct_hist(img, block=cfg.dct_block, bins=cfg.dct_hist_bins)
    for i, v in enumerate(dcth):
        feats[f"dcthist_{i:03d}"] = float(v)

    return feats
