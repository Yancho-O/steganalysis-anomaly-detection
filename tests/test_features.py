from pathlib import Path
import numpy as np
from PIL import Image

from stegano_anomaly.features import extract_features, FeatureConfig

def test_extract_features_runs(tmp_path: Path):
    img = (np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8))
    p = tmp_path / "x.png"
    Image.fromarray(img, mode="L").save(p)

    cfg = FeatureConfig(resize=(64, 64), grayscale=True, hist_bins=16, diff_hist_bins=16, dct_hist_bins=16)
    feats = extract_features(p, cfg)
    assert "pix_mean" in feats
    assert any(k.startswith("hist_") for k in feats.keys())
    assert any(k.startswith("dcthist_") for k in feats.keys())
