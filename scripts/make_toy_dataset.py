from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def random_cover(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    # Create a smooth-ish image by filtering random noise and adding gradients.
    base = rng.normal(127, 40, size=(size, size)).clip(0, 255)
    gx = np.linspace(0, 30, size)[None, :]
    gy = np.linspace(0, 30, size)[:, None]
    img = (base + gx + gy).clip(0, 255).astype(np.uint8)
    return img

def lsb_embed(rng: np.random.Generator, img: np.ndarray, rate: float = 0.15) -> np.ndarray:
    """Synthetic LSB embedding for demo data only.
    Flips LSB of a fraction of pixels to simulate stego noise.
    """
    x = img.copy()
    mask = rng.random(x.shape) < rate
    x[mask] ^= 1  # flip least significant bit
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/toy"), help="Output root directory.")
    ap.add_argument("--n", type=int, default=600, help="Total images.")
    ap.add_argument("--size", type=int, default=256, help="Image size (NxN).")
    ap.add_argument("--stego_rate", type=float, default=0.5, help="Fraction labeled as stego/anomaly.")
    ap.add_argument("--embed_rate", type=float, default=0.15, help="LSB flip rate for stego samples.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed.")
    args = ap.parse_args()

    out_clean = args.out / "clean"
    out_stego = args.out / "stego"
    out_clean.mkdir(parents=True, exist_ok=True)
    out_stego.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_stego = int(round(args.n * args.stego_rate))
    n_clean = args.n - n_stego

    for i in tqdm(range(n_clean), desc="clean"):
        img = random_cover(rng, size=args.size)
        Image.fromarray(img, mode="L").save(out_clean / f"clean_{i:05d}.png")

    for i in tqdm(range(n_stego), desc="stego"):
        img = random_cover(rng, size=args.size)
        img2 = lsb_embed(rng, img, rate=args.embed_rate)
        Image.fromarray(img2, mode="L").save(out_stego / f"stego_{i:05d}.png")

    print(f"Wrote dataset to: {args.out}")
    print("Folder labels: clean=0, stego=1 (usable with --label-from-parent)")

if __name__ == "__main__":
    main()
