from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: Optional[int] = None  # 0=clean, 1=stego/anomaly

def iter_images(root: Path) -> Iterable[Path]:
    root = root.expanduser().resolve()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p

def load_label_map_csv(path: Path) -> Dict[str, int]:
    """CSV: filename,label where filename may be basename or relative path."""
    m: Dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if "filename" not in row or "label" not in row:
                raise ValueError("Label CSV must have headers: filename,label")
            m[row["filename"]] = int(row["label"])
    return m

def build_records(
    images_dir: Path,
    labels_csv: Optional[Path] = None,
    label_from_parent: bool = False,
) -> List[ImageRecord]:
    """Build ImageRecord list.

    If labels_csv provided, it is used (by basename first, then relative path).
    If label_from_parent, parent folder name 'clean'/'cover' => 0 and 'stego'/'anomaly' => 1.
    """
    images_dir = images_dir.expanduser().resolve()
    label_map = load_label_map_csv(labels_csv) if labels_csv else {}
    records: List[ImageRecord] = []

    for p in iter_images(images_dir):
        lab: Optional[int] = None
        if labels_csv:
            key1 = p.name
            key2 = str(p.relative_to(images_dir))
            if key1 in label_map:
                lab = label_map[key1]
            elif key2 in label_map:
                lab = label_map[key2]
        if lab is None and label_from_parent:
            parent = p.parent.name.lower()
            if parent in {"clean", "cover", "normal", "benign"}:
                lab = 0
            elif parent in {"stego", "anomaly", "abnormal", "malicious"}:
                lab = 1
        records.append(ImageRecord(path=p, label=lab))
    return records
