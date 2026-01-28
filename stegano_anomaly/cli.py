from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import typer

from .data import build_records
from .features import FeatureConfig, extract_features
from .models import build_model, anomaly_score, available_models
from .eval import compute_metrics
from .utils import ensure_dir, sha256_file

app = typer.Typer(add_completion=False, no_args_is_help=True)

def _save_df(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

@app.command("models")
def list_models():
    """List available baselines."""
    for m in available_models():
        typer.echo(f"{m.name}\t({m.kind})")


@app.command("extract")
def extract_cmd(
    images_dir: Path = typer.Argument(..., help="Directory containing images (recursively scanned)."),
    out_features: Path = typer.Option(Path("artifacts/features.csv"), "--out", help="Output features CSV/Parquet."),
    labels_csv: Optional[Path] = typer.Option(None, help="Optional CSV with headers filename,label."),
    label_from_parent: bool = typer.Option(False, help="Infer labels from parent folder name (clean/stego)."),
    resize: int = typer.Option(256, help="Resize images to NxN for stable features (0 disables)."),
    color: bool = typer.Option(False, help="Use color instead of grayscale."),
    n_limit: Optional[int] = typer.Option(None, help="Optional limit on number of images (for quick experiments)."),
):
    """Extract engineered features from images."""
    cfg = FeatureConfig(
        resize=None if resize == 0 else (resize, resize),
        grayscale=not color,
    )
    recs = build_records(images_dir, labels_csv=labels_csv, label_from_parent=label_from_parent)
    if n_limit is not None:
        recs = recs[: int(n_limit)]

    rows: List[Dict[str, Any]] = []
    for r in recs:
        f = extract_features(r.path, cfg)
        row = {"path": str(r.path), "sha256": sha256_file(r.path)}
        if r.label is not None:
            row["label"] = int(r.label)
        row.update(f)
        rows.append(row)

    df = pd.DataFrame(rows)
    _save_df(df, out_features)
    typer.echo(f"Wrote {len(df)} rows to {out_features}")


@app.command("train")
def train_cmd(
    features_path: Path = typer.Argument(..., help="Features CSV/Parquet produced by extract."),
    model_name: str = typer.Option("iforest", "--model", help="Model baseline name. See `models`."),
    out_model: Path = typer.Option(Path("artifacts/model.joblib"), "--out", help="Output model file."),
    supervised: Optional[bool] = typer.Option(None, help="Force supervised/unsupervised; default inferred from labels."),
    test_size: float = typer.Option(0.25, help="Holdout proportion for evaluation report."),
    random_state: int = typer.Option(42, help="RNG seed."),
    out_report: Path = typer.Option(Path("artifacts/report.json"), help="Write metrics report JSON (if labels exist)."),
    out_plots_dir: Path = typer.Option(Path("artifacts/plots"), help="Write ROC/PR plots (if labels exist)."),
):
    """Train a baseline model; optionally evaluate if labels exist."""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    df = _load_df(features_path)
    has_labels = "label" in df.columns and df["label"].notna().all()

    if supervised is None:
        supervised = bool(has_labels and model_name.lower() in {"logreg", "rf"})

    if supervised and not has_labels:
        raise typer.BadParameter("Supervised training requires `label` column for all rows.")

    feature_cols = [c for c in df.columns if c not in {"path", "sha256", "label"}]
    X = df[feature_cols].to_numpy(dtype=float)

    if has_labels:
        y = df["label"].to_numpy(dtype=int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        y_tr = None
        X_tr, X_te = X, None

    model = build_model(model_name, supervised=supervised, random_state=random_state)
    if supervised:
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols, "model_name": model_name, "supervised": supervised}, out_model)
    typer.echo(f"Saved model to {out_model}")

    if has_labels and X_te is not None:
        scores = anomaly_score(model, X_te)
        metrics = compute_metrics(y_te, scores)

        ensure_dir(out_plots_dir)

        # ROC curve
        fpr, tpr, _thr = metrics.roc_curve
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC (AUC={metrics.roc_auc:.4f})")
        roc_path = out_plots_dir / "roc.png"
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close()

        # PR curve
        prec, rec, _thr2 = metrics.pr_curve
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall (AP={metrics.average_precision:.4f})")
        pr_path = out_plots_dir / "pr.png"
        plt.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close()

        report = {
            "model": model_name,
            "supervised": supervised,
            "n_total": int(len(df)),
            "n_test": int(len(y_te)),
            "roc_auc": metrics.roc_auc,
            "average_precision": metrics.average_precision,
            "feature_count": int(len(feature_cols)),
        }
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"Wrote report to {out_report}")
        typer.echo(f"Wrote plots to {out_plots_dir} (roc.png, pr.png)")


@app.command("predict")
def predict_cmd(
    model_path: Path = typer.Argument(..., help="Model produced by train."),
    features_path: Path = typer.Argument(..., help="Features CSV/Parquet."),
    out_scores: Path = typer.Option(Path("artifacts/scores.csv"), "--out", help="Output CSV with anomaly scores."),
):
    """Score feature rows; higher score => more anomalous."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = _load_df(features_path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"Missing feature columns in features file: {missing[:10]}{'...' if len(missing)>10 else ''}")

    X = df[feature_cols].to_numpy(dtype=float)
    scores = anomaly_score(model, X)

    out = df[["path", "sha256"]].copy()
    if "label" in df.columns:
        out["label"] = df["label"]
    out["anomaly_score"] = scores

    _save_df(out, out_scores)
    typer.echo(f"Wrote {len(out)} scores to {out_scores}")


def main():
    app()

if __name__ == "__main__":
    main()
