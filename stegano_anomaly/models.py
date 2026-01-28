from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

# Note: LOF does not support predict_proba and behaves differently at inference time.
# We handle it as a score-only method (fit on train, then score_samples on eval/infer).

@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "unsupervised" or "supervised"

def available_models() -> List[ModelSpec]:
    return [
        ModelSpec("iforest", "unsupervised"),
        ModelSpec("ocsvm", "unsupervised"),
        ModelSpec("lof", "unsupervised"),
        ModelSpec("logreg", "supervised"),
        ModelSpec("rf", "supervised"),
    ]

def build_model(name: str, supervised: bool, random_state: int = 42) -> Any:
    name = name.lower()
    if supervised:
        if name == "logreg":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)),
            ])
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=500,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
        raise ValueError(f"Unknown supervised model: {name}")
    else:
        if name == "iforest":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", IsolationForest(
                    n_estimators=500,
                    contamination="auto",
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ])
        if name == "ocsvm":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)),
            ])
        if name == "lof":
            # LOF uses its own scaling assumptions; we still standardize.
            # We set novelty=True to allow scoring on unseen samples.
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LocalOutlierFactor(n_neighbors=35, novelty=True)),
            ])
        raise ValueError(f"Unknown unsupervised model: {name}")

def anomaly_score(model: Any, X: np.ndarray) -> np.ndarray:
    """Return higher score => more anomalous."""
    # Try common APIs
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # For many anomaly detectors, lower decision_function => more anomalous.
        # We invert to standardize: higher => more anomalous.
        return -np.asarray(s, dtype=float)
    if hasattr(model, "score_samples"):
        s = model.score_samples(X)
        return -np.asarray(s, dtype=float)
    # Supervised classifiers: use predicted probability for class 1.
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return np.asarray(p[:, 1], dtype=float)
    # Fallback: use predict output
    y = model.predict(X)
    return np.asarray(y, dtype=float)
