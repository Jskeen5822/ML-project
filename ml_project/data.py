"""Data generation utilities for the ML project."""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from pathlib import Path


def generate_classification_csv(output: str, n_samples: int = 500, n_features: int = 6, n_classes: int = 3, random_state: int = 42):
    """Generate a synthetic multiclass classification dataset and write to CSV.

    Output CSV has columns: feature_0..feature_{n_features-1}, label
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=min(4, n_features),
                               n_redundant=0, n_classes=n_classes, random_state=random_state)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["label"] = y

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)
