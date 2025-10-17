"""Model training, saving and prediction helpers."""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(data_csv: str, model_out: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(data_csv)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column")
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])
    pipe.fit(X_train, y_train)

    out = Path(model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)

    # Return some basic metrics
    score = pipe.score(X_test, y_test)
    return {"model_path": str(out), "test_score": float(score)}


def load_model(model_path: str):
    return joblib.load(model_path)


def predict(model, X_df, top_k: int = 3):
    """Return predictions with probabilities.

    Returns a list of dicts: {"pred": int, "prob": float}
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_df)
        preds = probs.argsort(axis=1)[:, ::-1][:, :top_k]
        top_probs = -1 * np.sort(-probs, axis=1)[:, :top_k]
        results = []
        for row_idx in range(preds.shape[0]):
            row = []
            for k in range(preds.shape[1]):
                row.append({"pred": int(preds[row_idx, k]), "prob": float(top_probs[row_idx, k])})
            results.append(row)
        return results
    else:
        preds = model.predict(X_df)
        return [[{"pred": int(p), "prob": 1.0}] for p in preds]
