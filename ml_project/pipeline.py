"""Simple automation pipeline that maps model predictions to actions."""
from pathlib import Path
import pandas as pd
from typing import List, Dict


# Example mapping from class label to action
DEFAULT_ACTION_MAPPING = {
    0: "notify_team",
    1: "restart_service",
    2: "ignore",
}


def map_predictions_to_actions(predictions: List[List[Dict]], mapping: Dict = None):
    """Given predictions (list per-row of top-k), map the top-1 label to an action.

    Returns a list of dicts: {"pred": int, "prob": float, "action": str}
    """
    mapping = mapping or DEFAULT_ACTION_MAPPING
    out = []
    for row in predictions:
        top = row[0]
        label = top["pred"]
        action = mapping.get(label, "unknown")
        out.append({"pred": label, "prob": top["prob"], "action": action})
    return out


def run_pipeline(model, input_csv: str, mapping: Dict = None, save_actions: str = None):
    df = pd.read_csv(input_csv)
    X = df.drop(columns=["label"]) if "label" in df.columns else df

    preds = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    # Use the previous model.predict wrapper if available
    try:
        from .model import predict as _predict
        wrapped = _predict(model, X)
    except Exception:
        # fallback to top-1 predict
        p = model.predict(X)
        wrapped = [[{"pred": int(v), "prob": 1.0}] for v in p]

    mapped = map_predictions_to_actions(wrapped, mapping=mapping)

    if save_actions:
        out = Path(save_actions)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(mapped).to_csv(out, index=False)
        return str(out)

    return mapped
