import os
from ml_project import data as data_mod, model as model_mod


def test_train_and_predict(tmp_path):
    csv = tmp_path / "sample.csv"
    model_file = tmp_path / "model.joblib"

    data_mod.generate_classification_csv(str(csv), n_samples=200, n_features=5, n_classes=3)
    res = model_mod.train_model(str(csv), str(model_file))
    assert os.path.exists(str(model_file))
    assert "test_score" in res

    m = model_mod.load_model(str(model_file))
    import pandas as pd
    df = pd.read_csv(str(csv))
    X = df.drop(columns=["label"]) if "label" in df.columns else df
    preds = model_mod.predict(m, X, top_k=2)
    assert isinstance(preds, list)
    assert len(preds) == len(X)

