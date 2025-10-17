from ml_project import data as data_mod, model as model_mod, pipeline as pipeline_mod


def test_pipeline_maps_actions(tmp_path):
    csv = tmp_path / "sample.csv"
    model_file = tmp_path / "model.joblib"

    data_mod.generate_classification_csv(str(csv), n_samples=100, n_features=4, n_classes=3)
    model_mod.train_model(str(csv), str(model_file))
    m = model_mod.load_model(str(model_file))
    mapped = pipeline_mod.run_pipeline(m, str(csv))
    assert isinstance(mapped, list)
    assert "action" in mapped[0]
