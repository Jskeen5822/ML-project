ML-project

A small starter machine learning project demonstrating data generation, training, prediction, and basic automation mapping predictions to actions.

Overview

This repository contains a minimal end-to-end example of a supervised ML workflow and a simple automation layer that maps model predictions to actions. Key parts:

- `ml_project/data.py` — synthetic dataset generator (CSV output)
- `ml_project/model.py` — training, save/load, and prediction helpers using scikit-learn
- `ml_project/pipeline.py` — map predictions to actions and optionally save them
- `ml_project/cli.py` — a small command-line interface for common tasks
- `tests/` — pytest tests that validate training and pipeline mapping

Setup

1. Create and activate a Python virtual environment (PowerShell example):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

If you prefer pre-built wheels (faster), use the provided `requirements-wheels.txt` with this command:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install --only-binary :all: -r requirements-wheels.txt
```

Quick examples

Generate a dataset, train a model, and run predictions:

```powershell
python -m ml_project.cli generate-data --output data/sample.csv --rows 500
python -m ml_project.cli train --data data/sample.csv --model models/model.joblib
python -m ml_project.cli predict --model models/model.joblib --input data/sample.csv --top 5
python -m ml_project.cli run-pipeline --model models/model.joblib --input data/sample.csv --save-actions outputs/actions.csv
```

Run tests

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
```

Notes & next steps

- The classifier used is a RandomForest inside a scikit-learn Pipeline with a StandardScaler.
- The pipeline maps top-1 predicted label to a simple action mapping; extend `ml_project/pipeline.py` to implement real automation (webhooks, system calls, etc.).
- Consider adding GitHub Actions CI to run the tests on push (I can add a workflow file if you want).

License

MIT
A basic machine learning algorithm for automation and problem solving
