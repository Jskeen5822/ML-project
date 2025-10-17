"""Command-line interface for the ML project."""
import click
from pathlib import Path
from . import data as data_mod
from . import model as model_mod
from . import pipeline as pipeline_mod


@click.group()
def cli():
    """ML project CLI"""
    pass


@cli.command()
@click.option("--output", default="data/sample.csv", help="Output CSV path")
@click.option("--rows", default=500, help="Number of rows to generate")
def generate_data(output, rows):
    """Generate synthetic dataset"""
    path = data_mod.generate_classification_csv(output=output, n_samples=rows)
    click.echo(f"Wrote dataset to {path}")


@cli.command()
@click.option("--data", default="data/sample.csv", help="Input CSV path")
@click.option("--model", default="models/model.joblib", help="Output model path")
def train(data, model):
    """Train model from CSV"""
    res = model_mod.train_model(data, model)
    click.echo(f"Trained model saved to {res['model_path']}; test_score={res['test_score']:.4f}")


@cli.command()
@click.option("--model", default="models/model.joblib", help="Model path")
@click.option("--input", "input_csv", default="data/sample.csv", help="Input CSV to predict on")
@click.option("--top", "top_k", default=3, help="Top K predictions")
def predict(model, input_csv, top_k):
    """Predict using saved model"""
    m = model_mod.load_model(model)
    import pandas as pd
    df = pd.read_csv(input_csv)
    X = df.drop(columns=["label"]) if "label" in df.columns else df
    preds = model_mod.predict(m, X, top_k=top_k)
    for i, row in enumerate(preds[:10]):
        click.echo(f"Row {i}: {row}")


@cli.command(name="run-pipeline")
@click.option("--model", default="models/model.joblib", help="Model path")
@click.option("--input", "input_csv", default="data/sample.csv", help="Input CSV to run pipeline against")
@click.option("--save-actions", default=None, help="CSV path to save mapped actions")
def run_pipeline(model, input_csv, save_actions):
    m = model_mod.load_model(model)
    mapped = pipeline_mod.run_pipeline(m, input_csv, save_actions=save_actions)
    click.echo(f"Pipeline result: {mapped if isinstance(mapped, list) and len(mapped)<=5 else str(mapped)}")


if __name__ == "__main__":
    cli()