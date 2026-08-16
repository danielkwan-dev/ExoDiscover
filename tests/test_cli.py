import json
import shutil

from typer.testing import CliRunner

from exodiscover.cli import app

runner = CliRunner()

REQUIRED_KEYS = (
    "model",
    "features",
    "test",
    "cv",
    "ladder",
    "ablation",
    "transfer",
    "importance",
    "reliability",
)


def test_train_writes_the_full_artifact_contract(tmp_path, monkeypatch):
    monkeypatch.setattr("exodiscover.cli.settings.root", tmp_path)
    monkeypatch.setattr("exodiscover.evaluate.settings.root", tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy("tests/fixtures/koi_sample.csv", raw / "koi.csv")
    shutil.copy("tests/fixtures/toi_sample.csv", raw / "toi.csv")

    result = runner.invoke(app, ["train", "--fast"])
    assert result.exit_code == 0, result.output

    payload = json.loads((tmp_path / "docs" / "metrics" / "metrics.json").read_text())
    for key in REQUIRED_KEYS:
        assert key in payload, f"metrics.json missing required key {key}"

    assert (tmp_path / "models" / "production" / "model.joblib").exists()
    assert (tmp_path / "docs" / "metrics" / "top_candidates.csv").exists()
    for figure in ("pr_curve.png", "reliability.png", "confusion_matrix.png"):
        assert (tmp_path / "docs" / "metrics" / figure).exists()


def test_train_records_star_counts_not_just_row_counts(tmp_path, monkeypatch):
    monkeypatch.setattr("exodiscover.cli.settings.root", tmp_path)
    monkeypatch.setattr("exodiscover.evaluate.settings.root", tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy("tests/fixtures/koi_sample.csv", raw / "koi.csv")

    assert runner.invoke(app, ["train", "--fast"]).exit_code == 0
    model_block = json.loads((tmp_path / "docs" / "metrics" / "metrics.json").read_text())["model"]
    assert model_block["n_train_stars"] <= model_block["n_train_rows"]
    assert model_block["framing"] == "binary"


def test_predict_command_is_registered():
    assert runner.invoke(app, ["predict", "--help"]).exit_code == 0


def test_help_lists_every_command():
    output = runner.invoke(app, ["--help"]).output
    for command in ("ingest", "train", "eval", "predict"):
        assert command in output
