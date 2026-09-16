import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tfm.data import load_matches, split_by_date, walk_forward, recent_training_rows, shin_probabilities
from tfm.experiment import fit_shrinkage, fit_temperature, run, score, temperature_scale
from tfm.models import ModelConfig, TabularModel, probabilities
from selective import BASE_FEATURE_COLS, MARKET_COLS


@pytest.fixture
def raw(tmp_path):
    rows = []
    for year in range(2020, 2023):
        for i, day in enumerate(pd.date_range(f"{year}-08-01", periods=60)):
            h, a = [(2, 0), (1, 1), (0, 2)][i % 3]
            rows.append(dict(Date=str(day.date()), Season=str(year), Div="E0",
                HomeTeam=f"T{i % 6}", AwayTeam=f"T{(i + 2) % 6}", FTHG=h, FTAG=a,
                FTR=["H", "D", "A"][i % 3], AvgH=2.1, AvgD=3.4, AvgA=3.2,
                MaxH=2.2, MaxD=3.6, MaxA=3.4, AvgCH=2.0, AvgCD=3.5, AvgCA=3.4,
                MaxCH=2.1, MaxCD=3.7, MaxCA=3.6))
    path = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_future_and_current_results_do_not_leak(raw):
    before = load_matches(raw)
    data = pd.read_csv(raw)
    data.loc[120:, ["FTHG", "FTAG", "FTR"]] = [9, 0, "H"]
    data.to_csv(raw, index=False)
    after = load_matches(raw)
    pd.testing.assert_frame_equal(before.loc[:120, BASE_FEATURE_COLS + MARKET_COLS],
                                  after.loc[:120, BASE_FEATURE_COLS + MARKET_COLS])
    # Results only become available to subsequent fixtures.
    assert not before.loc[121:, BASE_FEATURE_COLS].equals(after.loc[121:, BASE_FEATURE_COLS])


def test_closing_odds_are_not_used_in_prematch_features(raw):
    before = load_matches(raw)
    data = pd.read_csv(raw)
    data[["AvgCH", "AvgCD", "AvgCA", "MaxCH", "MaxCD", "MaxCA"]] = 8.0
    data.to_csv(raw, index=False)
    after = load_matches(raw)
    pd.testing.assert_frame_equal(before[MARKET_COLS + ["MaxH"]], after[MARKET_COLS + ["MaxH"]])
    closing = load_matches(raw, "closing")
    np.testing.assert_allclose(closing[MARKET_COLS], 1 / 3)
    assert (closing.MaxH == 8).all()


def test_date_splits_and_training_cap(raw):
    frame = load_matches(raw)
    [(season, train, val, test)] = list(walk_forward(frame, min_season_matches=20))
    fit, stop = split_by_date(train)
    cal, shrink = split_by_date(val, .5)
    blocks = [fit, stop, cal, shrink, test]
    for a, b in zip(blocks, blocks[1:]):
        assert a.Date.max() < b.Date.min()
        assert set(a.match_key).isdisjoint(b.match_key)
    capped = recent_training_rows(fit, 20)
    assert len(capped) == 20
    assert capped.Date.max() == fit.Date.max()
    with pytest.raises(ValueError, match="Unknown test seasons"):
        list(walk_forward(frame, test_seasons=["missing"]))


def test_duplicate_and_overlapping_seasons_fail(raw):
    data = pd.read_csv(raw)
    pd.concat([data, data.iloc[:1]]).to_csv(raw, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_matches(raw)
    data.to_csv(raw, index=False)
    frame = load_matches(raw)
    frame.loc[0, "Season"] = "2021"
    with pytest.raises(ValueError, match="overlap"):
        list(walk_forward(frame, 20))


def test_class_alignment_and_scoring():
    reordered = probabilities([[.2, .5, .3]], classes=[2, 0, 1])
    np.testing.assert_allclose(reordered, [[.5, .3, .2]])
    with pytest.raises(ValueError):
        probabilities([[np.nan, .2, .8]])
    with pytest.raises(ValueError):
        probabilities([[.2, .8]], [0, 1])
    perfect = score([0, 1, 2], np.eye(3))
    assert perfect["brier"] < 1e-20
    uniform = score([0, 1, 2], np.full((3, 3), 1 / 3))
    assert uniform["log_loss"] == pytest.approx(np.log(3))
    assert uniform["brier"] == pytest.approx(2 / 3)


def test_shin_reference_and_probability_limits():
    # Published example: https://github.com/mberk/shin#usage
    expected = [0.37299406033208965, 0.4047794109200184, 0.2222265287474275]
    np.testing.assert_allclose(shin_probabilities([[2.6, 2.4, 4.3]])[0], expected, atol=1e-10)
    p = shin_probabilities([[3, 3, 3], [4, 4, 4], [1.01, 100, 100], [1.01, 1.02, 1.03]])
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_allclose(p[:2], 1 / 3)
    assert np.isfinite(p).all() and (p > 0).all()


def test_calibration_shrinkage_and_missing_validation_classes():
    p = np.tile([.05, .90, .05], (20, 1))
    y = np.zeros(20, dtype=int)
    market = np.tile([.90, .05, .05], (20, 1))
    t = fit_temperature(p, y)
    assert score(y, temperature_scale(p, t))["log_loss"] <= score(y, p)["log_loss"]
    assert fit_shrinkage(market, p, y) == 0


@pytest.mark.parametrize("name", ["lightgbm", "tabnet", "ft_transformer"])
def test_real_model_fit_predict_and_train_only_imputation(name):
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(48, 3)), columns=list("abc"))
    X["missing"] = np.nan
    y = np.tile([0, 1, 2], 16)
    stop = X.iloc[:9].copy()
    stop["a"] = 10000
    m = TabularModel(name, ModelConfig(epochs=2, patience=1, batch_size=16)).fit(X, y, stop, y[:9])
    np.testing.assert_allclose(m.scaler_.mean_[0], X.a.mean())
    p = m.predict_proba(stop)
    assert p.shape == (9, 3)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert len(m.feature_names_) == 4
    with pytest.raises(ValueError, match="schema"):
        m.predict_proba(stop[list(reversed(stop.columns))])


def test_tabpfn_adapter_without_download(monkeypatch):
    import tfm.models as module
    import types
    captured = {}
    class FakePFN:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            captured.update(version=version, **kwargs)
            return cls()
        def fit(self, X, y):
            captured["context_rows"] = len(y)
            self.classes_ = np.array([2, 0, 1])
        def predict_proba(self, X):
            return np.tile([.2, .5, .3], (len(X), 1))
    original = module.dependency
    def fake(module_name, package):
        if module_name == "tabpfn":
            return types.SimpleNamespace(TabPFNClassifier=FakePFN)
        if module_name == "tabpfn.constants":
            return types.SimpleNamespace(ModelVersion=types.SimpleNamespace(V2="v2", V2_5="v2.5"))
        return original(module_name, package)
    monkeypatch.setattr(module, "dependency", fake)
    X = pd.DataFrame({"a": np.arange(12)})
    m = TabularModel("tabpfn", ModelConfig(batch_size=4)).fit(X, np.tile([0, 1, 2], 4), X[:3], [0, 1, 2])
    np.testing.assert_allclose(m.predict_proba(X), np.tile([.5, .3, .2], (12, 1)))
    assert captured["context_rows"] == 12
    assert captured["version"] == "v2"


def test_end_to_end_artifacts(raw, tmp_path):
    output = tmp_path / "run"
    metrics = run(raw, output, models=["lightgbm"], min_season_matches=20,
                  max_train_rows=30, n_boot=10, thresholds=[.02])
    assert set(metrics.arm) == {"raw", "calibrated", "corrected", "market"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    splits = pd.read_csv(output / "split_assignments.csv")
    assert not splits.duplicated(["fold", "match_key"]).any()
    assert len(list(output.glob("predictions_*.csv"))) == 1
    with pytest.raises(ValueError, match="not empty"):
        run(raw, output)


@pytest.mark.skipif(os.environ.get("RUN_TABPFN_INTEGRATION") != "1", reason="Explicit checkpoint/network integration test")
def test_real_tabpfn():
    import torch
    torch.set_num_threads(1)
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(30, 4)))
    model = TabularModel("tabpfn", ModelConfig(tabpfn_estimators=1, batch_size=16))
    model.fit(X, np.tile([0, 1, 2], 10), X[:6], [0, 1, 2, 0, 1, 2])
    p = model.predict_proba(X[:4])
    assert p.shape == (4, 3)
    np.testing.assert_allclose(p.sum(1), 1)
