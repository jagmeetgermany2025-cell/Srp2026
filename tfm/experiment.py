"""Walk-forward comparison with disjoint stopping, calibration and shrinkage."""
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from selective import (BASE_FEATURE_COLS, MARKET_COLS, apply_static_shrinkage,
                       count_candidates, select_bets)
from tfm.data import load_matches, recent_training_rows, split_by_date, walk_forward
from tfm.models import ModelConfig, TabularModel, probabilities


def temperature_scale(p, temperature):
    logits = np.log(probabilities(p)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    return probabilities(np.exp(logits))


def fit_temperature(p, y):
    result = minimize_scalar(lambda t: log_loss(y, temperature_scale(p, t), labels=[0, 1, 2]),
                             bounds=(0.25, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    # Include identity so calibration cannot worsen its own fitting objective.
    return float(min([1.0, result.x], key=lambda t: log_loss(y, temperature_scale(p, t), labels=[0, 1, 2])))


def fit_shrinkage(market, model, y):
    objective = lambda w: log_loss(y, apply_static_shrinkage(market, model, w), labels=[0, 1, 2])
    result = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    if not result.success:
        raise RuntimeError("Shrinkage optimization failed")
    return float(min([0.0, 1.0, result.x], key=objective))


def score(y, p):
    y, p = np.asarray(y, dtype=int), probabilities(p)
    one_hot = np.eye(3)[y]
    confidence, predicted = p.max(axis=1), p.argmax(axis=1)
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(np.mean(bins == b) * abs(np.mean(predicted[bins == b] == y[bins == b]) -
              confidence[bins == b].mean()) for b in range(15) if np.any(bins == b))
    return {"n": len(y), "log_loss": log_loss(y, p, labels=[0, 1, 2]),
            "brier": np.mean(np.sum((p - one_hot) ** 2, axis=1)),
            "rps": np.mean(np.sum((p.cumsum(axis=1)[:, :2] - one_hot.cumsum(axis=1)[:, :2]) ** 2, axis=1) / 2),
            "ece_15": ece, "accuracy": np.mean(predicted == y)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n")


def roi_interval(bets, seed=42, n_boot=1000):
    """Cluster bootstrap by match: keep multiple selections together."""
    if bets.empty:
        return {"roi": None, "roi_low": None, "roi_high": None}
    groups = bets.groupby("match_key")[["PnL", "Stake"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        totals = groups[rng.integers(len(groups), size=len(groups))].sum(axis=0)
        boot.append(totals[0] / totals[1])
    return {"roi": float(groups[:, 0].sum() / groups[:, 1].sum()),
            "roi_low": float(np.quantile(boot, .025)), "roi_high": float(np.quantile(boot, .975))}


def run(source, output, models=("lightgbm", "tabpfn", "tabnet", "ft_transformer"),
        seeds=(42,), market="prematch", feature_set="baseline", max_train_rows=1000,
        min_season_matches=1000, test_seasons=None, model_config=None,
        thresholds=(0.0, 0.01, 0.02, 0.05, 0.1), n_boot=1000):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new run directory")
    if not models or not seeds or not thresholds or n_boot < 1 or min_season_matches < 1:
        raise ValueError("Models, seeds, thresholds and positive sample/bootstrap counts are required")
    if feature_set not in {"baseline", "market_blind"} or not np.isfinite(thresholds).all():
        raise ValueError("Invalid feature set or thresholds")
    output.mkdir(parents=True, exist_ok=True)
    cfg = model_config or ModelConfig()
    features = list(BASE_FEATURE_COLS) if feature_set == "market_blind" else list(MARKET_COLS) + list(BASE_FEATURE_COLS)
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm", "tabpfn", "pytorch-tabnet", "rtdl-revisiting-models"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    manifest = dict(status="running", source=str(Path(source).resolve()), source_sha256=file_sha256(source),
        baseline_commit=git.stdout.strip(), python=platform.python_version(), packages=packages,
        features=features, models=list(models), seeds=list(seeds), market=market,
        max_train_rows=max_train_rows, min_season_matches=min_season_matches,
        test_seasons=test_seasons, thresholds=list(thresholds), n_boot=n_boot, model_config=asdict(cfg),
        source_code_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha256(p)
                           for p in sorted(Path(__file__).resolve().parent.glob("*.py"))})
    if cfg.checkpoint:
        manifest["checkpoint_sha256"] = file_sha256(cfg.checkpoint)
    write_json(output / "manifest.json", manifest)
    metrics, curves, diagnostics, splits = [], [], [], []
    try:
        frame = load_matches(source, market)
        manifest["data_audit"] = {**frame.attrs, "eligible_rows": len(frame),
            "season_counts": frame.groupby("Season").size().to_dict()}
        # Materialize before training so invalid requested folds fail early.
        folds = list(walk_forward(frame, min_season_matches, test_seasons))
        for season, train, validation, test in folds:
            fit, stop = split_by_date(train, .8)
            fit = recent_training_rows(fit, max_train_rows)
            calibration, shrink = split_by_date(validation, .5)
            blocks = [("fit", fit), ("early_stop", stop), ("calibration", calibration), ("shrinkage", shrink), ("test", test)]
            for (left, a), (right, b) in zip(blocks, blocks[1:]):
                if a.Date.max() >= b.Date.min():
                    raise ValueError(f"Information boundary violated: {left}, {right}")
            if set(fit.target.unique()) != {0, 1, 2}:
                raise ValueError(f"Training fold {season} does not contain all three outcomes")
            for role, block in blocks:
                splits.append(block[["match_key", "Date", "Season"]].assign(fold=season, role=role))
            for seed in seeds:
                for name in models:
                    print(f"fold={season} seed={seed} model={name} fit={len(fit)} stop={len(stop)} cal={len(calibration)} shrink={len(shrink)} test={len(test)}", flush=True)
                    started = time.perf_counter()
                    model = TabularModel(name, ModelConfig(**{**asdict(cfg), "seed": seed}))
                    model.fit(fit[features], fit.target, stop[features], stop.target)
                    temperature = fit_temperature(model.predict_proba(calibration[features]), calibration.target)
                    p_shrink = temperature_scale(model.predict_proba(shrink[features]), temperature)
                    weight = fit_shrinkage(shrink[MARKET_COLS].to_numpy(float), p_shrink, shrink.target)
                    raw = model.predict_proba(test[features])
                    calibrated = temperature_scale(raw, temperature)
                    corrected = apply_static_shrinkage(test[MARKET_COLS].to_numpy(float), calibrated, weight)
                    preds = test[["match_key", "Date", "Season", "Div", "HomeTeam", "AwayTeam", "target", "FTR", "market_timestamp"] +
                                 MARKET_COLS + [f"{b}{o}" for b in ("Max", "Avg") for o in "HDA"]].copy()
                    arms = {"raw": raw, "calibrated": calibrated, "corrected": corrected,
                            "market": test[MARKET_COLS].to_numpy(float)}
                    for arm, p in arms.items():
                        prefix = {"raw": "p_raw", "calibrated": "p_model", "corrected": "p_corr", "market": "p_ref"}[arm]
                        preds[[f"{prefix}_{o}" for o in "HDA"]] = p
                        metrics.append(dict(model=name, seed=seed, fold=season, arm=arm, **score(test.target, p)))
                    tag = f"{name}_{seed}_{season}"
                    preds.to_csv(output / f"predictions_{tag}.csv", index=False)
                    for arm, prefix in [("raw", "p_raw"), ("calibrated", "p_model"), ("corrected", "p_corr"), ("market", "p_ref")]:
                        for tau in thresholds:
                            bets = select_bets(preds, float(tau), prefix, stake_rule="flat", criterion="ev")
                            denom = count_candidates(preds)
                            curves.append(dict(model=name, seed=seed, fold=season, arm=arm, threshold=tau,
                                n_bets=len(bets), coverage=len(bets) / denom if denom else 0,
                                **roi_interval(bets, seed, n_boot)))
                    checkpoint = getattr(model.model_, "model_path", None)
                    if isinstance(checkpoint, (str, Path)) and Path(checkpoint).is_file():
                        checkpoint_digest = file_sha256(checkpoint)
                    else:
                        checkpoint_digest = None
                    diagnostics.append(dict(model=name, seed=seed, fold=season,
                        temperature=temperature, shrinkage_weight=weight, best_epoch=model.best_epoch_,
                        elapsed_seconds=time.perf_counter() - started, n_fit=len(fit), n_stop=len(stop),
                        n_calibration=len(calibration), n_shrinkage=len(shrink), n_test=len(test),
                        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_digest))
                    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
                    pd.DataFrame(curves).to_csv(output / "risk_coverage.csv", index=False)
                    pd.DataFrame(diagnostics).to_csv(output / "folds.csv", index=False)
        pd.concat(splits).to_csv(output / "split_assignments.csv", index=False)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return pd.DataFrame(metrics)
