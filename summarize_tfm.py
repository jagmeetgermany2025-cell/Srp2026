"""Summarize one completed, single-seed TFM experiment without double-counting matches."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from selective import select_bets
from tfm.experiment import roi_interval, score


def paired_date_bootstrap(frame, p_model, p_reference, n_boot=5000, seed=2026):
    """Paired mean log-loss difference; resample complete calendar-date clusters."""
    y = frame.target.to_numpy(int)
    idx = np.arange(len(y))
    difference = -np.log(np.clip(p_model[idx, y], 1e-12, 1)) + np.log(np.clip(p_reference[idx, y], 1e-12, 1))
    # Zero-weight shrinkage reproduces the market up to roundoff. Do not
    # present machine-precision noise as a nonzero statistical difference.
    if np.allclose(p_model, p_reference, rtol=0, atol=1e-12):
        difference[:] = 0.0
    groups = pd.DataFrame({"Date": frame.Date, "delta": difference}).groupby("Date").delta.agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        sums = groups[rng.integers(0, len(groups), size=len(groups))].sum(0)
        samples[i] = sums[0] / sums[1]
    return dict(delta_log_loss=float(difference.mean()), ci_low=float(np.quantile(samples, .025)),
                ci_high=float(np.quantile(samples, .975)), date_clusters=len(groups))


def markdown_table(df, formats=None):
    formats = formats or {}
    rows = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.to_dict("records"):
        values = [formats[c].format(row[c]) if c in formats and pd.notna(row[c]) else str(row[c]) for c in df.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only completed experiments can be summarized")
    if len(manifest["seeds"]) != 1:
        raise ValueError("This summary requires one seed; do not pool seeds as extra matches")
    seed = manifest["seeds"][0]
    data, rows = {}, []
    prefixes = {"raw": "p_raw", "calibrated": "p_model", "corrected": "p_corr"}
    for model in manifest["models"]:
        paths = sorted(directory.glob(f"predictions_{model}_{seed}_*.csv"))
        frame = pd.concat([pd.read_csv(p, dtype={"Season": str}) for p in paths], ignore_index=True)
        frame = frame.sort_values(["Date", "match_key"]).reset_index(drop=True)
        if frame.match_key.duplicated().any():
            raise ValueError("Test fixtures are duplicated across folds")
        data[model] = frame
        if len(data) > 1:
            first = next(iter(data.values()))
            pd.testing.assert_frame_equal(frame[["match_key", "target"]], first[["match_key", "target"]])
            np.testing.assert_allclose(frame[[f"p_ref_{o}" for o in "HDA"]], first[[f"p_ref_{o}" for o in "HDA"]])
        for arm, prefix in prefixes.items():
            rows.append(dict(model=model, arm=arm, **score(frame.target, frame[[f"{prefix}_{o}" for o in "HDA"]].to_numpy())))
    first = next(iter(data.values()))
    market = first[[f"p_ref_{o}" for o in "HDA"]].to_numpy()
    rows.append(dict(model="market", arm="market", **score(first.target, market)))
    pooled = pd.DataFrame(rows)
    pooled.to_csv(directory / "pooled_metrics.csv", index=False)
    intervals = []
    for model, frame in data.items():
        for arm, prefix in prefixes.items():
            pred = frame[[f"{prefix}_{o}" for o in "HDA"]].to_numpy()
            intervals.append(dict(model=model, arm=arm, reference="market", **paired_date_bootstrap(frame, pred, market)))
            if model != "lightgbm" and "lightgbm" in data:
                reference = data["lightgbm"][[f"{prefix}_{o}" for o in "HDA"]].to_numpy()
                intervals.append(dict(model=model, arm=arm, reference="lightgbm", **paired_date_bootstrap(frame, pred, reference)))
    pd.DataFrame(intervals).to_csv(directory / "paired_logloss_intervals.csv", index=False)
    bets_rows = []
    # Report one pre-specified threshold, not the best threshold on test data.
    for model, frame in list(data.items()) + [("market", first)]:
        prefix = "p_ref" if model == "market" else "p_corr"
        bets = select_bets(frame, .02, prefix, stake_rule="flat", criterion="ev")
        summary = roi_interval(bets, seed=2026, n_boot=5000)
        # Conservative repricing sensitivity, same selections settled at Avg odds.
        mean_roi = float(bets.PnL_Avg.sum() / bets.loc[bets.PnL_Avg.notna(), "Stake"].sum()) if bets.PnL_Avg.notna().any() else np.nan
        bets_rows.append(dict(model=model, threshold=.02, n_bets=len(bets), total_stake=float(bets.Stake.sum()),
            pnl=float(bets.PnL.sum()), **summary, roi_at_avg_odds=mean_roi))
    betting = pd.DataFrame(bets_rows)
    betting.to_csv(directory / "pooled_betting.csv", index=False)
    folds = pd.read_csv(directory / "folds.csv")
    table = pooled[pooled.arm.isin(["calibrated", "market"])][["model", "log_loss", "brier", "accuracy"]].copy()
    corrected = pooled[pooled.arm.isin(["corrected", "market"])].set_index("model").log_loss
    table["corrected_log_loss"] = table.model.map(corrected)
    table = table.sort_values("log_loss")
    fig_path = directory / "comparison.png"
    plot(pooled, fig_path)
    text = f"""# Tabular Model Experiment Results

Completed experiment: `{directory.name}`. Source SHA-256: `{manifest['source_sha256']}`.

- Test seasons: 2024/25 and 2025/26, comprising {len(first):,} distinct matches.
- Random seed: {seed}. All models use the same training rows, capped at {manifest['max_train_rows']}.
- Actual training rows by test season: {folds.groupby('fold').n_fit.first().to_dict()}.
- Maximum training epochs: {manifest['model_config']['epochs']}; early-stopping patience: {manifest['model_config']['patience']} epochs.
- TabPFN v2: {manifest['model_config']['tabpfn_estimators']} ensemble members, CPU, `{manifest['model_config']['tabpfn_fit_mode']}`.
- Market benchmark: Shin probabilities derived from pre-match Avg odds. Execution prices: Max odds.
- Total model runtime (sum of recorded fold durations): {folds.elapsed_seconds.sum() / 60:.2f} minutes.

## Predictive Performance

Lower log loss and Brier scores are better. `accuracy` is expressed as a proportion.
`log_loss` measures performance after calibration but before shrinkage toward the market.
`corrected_log_loss` measures performance after combining model and market probabilities
using a weight fitted on validation data. Each match is counted only once in the market reference row.

{markdown_table(table, {c: '{:.6f}' for c in table.columns if c != 'model'})}

All raw, calibrated and corrected scores are saved in `pooled_metrics.csv`.
Season-level results are saved in `metrics.csv`. Pooled expected calibration error (ECE)
was recomputed from predictions rather than calculated by averaging fold-level ECE values.

## Betting Simulation

The expected-value (EV) threshold is fixed at 2%, with home/away selections only
and a flat stake of one unit per selection. Returns in the table are proportions:
0.01 means 1%. The threshold was not selected using test results.
Model rows use probabilities after shrinkage toward the market.

{markdown_table(betting, {c: '{:.6f}' for c in ['threshold', 'pnl', 'roi', 'roi_low', 'roi_high', 'roi_at_avg_odds']})}

ROI confidence intervals use 5,000 bootstrap resamples clustered by match.
`roi_at_avg_odds` settles the same selections at average bookmaker odds. This
sensitivity check helps distinguish model performance from the benefit of taking
the best available price. Simultaneous availability of the historical maximum
prices has not been verified.

## Uncertainty and Scope

`paired_logloss_intervals.csv` contains paired log-loss differences for the same matches.
A negative difference favors the model over the specified reference. The 95% intervals
come from an exploratory bootstrap analysis with 5,000 resamples of calendar days,
retaining all matches within each sampled day. This does not fully account for
team/season dependence across days or multiple comparisons.

These results describe a controlled experiment with one random seed and a maximum
of 1,000 training rows. They should not be interpreted as results using the full
training history or multiple seeds. A corrected model score close to the market
does not, by itself, demonstrate additional predictive value; the shrinkage weights
in `folds.csv` should also be examined. The betting simulation does not establish
real-money profitability.
"""
    (directory / "RESULTS.md").write_text(text)
    print(table.to_string(index=False))
    print("\nBETTING\n", betting.to_string(index=False))
    print("\nFOLDS\n", folds.to_string(index=False))
    print("\nPAIRED\n", pd.DataFrame(intervals).to_string(index=False))


def plot(pooled, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"lightgbm": "#475569", "tabnet": "#d97706", "ft_transformer": "#0891b2", "tabpfn": "#7c3aed"}
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    models = list(colors)
    x = np.arange(len(models))
    for offset, arm, label, alpha in [(-.18, "calibrated", "Calibrated model", 1), (.18, "corrected", "After market shrinkage", .45)]:
        vals = pooled[pooled.arm == arm].set_index("model").loc[models, "log_loss"]
        bars = ax.bar(x + offset, vals, width=.34, color=[colors[m] for m in models], alpha=alpha, label=label)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    market = pooled[pooled.model == "market"].log_loss.iloc[0]
    ax.axhline(market, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Market baseline ({market:.4f})")
    vals = pooled[pooled.arm.isin(["calibrated", "corrected", "market"])].log_loss
    ax.set_ylim(vals.min() - .015, vals.max() + .025)
    ax.set_xticks(x, ["LightGBM", "TabNet", "FT-Transformer", "TabPFN v2"])
    ax.set_ylabel("Log loss (lower is better; truncated y-axis)")
    ax.set_title("15,300 held-out matches | 2024/25 + 2025/26\nSingle seed, common training cap: 1,000 rows", loc="left", pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
