"""
Does the new data make a market-blind model good enough to be worth shrinking?

The whole project stalled on one number: stage 1 closed about a quarter of the
gap between knowing nothing and the market, so the fitted shrinkage weight
collapsed to zero and the "corrected" model was just the market. This asks,
cheaply and before any pipeline surgery, what Elo, squad value and xG do to
that number.

Walk-forward inside the season: the matches are cut into chronological blocks,
each block predicted by a model trained only on the blocks before it. No odds
are given to the model -- the market is the thing being measured against, and
is scored on exactly the same rows.

The model is a regularised logistic regression, not a boosted forest. A forest
of 300 trees on a thousand-match fold scored 1.07 where Elo's own arithmetic
scored 1.02: it memorised the fold and reported the memory as confidence. That
is the same overfitting the whole project is about, showing up one level down,
and the fix is the same -- a model small enough for the data it has.

Elo's own probabilities are reported as a row of their own. Any feature set
that cannot beat them has not earned its place.

Feature sets are tested on the rows where they exist, and the market baseline
is recomputed on those same rows every time. Comparing a five-league result
against a 22-division baseline would flatter or damn a feature for no reason
other than which matches it covers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from two_stage import shin_de_vig

BLOCKS = 6
TARGET = {"H": 0, "D": 1, "A": 2}

ELO = ["elo_diff", "elo_p_H", "elo_p_D", "elo_p_A"]
VALUE = ["value_log_ratio", "home_value_rel", "away_value_rel"]
XG = ["home_xg_for_5", "home_xg_against_5", "home_xg_diff_10", "home_luck_10",
      "away_xg_for_5", "away_xg_against_5", "away_xg_diff_10", "away_luck_10"]


def walk_forward(df: pd.DataFrame, features: list, seed: int = 42) -> np.ndarray:
    """Out-of-sample probabilities, each block predicted from its own past."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # A stable sort, and the answers carried back on the caller's own index.
    # An ordinary sort_values here reorders matches that share a date, and the
    # returned array then lines up with nothing: every probability scored
    # against some other match's result. It cost an afternoon.
    d = df.sort_values("Date", kind="mergesort")
    block = pd.qcut(np.arange(len(d)), BLOCKS, labels=False)
    out = pd.DataFrame(np.nan, index=d.index, columns=[0, 1, 2])
    for b in range(1, BLOCKS):
        tr, te = d.index[block < b], d.index[block == b]
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=0.5,
                                                 random_state=seed))
        model.fit(d.loc[tr, features], d.loc[tr, "y"])
        out.loc[te] = model.predict_proba(d.loc[te, features])
    return out.reindex(df.index).to_numpy()


def market_probs(df: pd.DataFrame) -> np.ndarray:
    return np.array([shin_de_vig(h, d, a) for h, d, a
                     in df[["AvgCH", "AvgCD", "AvgCA"]].to_numpy()])


def report(df: pd.DataFrame, name: str, features: list | None) -> None:
    """features=None scores Elo's own probabilities, with no model in between."""
    need = (features or []) + ["elo_p_H", "AvgCH", "AvgCD", "AvgCA"]
    sub = df.dropna(subset=need).copy().sort_values("Date").reset_index(drop=True)
    if len(sub) < 500:
        print(f"{name:<34} too few rows ({len(sub)})")
        return
    if features is None:
        p_model = sub[["elo_p_H", "elo_p_D", "elo_p_A"]].to_numpy().copy()
        p_model[pd.qcut(np.arange(len(sub)), BLOCKS, labels=False) == 0] = np.nan
    else:
        p_model = walk_forward(sub, features)
    scored = ~np.isnan(p_model[:, 0])
    y = sub.loc[scored, "y"].to_numpy()

    base = np.tile(np.bincount(sub.loc[~scored, "y"], minlength=3) /
                   max((~scored).sum(), 1), (scored.sum(), 1))
    ll_base = log_loss(y, base, labels=[0, 1, 2])
    ll_model = log_loss(y, p_model[scored], labels=[0, 1, 2])
    ll_market = log_loss(y, market_probs(sub[scored]), labels=[0, 1, 2])
    closed = (ll_base - ll_model) / (ll_base - ll_market)
    print(f"{name:<34} n={scored.sum():>5,}  model {ll_model:.4f}  "
          f"market {ll_market:.4f}  base {ll_base:.4f}   gap closed {closed:>5.0%}")


def main() -> None:
    df = pd.read_csv("dataset_2526.csv", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["FTR"].isin(TARGET)].copy()
    df["y"] = df["FTR"].map(TARGET)
    weather = [c for c in ["temperature_2m_mean", "precipitation_sum",
                           "wind_speed_10m_max"] if c in df.columns]

    print(f"2025/26, {len(df):,} matches\n")
    print("ALL 22 DIVISIONS")
    report(df, "  elo probabilities, no model", None)
    report(df, "  elo", ELO)
    if weather:
        report(df, "  elo + weather", ELO + weather)

    print("\n11 TOP DIVISIONS (squad values exist)")
    top = df.dropna(subset=VALUE)
    report(top, "  elo probabilities, no model", None)
    report(top, "  elo", ELO)
    report(top, "  elo + squad value", ELO + VALUE)
    if weather:
        report(top, "  elo + squad value + weather", ELO + VALUE + weather)

    print("\n5 LEAGUES (xG exists)")
    five = df.dropna(subset=XG[:1])
    report(five, "  elo probabilities, no model", None)
    report(five, "  elo", ELO)
    report(five, "  elo + xG", ELO + XG)
    report(five, "  elo + squad value + xG", ELO + VALUE + XG)


if __name__ == "__main__":
    main()
