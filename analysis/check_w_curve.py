"""
Is w = 0 the right answer, or a broken objective?

Every arm's weight came out at zero, which is either a real finding -- the
market dominates stage 1 everywhere, so the best blend is no blend -- or a
mistake in how the weight is fitted. The difference matters enough to check
directly, so this sweeps w by hand on one fold and prints what happens.

Three curves, because they can disagree and the disagreement is the point:

  log loss, all matches   what fit_static_shrinkage currently minimises
  log loss, bet band      the same measure, on the matches a strategy acts on
  realised ROI, bet band  what the strategy is actually paid

A weight that is worthless for the first can still matter for the third. Log
loss asks whether the blend describes every match better; the strategy only
ever asks whether the bets it places win.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from two_stage import (MatchEngine, apply_static_shrinkage, load_multiseason,
                       make_folds, outcome_cols, probs, rankable, score_pool,
                       select_bets, selection_band, OP_CRITERION,
                       STAGE1_FEATURES, STAGE1_REQUIRED)

GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
COVERAGE = 0.10


def roi_of(sub: pd.DataFrame, p: np.ndarray, coverage: float) -> tuple:
    """
    Flat-stake ROI of the top `coverage` bets, chosen by the PIPELINE'S rule.

    This used to rank by expected value over all three outcomes, draws
    included, while the pipeline ranks home and away bets by edge. A curve
    drawn with a different rule says nothing about the pipeline's, so this
    now calls the pipeline's own selection. At w = 0 the blend is the market,
    there is no edge to rank, and the row is left blank rather than filled
    with a sort of rounding residue.
    """
    frame = sub.copy()
    frame[outcome_cols("p_corr")] = p
    if not rankable(frame, "p_corr", OP_CRITERION):
        return np.nan, 0, np.nan
    tau = float(np.quantile(score_pool(frame, "p_corr", criterion=OP_CRITERION),
                            1.0 - coverage))
    bets = select_bets(frame, tau, "p_corr", criterion=OP_CRITERION)
    if bets.empty:
        return np.nan, 0, np.nan
    roi = float(bets["PnL"].sum() / bets["Stake"].sum() * 100)
    return roi, len(bets), float(bets["Score"].mean() * 100)


def main() -> None:
    df = load_multiseason("matches_all.csv")
    folds, _ = make_folds(df, STAGE1_FEATURES, required=STAGE1_REQUIRED)
    fold = folds[-1]                       # the most recent season
    tr, va, te = fold.train, fold.val, fold.test
    print(f"\nfold: train {len(tr):,}  val {va['Season'].iat[0]} ({len(va):,})  "
          f"test {fold.test_unit} ({len(te):,})")

    # rho from out-of-fold predictions on the training seasons, as the pipeline
    # does. Fitting it on the validation season and then predicting that same
    # season, as this used to, scored the draw correction in-sample.
    engine = MatchEngine(seed=42)
    oof = engine.fit_predict_oof(tr)
    engine.fit(tr)
    engine.fit_rho(oof, tr)
    p_model = np.vstack([engine.probs_from_physical(oof, tr),
                         engine.predict_proba(va)])
    p_ref = np.vstack([probs(tr, "p_ref"), probs(va, "p_ref")])
    y = np.concatenate([tr["target"].astype(int), va["target"].astype(int)])
    pool = pd.concat([tr, va], ignore_index=True)

    band = selection_band(p_ref, p_model, COVERAGE)
    print(f"band: {band.sum():,} of {len(band):,} matches")
    print("Scored on the fitting window (train out-of-fold + validation), not "
          "the test season.\n")
    print(f"{'w':>5}  {'log loss all':>12}  {'log loss band':>13}  "
          f"{'ROI top10 %':>10}  {'bets':>6}  {'edge %':>10}")
    for w in GRID:
        blend = apply_static_shrinkage(p_ref, p_model, w)
        ll_all = log_loss(y, blend, labels=[0, 1, 2])
        ll_band = log_loss(y[band], blend[band], labels=[0, 1, 2])
        roi, n, exp = roi_of(pool, blend, COVERAGE)
        print(f"{w:>5.1f}  {ll_all:>12.4f}  {ll_band:>13.4f}  "
              f"{roi:>10.2f}  {n:>6,}  {exp:>10.2f}")


if __name__ == "__main__":
    main()
