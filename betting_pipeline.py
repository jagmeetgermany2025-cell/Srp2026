"""
betting_pipeline.py -- the two-stage selective betting pipeline, in one file.

This file merges four that used to sit side by side:

    two_stage.py        the pipeline, stage 1 on gradient-boosted trees
    two_stage_torch.py  the same pipeline, stage 1 on a neural net
    elo.py              Elo ratings with every constant fitted
    seed_sweep.py       the pipeline re-run under many seeds

The method is unchanged. The two pipeline copies were identical apart from the
stage 1 learner, so they are now one pipeline with a --learner switch:

    --learner torch     stage 1 MLP, stage 2 sklearn boosting   (two_stage_torch.py)
    --learner lgb       LightGBM in both stages                 (two_stage.py)
    --learner hgb       sklearn boosting in both stages
    --learner catboost  CatBoost in both stages

TORCH AND LIGHTGBM ARE BOTH IMPORTED LAZILY, and never into the same process.
On macOS the first parallel region after torch has run dies with
"OMP: Error #179" if LightGBM is loaded too -- a segfault with no traceback.
That is why the two pipelines used to be separate files. Importing each one only
when its learner is chosen keeps them apart without keeping two copies of the
code, and a guard refuses to load the second if the first is already in.

NO NUMBERS APPEAR IN THESE COMMENTS. A figure quoted in a comment is a figure
somebody has to trust without checking. Run it and read the output.

HOW TO RUN

    python betting_pipeline.py                                  # 2025/26 export
    python betting_pipeline.py run --mode multiseason --learner lgb
    python betting_pipeline.py run --stage2 deviation
    python betting_pipeline.py sweep --mode multiseason --seeds 100 --learner catboost --ensemble-k 5
    python betting_pipeline.py elo --source matches_multiseason.csv
    python betting_pipeline.py clv --mode multiseason --source matches_multiseason.csv
    python betting_pipeline.py ou  --mode multiseason --source matches_multiseason.csv
    python betting_pipeline.py traps --mode multiseason --source matches_multiseason.csv
    python betting_pipeline.py ah --mode multiseason --source matches_multiseason.csv
    python betting_pipeline.py download                          # older seasons, on your computer
    python betting_pipeline.py stack raw_history --existing matches_multiseason.csv
    python betting_pipeline.py value --source matches_multiseason.csv
    python betting_pipeline.py pinnacle --mode multiseason --source matches_multiseason.csv
    python betting_pipeline.py scan fixtures.csv --bank 1000          # today's bets
    python betting_pipeline.py scan fixtures.csv --history matches_all_seasons.csv  # + model view
    python betting_pipeline.py pinnacle-sweep --seeds 1,2,3,4,5 --learners hgb
    python betting_pipeline.py stage2-value --source matches_all_seasons.csv
    python betting_pipeline.py placebo --source matches_all_seasons.csv
    python betting_pipeline.py value --source matches_all_seasons.csv --exclude-books IW
    python betting_pipeline.py review --results matches_multiseason.csv  # how it's going

With no subcommand, "run" is assumed.

LAYOUT

    1  constants              8  shrinkage -- the correction
    2  loading                9  walk-forward folds
    3  features               10 selection
    4  estimators             11 evaluation
    5  stage 1, match engine  12 orchestration
    6  stage 2, market model  13 report
    7  auxiliary heads        14 Elo     15 seed sweep     16 command line
"""
import contextlib
import csv
import importlib
import io
import os
import sys
import time
import warnings

# Must be set before torch is imported, if it ever is. scikit-learn and torch
# each bring an OpenMP runtime, and two of them in one process is the crash
# described above.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, field, astuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict


# --- the two libraries that must never meet ----------------------------
_TORCH = None


def _torch():
    """Import torch on first use, and refuse if LightGBM is already loaded."""
    global _TORCH
    if _TORCH is None:
        if "lightgbm" in sys.modules:
            raise RuntimeError(
                "LightGBM is already loaded in this process; loading torch now "
                "risks the OpenMP crash. Run torch and lgb learners separately.")
        import torch
        from torch import nn
        torch.set_num_threads(1)          # one OpenMP pool in this process
        _TORCH = (torch, nn)
    return _TORCH


def _lightgbm():
    """Import LightGBM on first use, and refuse if torch is already loaded."""
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch is already loaded in this process; loading LightGBM now "
            "risks the OpenMP crash. Run torch and lgb learners separately.")
    import lightgbm
    return lightgbm


# --- the network that replaces the boosted trees in stage 1 --------------
TORCH_HIDDEN = (64, 32)
TORCH_EPOCHS = 200
TORCH_PATIENCE = 12
TORCH_BATCH = 512
TORCH_LR = 3e-3
TORCH_VAL_SHARE = 0.15

# --- which learner each stage uses, and how many copies it averages --------
# Set through configure(), never by editing these. The defaults reproduce the
# pipeline exactly as it was: the MLP in stage 1, sklearn's boosting in stage 2
# and one copy of each.
#
#   LEARNER     "torch"    -- stage 1 MLP, stage 2 sklearn boosting (default)
#               "hgb"      -- both stages sklearn boosting, Poisson in stage 1
#               "catboost" -- both stages CatBoost, Poisson in stage 1
#               "lgb"      -- both stages LightGBM, Poisson in stage 1; this is
#                             exactly the old two_stage.py
#   ENSEMBLE_K  copies per model. Above one, each copy is fitted on a bootstrap
#               resample under its own seed and their predictions averaged.
#
# THE ENSEMBLE NEVER CROSSES THE SEAM. Stage 1 copies are averaged with stage 1
# copies and stage 2 with stage 2. Averaging across the stages would put the
# market price into the arm that is meant to be blind to it, and the shrinkage
# would then be correcting a copy of the market. The auxiliary heads -- the
# uncertainty head, the stake model, the reaction model -- are left as they
# were; they are diagnostics, not the estimate being corrected.
LEARNERS = ("torch", "hgb", "catboost", "lgb")
LEARNER = "torch"
ENSEMBLE_K = 1

CATBOOST_PARAMS = dict(iterations=300, learning_rate=0.03, depth=4,
                       l2_leaf_reg=5.0, verbose=0, thread_count=1,
                       allow_writing_files=False)


def configure(learner: str = None, ensemble_k: int = None) -> None:
    """Choose the learner and the ensemble size for every later fit."""
    global LEARNER, ENSEMBLE_K
    if learner is not None:
        if learner not in LEARNERS:
            raise ValueError(f"unknown learner {learner!r}; use one of {LEARNERS}")
        LEARNER = learner
    if ensemble_k is not None:
        if ensemble_k < 1:
            raise ValueError("ensemble_k must be at least 1")
        ENSEMBLE_K = int(ensemble_k)


# =========================================================================
# PART 1 -- CONSTANTS
# =========================================================================

# Paths are anchored to the project root, not to the working directory, so a
# script runs the same from anywhere. Everything it reads or writes lives under
# data/; everything it imports lives under pipeline/.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def resolve(path) -> Path:
    """A bare name means a file in data/; an explicit path is left alone."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else DATA / p


OUTCOMES = ("H", "D", "A")
TARGET_MAP = {"H": 0, "D": 1, "A": 2}

COLUMN_RENAMES = {
    "PinnacleH": "PSH", "PinnacleD": "PSD", "PinnacleA": "PSA",
    "PH": "PSH", "PD": "PSD", "PA": "PSA",
    "BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA",
    "BbMxH": "MaxH", "BbMxD": "MaxD", "BbMxA": "MaxA",
}


def apply_column_renames(df: pd.DataFrame, renames: dict = None) -> pd.DataFrame:
    """
    COLUMN_RENAMES, applied without ever creating two columns of one name.

    A stacked file can hold an old name and its new one side by side -- BbAvH
    for the seasons before 2019/20 and AvgH after. A plain rename would leave
    two AvgH columns, and every later df["AvgH"] would return a table instead
    of a column. So where the new name already exists, it is filled from the
    old one row by row and the old one is dropped.
    """
    renames = COLUMN_RENAMES if renames is None else renames
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for old, new in renames.items():
        if old not in df.columns:
            continue
        if new in df.columns:
            df[new] = pd.to_numeric(df[new], errors="coerce").fillna(
                pd.to_numeric(df[old], errors="coerce"))
            df = df.drop(columns=old)
        else:
            df = df.rename(columns={old: new})
    return df

# --- price sources -------------------------------------------------------
# The benchmark and the execution price must be DIFFERENT columns. Using one
# book for both is circular: the edge is measured against the de-vigged price
# of the same book that pays the bet, so a real edge and a modelling error look
# identical. Pinnacle is also the least complete, so requiring it discards most
# of the data before modelling starts.
#
# They must also be quoted at the SAME MOMENT. Selecting on the closing price
# and settling at an earlier one lets the backtest bet on information it could
# not have had while that earlier price was on offer: the closing price already
# knows what moved the market, so "the early price beats the closing price"
# becomes a selection rule. So each benchmark carries its own execution prices,
# and pick_price_columns() hands them out as a set.
BENCHMARK_COLS = ["AvgCH", "AvgCD", "AvgCA"]   # what we measure against
EXECUTION_COLS = ["MaxCH", "MaxCD", "MaxCA"]   # what we are paid at
REFERENCE_COLS = ["AvgH", "AvgD", "AvgA"]      # the opening average
OPENING_EXECUTION_COLS = ["MaxH", "MaxD", "MaxA"]
PINNACLE_COLS = ["PSH", "PSD", "PSA"]          # fallback only

# (benchmark, best price paid, average price) -- one row per moment, in order
# of preference. Pinnacle's PS* columns are pre-closing, so they pair with the
# pre-closing Max and Avg.
PRICE_MOMENTS = [
    (BENCHMARK_COLS, EXECUTION_COLS, BENCHMARK_COLS),
    (REFERENCE_COLS, OPENING_EXECUTION_COLS, REFERENCE_COLS),
    (PINNACLE_COLS, OPENING_EXECUTION_COLS, REFERENCE_COLS),
]

# --- load-time regression checks -----------------------------------------
# Filled in only from a run somebody has inspected, which makes them a
# regression test: a later change to the loader
# that alters the row count or the coverage then fails immediately instead of
# quietly producing different results. A number nobody has checked is worse
# than no number at all, because it looks verified.
#
# Measured 2026-09-22 on all-euro-data-2025-2026.csv, after the loader fix that
# stopped the first division (E0, 360 matches) being dropped. Loading any other
# export will fail these checks until they are re-measured for it.
# These describe all-euro-data-2025-2026.csv ONLY -- the single-season export
# they were measured on. load_and_prepare() checks them when reading that file
# and skips them otherwise, so a different export is not silently judged
# against another file's row count.
EXPECTED_SOURCE = "all-euro-data-2025-2026.csv"
EXPECTED_ROWS = 7466
EXPECTED_DIVISIONS = 22
EXPECTED_COVERAGE = {"AvgCH": 100.0, "MaxCH": 100.0, "MaxH": 99.84,
                     "HST": 92.59}
COVERAGE_TOLERANCE = 0.5      # percentage points

# Warn when the fitted correction weight goes above this. A high weight means
# the model is being trusted more than the market, which is worth stopping to
# explain rather than discovering later in a plot.
W_WARNING_LEVEL = 0.5

# And warn when it sits below this across folds: the corrected arm is then the
# market price under another name, and every comparison against it measures
# the market rather than a correction.
W_COLLAPSE_LEVEL = 0.05

# How bets are ranked at the operating point. "edge" is our probability minus
# the market's; "ev" multiplies by the odds and therefore drifts toward long
# prices as the threshold tightens. Both are reported in the tables.
OP_CRITERION = "edge"

# --- features ------------------------------------------------------------
# Stage 1 trains on these and nothing else. Every one is computable at kickoff
# and none of them touches a price.
BASE_FEATURE_COLS = [
    "home_rest", "away_rest", "diff_rest_days",
    "home_GF_5", "home_GA_5", "home_GD_5",
    "away_GF_5", "away_GA_5", "away_GD_5",
    "home_GF_10", "home_GA_10", "away_GF_10", "away_GA_10",
    "diff_roll_GF_5", "diff_roll_GA_5",
]

SHOT_FEATURE_COLS = [
    "home_SF_5", "home_SA_5", "home_STF_5", "home_STA_5",
    "away_SF_5", "away_SA_5", "away_STF_5", "away_STA_5",
    "home_shot_quality_5", "away_shot_quality_5",
]

# Features built by the scripts beside this file rather than read from the
# export. All three are market-blind and all three look only backwards:
# Elo carries a team's record into the match (build_elo.py), the history block
# is last season's table (build_history.py), and the xG stand-in is a Poisson
# fit of goals on shot counts, rolled over earlier matches (build_xgproxy.py).
ELO_FEATURE_COLS = ["elo_home", "elo_away", "elo_diff",
                    "elo_p_H", "elo_p_D", "elo_p_A"]
HISTORY_FEATURE_COLS = [f"{side}_{stat}" for side in ("home", "away")
                        for stat in ("prev_ppg", "prev_gd_pg", "prev_gf_pg",
                                     "prev_ga_pg", "prev_tier", "tier_change",
                                     "is_promoted", "is_new")]
PROXY_XG_FEATURE_COLS = [f"{side}_pxg_{stat}_{span}"
                         for side in ("home", "away")
                         for stat in ("for", "against", "diff", "luck")
                         for span in (5, 10)]

# Which league this is. Twenty-two divisions were pooled with no marker of
# which was which, so a 1.1-goal Greek match and a 3.5-goal German one trained
# the same model with no way to tell them apart.
LEAGUE_CONTEXT_COLS = ["Tier", "league_goals_20", "league_home_rate_20"]

STAGE1_FEATURES = (BASE_FEATURE_COLS + SHOT_FEATURE_COLS + ELO_FEATURE_COLS
                   + HISTORY_FEATURE_COLS + PROXY_XG_FEATURE_COLS
                   + LEAGUE_CONTEXT_COLS)

# A row is usable if it has these. The rest may be missing -- the tree models
# handle nulls natively, and the newer blocks are legitimately absent in some
# divisions. Requiring all of them would silently drop half the data.
STAGE1_REQUIRED = BASE_FEATURE_COLS

# built by: build_elo.py, build_history.py, build_xgproxy.py
EXTERNAL_FEATURES = ["elo/elo_by_match.csv",
                     "history/prev_season.csv",
                     "history/xgproxy_features.csv"]

# The de-vigged market price. Identical to outcome_cols("p_ref"), which is how
# it is read everywhere; named here so the lists below can be built above the
# helper that generates it.
MARKET_COLS = ["p_ref_H", "p_ref_D", "p_ref_A"]

# Price movement. A candidate predictor, and the market-reaction target.
MARKET_FEATURE_COLS = ["drift_H", "drift_D", "drift_A", "abs_drift"]

# Inputs to the per-match shrinkage weight. These answer "how much should we
# trust our own estimate here", not "what are these teams like". Rest days
# answer the second question and cannot do this job.
UNCERTAINTY_COLS = ["spread_mean", "overround_close", "p_ref_entropy",
                    "abs_deviation",
                    # How little we know here, as opposed to how open the match
                    # is. A fourth-tier fixture in the opening weeks, with no
                    # previous season on record, is one the model should be
                    # made to doubt -- and each of these says so directly.
                    "Tier", "min_played", "has_prev_season", "has_pxg"]

# What stage 2 may see, on top of stage 1's output.
MARKET_FEATURES = (MARKET_COLS
                   + ["spread_mean", "overround_close", "p_ref_entropy"]
                   + MARKET_FEATURE_COLS)

# The physical quantities stage 1 predicts. POST-MATCH columns: labels here,
# never features.
PHYSICAL_TARGETS = {
    "goals_home": "FTHG", "goals_away": "FTAG",
    "shots_home": "HS",   "shots_away": "AS",
    "sot_home":   "HST",  "sot_away":   "AST",
}

# The opening price. The ONLY price the market-reaction model may see, since
# its target is the move from opening to closing and the closing price is the
# answer.
OPENING_COLS = ["AvgH", "AvgD", "AvgA"]
DRIFT_COLS = [f"drift_{o}" for o in OUTCOMES]

# Properties of an already-selected bet that the stake model sizes on.
STAKE_FEATURES = ("p_est", "p_ref", "EV", "Odds")

# Any feature name containing one of these is a price or derived from one.
PRICE_TOKENS = ("PS", "Max", "Avg", "B365", "BW", "IW", "WH", "VC",
                "p_ref", "drift", "spread", "overround", "odds")

MAX_GOALS = 10          # Poisson grid depth; covers essentially every match
RHO_BOUNDS = (-0.25, 0.25)

BET_COLS = ["match_key", "Date", "Season", "Div", "Selection", "Odds",
            "Odds_Avg", "p_est", "p_ref", "EV", "Score", "IsWin", "Stake",
            "PnL", "PnL_Avg"]

# Division to country and tier, for the export parser.
TIERS = {
    "E0": ("England", 1), "E1": ("England", 2), "E2": ("England", 3),
    "E3": ("England", 4), "EC": ("England", 5),
    "SC0": ("Scotland", 1), "SC1": ("Scotland", 2), "SC2": ("Scotland", 3),
    "SC3": ("Scotland", 4), "D1": ("Germany", 1), "D2": ("Germany", 2),
    "SP1": ("Spain", 1), "SP2": ("Spain", 2), "I1": ("Italy", 1),
    "I2": ("Italy", 2), "F1": ("France", 1), "F2": ("France", 2),
    "N1": ("Netherlands", 1), "B1": ("Belgium", 1), "P1": ("Portugal", 1),
    "T1": ("Turkey", 1), "G1": ("Greece", 1),
}

TEXT_COLS = {"Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR", "HTR",
             "Referee", "League", "Country", "Season"}


def outcome_cols(prefix: str) -> list:
    """The home, draw and away triple for one prefix, generated in one place."""
    return [f"{prefix}_{o}" for o in OUTCOMES]


def probs(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """The H/D/A triple for one prefix, as a float array."""
    return df[outcome_cols(prefix)].to_numpy(float)


def has_arm(df: pd.DataFrame, prefix: str) -> bool:
    """Whether every column of a prefix's triple is present."""
    return all(c in df for c in outcome_cols(prefix))


def show(title: str, frame: pd.DataFrame, *notes: str) -> pd.DataFrame:
    """Print a titled table with optional footnotes, and return it unchanged."""
    print(f"\n{title}")
    print(frame.round(4).to_string(index=False))
    for note in notes:
        print(f"  {note}")
    return frame


def assert_market_blind(features: list, where: str = "stage 1") -> list:
    """
    Fail loudly if a price has leaked into a feature list that must not have one.

    WHY THIS IS AN ASSERTION AND NOT A COMMENT. The entire contribution rests on
    stage 1 being market-blind. If a price reaches it, its disagreement with the
    market becomes an echo of the market, the fitted w runs toward one, and the
    correction switches itself off while still appearing to run and still
    printing a plausible curve. That failure is silent, so it needs a guard.
    """
    leaked = [c for c in features
              if any(tok.lower() in c.lower() for tok in PRICE_TOKENS)]
    if leaked:
        raise ValueError(
            f"Market data leaked into {where}: {leaked}\n"
            "Stage 1 must be blind to price. Remove these or move them to "
            "MARKET_FEATURES, which stage 2 is allowed to see.")
    return features


# =========================================================================
# PART 2 -- LOADING
# =========================================================================


def load_export(path: str, season: str = "2526",
                encoding: str = "latin-1") -> pd.DataFrame:
    """
    Read the concatenated football-data export, which is malformed four ways.

    All four faults fail SILENTLY under pd.read_csv, which is why this exists:

      1. Semicolon-delimited, not comma.
      2. Comma decimal separator -- values are "1,44" not "1.44". This is the
         one that matters most: most numeric cells are affected and
         pd.to_numeric turns them all into NaN without raising.
      3. Embedded header rows -- each division sheet kept its own header, and
         the sheets have different column counts.
      4. Dates are m/d/yy, US order. dayfirst=True misparses silently.

    Splits the file into blocks at each embedded header, parses each block with
    its own columns, and concatenates on the union.
    """
    with open(path, encoding=encoding, newline="") as f:
        rows = [r for r in csv.reader(f, delimiter=";") if any(x.strip() for x in r)]

    # A header row is matched on endswith, not equality: the export's first
    # line arrives as "I Div", and an exact match treated it as data, so the
    # whole first division was discarded along with it.
    blocks, header, body = [], None, []
    for r in rows:
        if r[0].strip().endswith("Div"):          # a header row
            if header is not None and body:
                blocks.append((header, body))
            header, body = ["Div"] + [c.strip() for c in r[1:]], []
        elif header is None:
            raise ValueError(f"{path}: data row before any header row: "
                             f"{r[:5]}")
        else:
            body.append(r)
    if header is not None and body:
        blocks.append((header, body))

    frames = []
    for hdr, rws in blocks:
        n = len(hdr)
        rws = [r[:n] + [""] * (n - len(r)) for r in rws]      # pad or trim
        frames.append(pd.DataFrame(rws, columns=hdr))

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[df["HomeTeam"].astype(str).str.strip() != ""]

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    if df["Date"].isna().any():
        alt = pd.to_datetime(df.loc[df["Date"].isna(), "Date"],
                             dayfirst=True, errors="coerce")
        df.loc[df["Date"].isna(), "Date"] = alt

    num = [c for c in df.columns if c not in TEXT_COLS]
    conv = (df[num].astype(str)
                   .apply(lambda s: s.str.strip().str.replace(",", ".", regex=False))
                   .replace({"": None, "nan": None})
                   .apply(pd.to_numeric, errors="coerce"))

    meta = pd.DataFrame({
        "League": df["Div"].values,
        "Season": season,
        "Country": df["Div"].map(lambda x: TIERS.get(x, ("?", 0))[0]).values,
        "Tier": df["Div"].map(lambda x: TIERS.get(x, ("?", 0))[1]).values,
    }, index=df.index)

    out = pd.concat([df.drop(columns=num), conv, meta], axis=1)
    return out.sort_values("Date").reset_index(drop=True)


def add_league_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    How this division has been scoring lately, and how often the home side wins.

    Both are rolling means over the division's PREVIOUS matches, shifted by one
    so a match cannot see itself. They give the model the one thing it had no
    way to know: which league it is looking at.
    """
    out = df.sort_values("Date").copy()
    goals = out["FTHG"] + out["FTAG"]
    home_win = (out["FTR"] == "H").astype(float)
    out["league_goals_20"] = (goals.groupby(out["Div"]).transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).mean()))
    out["league_home_rate_20"] = (home_win.groupby(out["Div"]).transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).mean()))
    # SAME-DAY MATCHES MUST NOT SEE EACH OTHER. Shifting by one row only keeps
    # a match from seeing itself; a match later in the same day's sort order
    # would still see the result of one kicking off alongside it. Every match
    # on a date takes the value of that date's first row, which saw only
    # earlier days.
    for col in ("league_goals_20", "league_home_rate_20"):
        # iloc[0], not "first": "first" skips nulls, and would hand an early
        # match with no value the value of a later one.
        out[col] = out.groupby(["Div", "Date"])[col].transform(lambda x: x.iloc[0])
    if "Tier" not in out.columns:
        out["Tier"] = out["Div"].map(lambda d: TIERS.get(d, ("?", 0))[1])

    # How many league matches each side has played this season before this one.
    # Early in a season the model is working from almost nothing, and that is
    # exactly when it should be shrunk hardest toward the market.
    played = {}
    for side in ("HomeTeam", "AwayTeam"):
        played[side] = out.groupby([side, "Season"]).cumcount()
    out["min_played"] = np.minimum(played["HomeTeam"], played["AwayTeam"])
    return out.sort_index()


def attach_external_features(df: pd.DataFrame,
                             paths: list = None) -> pd.DataFrame:
    """
    Join the feature files built beside this script, if they are there.

    Absence is not an error: the pipeline still runs on the export alone, with
    the newer columns simply missing, and STAGE1_REQUIRED keeps such a run
    honest. Each file is keyed on division, date and the two team names.
    """
    key = ["Div", "Date", "HomeTeam", "AwayTeam"]
    out = df
    for rel in (paths or EXTERNAL_FEATURES):
        path = DATA / rel
        if not path.exists():
            if rel not in BUILT_IN_PIPELINE:
                warnings.warn(f"{rel} not built; its features will be missing")
            continue
        block = pd.read_csv(path, low_memory=False)
        block["Date"] = pd.to_datetime(block["Date"], errors="coerce")
        block = block.drop(columns=[c for c in ("Season", "month", "match_id")
                                    if c in block.columns])
        block = block.drop_duplicates(key)
        before = set(out.columns)
        out = out.merge(block, on=key, how="left")
        added = [c for c in out.columns if c not in before]
        if added:
            print(f"  + {Path(rel).stem}: {len(added)} columns, "
                  f"{out[added[0]].notna().mean():.0%} coverage")

    # Whether a block reached this match is itself information: it is missing
    # by division, and a division the model knows less about is one to shrink
    # harder. Flags rather than filled-in values, so nothing is invented.
    # Built here when the file is absent. Elo is NOT built here: its constants
    # are fitted, so it has to be fitted per fold -- see fold_elo().
    if "home_prev_ppg" not in out.columns:
        out = add_history_features(out)
    out["has_prev_season"] = (out["home_prev_ppg"].notna().astype(float)
                              if "home_prev_ppg" in out.columns else 0.0)
    out["has_pxg"] = (out["home_pxg_for_5"].notna().astype(float)
                      if "home_pxg_for_5" in out.columns else 0.0)
    return out


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    How complete the key columns are, season by season.

    It decides the modelling window: closing odds and best/average prices each
    start in different seasons, so anything depending on them can only be used
    from the season it becomes available. It also exposes provider changes,
    where a column keeps its name but changes meaning -- pooling across such a
    break would mix two different quantities under one heading.
    """
    cols = [c for c in ["B365H", "B365CH", "PSH", "MaxH", "AvgH", "AvgCH", "HST"]
            if c in df.columns]
    out = df.groupby("Season").agg(rows=("Date", "size"))
    for c in cols:
        out[c] = df.groupby("Season")[c].apply(
            lambda s: s.notna().mean() * 100).round(1)
    return out


def load_and_prepare(export_csv: str = "all-euro-data-2025-2026.csv",
                     season: str = "2526", strict: bool = True) -> pd.DataFrame:
    """
    Read the single-season export and build every feature.

    The checks are the important part. All four faults load_export() repairs
    fail silently -- nothing raises, rows vanish, odds columns go blank, and the
    only visible symptom is a coverage figure far below what it should be. So we
    assert on load and stop, rather than hope somebody notices later.
    """
    df = clean_prices(apply_column_renames(load_export(resolve(export_csv), season=season)))

    measured_here = Path(export_csv).name == EXPECTED_SOURCE
    checks = {
        "rows": (len(df), EXPECTED_ROWS if measured_here else None),
        "divisions": (df["Div"].nunique(),
                      EXPECTED_DIVISIONS if measured_here else None),
        "unparsed dates": (int(df["Date"].isna().sum()), 0),
    }
    problems = [f"{k}: got {got}, expected {want}"
                for k, (got, want) in checks.items()
                if want is not None and got != want]

    for col, want in (EXPECTED_COVERAGE if measured_here else {}).items():
        if col in df.columns and want is not None:
            got = df[col].notna().mean() * 100
            if abs(got - want) > COVERAGE_TOLERANCE:
                problems.append(f"{col} coverage: got {got:.1f}%, expected {want}%")

    if problems:
        msg = "Data did not match the expected export:\n  " + "\n  ".join(problems)
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)

    print(f"Loaded {len(df):,} matches, {df['Div'].nunique()} divisions, "
          f"{df['Date'].min().date()} to {df['Date'].max().date()}")

    # THE SAME FEATURE BUILD AS load_multiseason(). This used to stop at the
    # shot features, so an export run trained stage 1 on 26 of the 66
    # STAGE1_FEATURES -- no Elo, no previous season, no xG proxy, no league
    # context -- and said nothing, because the only warning about missing
    # blocks lives in attach_external_features(), which never ran.
    built = add_shot_features(add_market_features(calculate_rolling_features(df)))
    return attach_external_features(add_league_context(built))


def clean_prices(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Blank out price sets that cannot be real, before anything is built on them.

    WHY THIS EXISTS. One weekend in the multi-season file carries away-handicap
    prices around 4.3 with a "best" of 22.0 on lines priced near 2.0 everywhere
    else. A model hunting for value finds them instantly, and a backtest then
    reports a triple-digit return that is nothing but a typo. Two checks, both
    about whether a price COULD be real, neither about whether it was profitable:

      1. A best price more than 1.5 times the average quoted at the same
         moment is a stale or mistyped quote, not a price anyone could take
         at size. That single best price is discarded. (Checked first, while
         the average it is compared against is still there.)
      2. A single book's prices -- or the market average -- that add up to
         less than 100% would be a guaranteed profit at that book. Books do
         not quote that; the set is discarded.

    Discarded means missing, never zero or replaced: a row with no usable price
    in a market simply cannot be bet in that market.
    """
    out = df.copy()
    markets = {
        "1X2": (("H", "D", "A"),
                ["Avg", "AvgC", "PS", "PSC", "B365", "B365C", "BFE", "BFEC"],
                [("Max", "Avg"), ("MaxC", "AvgC")]),
        "O/U 2.5": ((">2.5", "<2.5"),
                    ["Avg", "AvgC", "P", "PC", "B365", "B365C", "BFE", "BFEC"],
                    [("Max", "Avg"), ("MaxC", "AvgC")]),
        "Asian handicap": (("H", "A"),
                           ["AvgAH", "AvgCAH", "PAH", "PCAH", "B365AH", "B365CAH",
                            "BFEAH", "BFECAH"],
                           [("MaxAH", "AvgAH"), ("MaxCAH", "AvgCAH")]),
    }
    report = []
    for market, (sides, books, best_pairs) in markets.items():
        for best, avg in best_pairs:
            for sd in sides:
                b, a = f"{best}{sd}", f"{avg}{sd}"
                if b not in out.columns or a not in out.columns:
                    continue
                bad = out[b] > 1.5 * out[a]
                if bad.any():
                    out.loc[bad, b] = np.nan
                    report.append((market, b, int(bad.sum()), "best > 1.5 x average"))
        for book in books:
            cols = [f"{book}{sd}" for sd in sides]
            if not all(c in out.columns for c in cols):
                continue
            prices = out[cols].apply(pd.to_numeric, errors="coerce")
            booksum = (1.0 / prices.where(prices > 1.0)).sum(axis=1, min_count=len(cols))
            bad = (booksum < 1.0) | (prices.notna().all(axis=1) & (prices <= 1.0).any(axis=1))
            if bad.any():
                out.loc[bad, cols] = np.nan
                report.append((market, book, int(bad.sum()), "set adds up to < 100%"))
    if verbose:
        if report:
            print("Price sanity check -- discarded as impossible:")
            for market, what, n, why in report:
                print(f"  {market:15s} {what:9s} {n:5d} rows  ({why})")
        else:
            print("Price sanity check: nothing discarded.")
    return out


def read_match_table(path) -> pd.DataFrame:
    """
    Read a stacked match file, in either of the two layouts it arrives in.

    Plain CSV (commas, dot decimals) and the European layout the
    football-data exports use (semicolons, COMMA decimals -- "2,4" for 2.4).
    Reading the second as plain CSV fails silently: every row lands in one
    column, or every price turns into text. So the header decides: more
    semicolons than commas means the European layout.
    """
    path = resolve(path)
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    opts = ({"sep": ";", "decimal": ","} if header.count(";") > header.count(",")
            else {})
    df = None
    for enc in ("utf-8-sig", "cp1252"):        # older files are Windows-encoded
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc, **opts)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:          # a few old files have ragged rows
            df = pd.read_csv(path, encoding=enc, engine="python",
                             on_bad_lines="skip", **opts)
            break
    if df is None:
        raise ValueError(f"could not read {path}")
    # Spreadsheet software reads the 1XBet headers "1XBD" / "1XBA" as numbers
    # and writes them back as "1,00 XBD". Put the names back.
    fix = {c: "1XB" + c.split("XB", 1)[1] for c in df.columns
           if "XB" in c and not c.startswith("1XB")}
    return harmonise_columns(df.rename(columns=fix))


# Before 2019/20 football-data took its market-wide prices from Betbrain and
# named them Bb*; from 2019/20 they are Max*/Avg*. Same meaning, same moment
# (pre-closing). Where a stacked file holds both eras, the new name is filled
# from the old one row by row, never overwritten.
BETBRAIN_NAMES = {
    **{f"BbMx{o}": f"Max{o}" for o in ("H", "D", "A", ">2.5", "<2.5")},
    **{f"BbAv{o}": f"Avg{o}" for o in ("H", "D", "A", ">2.5", "<2.5")},
    "BbAHh": "AHh",
    "BbMxAHH": "MaxAHH", "BbMxAHA": "MaxAHA",
    "BbAvAHH": "AvgAHH", "BbAvAHA": "AvgAHA",
}


def harmonise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give every era's price columns their current football-data names."""
    df = df.loc[:, ~df.columns.duplicated()]
    found = {old: new for old, new in BETBRAIN_NAMES.items() if old in df.columns}
    if not found:
        return df
    df = df.copy()
    for old, new in found.items():
        vals = pd.to_numeric(df[old], errors="coerce")
        df[new] = pd.to_numeric(df[new], errors="coerce").fillna(vals) \
            if new in df.columns else vals
    return df


def load_multiseason(path: str = "matches_all.csv",
                     drop_partial_seasons: bool = True,
                     min_season_matches: int = 1000) -> pd.DataFrame:
    """
    Load a stacked multi-season file and build every feature.

    This file is already clean -- ordinary commas, ISO dates, one Season column.
    None of the four faults load_export() exists to repair apply here, so it
    must NOT be routed through that loader; doing so would misread the dates.

    Part-played seasons are dropped by default. A season with a handful of
    matches contributes almost no test data while still counting as a
    walk-forward unit, which makes the fold diagnostics misleading.
    """
    df = clean_prices(apply_column_renames(read_match_table(path)))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Season"].astype(str)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    if drop_partial_seasons:
        counts = df["Season"].value_counts()
        partial = counts[counts < min_season_matches].index.tolist()
        if partial:
            print(f"Dropping part-played seasons: {sorted(partial)}")
            df = df[~df["Season"].isin(partial)]

    print(f"Loaded {len(df):,} matches, {df['Div'].nunique()} divisions, "
          f"{df['Season'].nunique()} seasons "
          f"({df['Date'].min().date()} to {df['Date'].max().date()})")
    print(coverage_table(df))

    built = add_shot_features(add_market_features(calculate_rolling_features(df)))
    return attach_external_features(add_league_context(built))


# =========================================================================
# PART 3 -- FEATURES
# =========================================================================


def shin_de_vig(odd_h: float, odd_d: float, odd_a: float) -> tuple:
    """
    Turn three decimal odds into three probabilities that sum to one.

    WHY NOT JUST NORMALISE. The bookmaker's margin is not spread evenly across
    outcomes -- longshots carry more of it than favourites, which is the
    favourite-longshot bias. Dividing through by the booksum assumes it is
    spread evenly and therefore misprices exactly the long odds where the
    selection rule does most of its work.

    Shin's model instead assumes the margin exists because some bettors are
    informed, and solves for the share of informed money z that makes the
    implied probabilities sum to one:

        p_i = [sqrt(z^2 + 4(1-z) q_i^2 / S) - z] / (2(1-z))

    with q_i = 1/odds_i and S = sum(q_i). Solved by bisection.
    """
    odds = (odd_h, odd_d, odd_a)
    if any(o is None or not np.isfinite(o) or o <= 1.0 for o in odds):
        return (np.nan, np.nan, np.nan)

    q = np.array([1.0 / o for o in odds], dtype=float)
    booksum = q.sum()
    if booksum <= 1.0:                      # no margin: usually a data error
        return tuple(q / booksum)

    def implied(z):
        root = np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / booksum)
        return (root - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.99
    for _ in range(30):
        mid = (lo + hi) / 2.0
        if implied(mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid

    p = implied((lo + hi) / 2.0)
    return tuple(p / p.sum())


def pick_price_columns(df: pd.DataFrame) -> tuple:
    """
    Choose the benchmark columns and the execution prices quoted with them.

    Returns (benchmark, best, average), all from ONE moment -- see the
    PRICE_MOMENTS comment for why they may never be mixed.

    Order of preference: closing average, then pre-match average, then Pinnacle.
    Closing average first because it is every book's final opinion and is
    independent of the price we bet at. Pinnacle last because using one book as
    both benchmark and execution price is circular, and it covers less of the
    data. The fallback exists because closing odds are only available in the
    later part of the historical record.
    """
    for bench, best, avg in PRICE_MOMENTS:
        cols = bench + best + avg
        if all(c in df.columns for c in cols) and df[bench[0]].notna().any():
            return bench, best, avg
    raise KeyError("No usable benchmark and execution columns. Tried "
                   + "; ".join(f"{b} paid at {x}" for b, x, _ in PRICE_MOMENTS))


def pick_benchmark_columns(df: pd.DataFrame) -> list:
    """The benchmark half of pick_price_columns()."""
    return pick_price_columns(df)[0]


def match_key(df: pd.DataFrame) -> pd.Series:
    """
    A stable unique id per match.

    bootstrap_roi() resamples matches rather than bets, so it needs to know
    which bets belong to the same match. A key that collided across divisions
    would merge unrelated matches and make the intervals too narrow.
    """
    return (pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            + "|" + df["HomeTeam"].astype(str)
            + "|" + df["AwayTeam"].astype(str))


# --- the long table, which every rolling feature is built on -------------
#
# THE BUG THESE THREE FUNCTIONS EXIST TO PREVENT. Group a wide match table by
# HomeTeam and you get a team's history at HOME only, so a side that played away
# midweek looks fully rested at the weekend and its recent form ignores half its
# games. The fix is to stop working on the wide table: reshape so each team
# appears once per match whatever the venue, compute there, then scatter back.


def long_table(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Reshape to one row per team per match.

    `stats` maps each output name to the pair of source columns carrying it from
    the home and away side's point of view, e.g.
    {"GF": ("FTHG", "FTAG"), "GA": ("FTAG", "FTHG")} -- goals for is the home
    team's goals on a home row and the away team's goals on an away row.

    `match_id` carries the wide frame's row index so split_back() can undo it.
    """
    sides = []
    for is_home, team_col, pick in ((True, "HomeTeam", 0), (False, "AwayTeam", 1)):
        side = pd.DataFrame({"Date": df["Date"], "Team": df[team_col]})
        for name, cols in stats.items():
            side[name] = df[cols[pick]]
        side["is_home"] = is_home
        side["match_id"] = df.index
        sides.append(side)
    return (pd.concat(sides)
              .sort_values(["Date", "match_id"]).reset_index(drop=True))


def rolling_by_team(long: pd.DataFrame, cols, span: int,
                    min_periods: int = 1) -> pd.DataFrame:
    """
    Exponentially weighted mean of each column over a team's previous matches.

    THE SHIFT IS THE WHOLE POINT. Every column is shifted by one match before
    the window is taken, so a match never sees its own result. Without that
    these are not features, they are the answer.
    """
    for col in cols:
        long[f"{col}_{span}"] = long.groupby("Team")[col].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=min_periods).mean())
    return long


def split_back(df: pd.DataFrame, long: pd.DataFrame, cols) -> pd.DataFrame:
    """Write long-table columns back onto the wide frame as home_/away_ pairs."""
    home = long[long["is_home"]].set_index("match_id")
    away = long[~long["is_home"]].set_index("match_id")
    for col in cols:
        df[f"home_{col}"] = home[col]
        df[f"away_{col}"] = away[col]
    return df


def calculate_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build every feature known before kickoff: de-vigged market probabilities,
    rest days for both teams, and rolling goals scored and conceded.

    LEAKAGE RULE: every rolling number is shifted by one match before use, so a
    match never sees its own result.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # --- market probabilities from the benchmark price --------------------
    bench, best, avg = pick_price_columns(df)
    devigged = [shin_de_vig(h, d, a)
                for h, d, a in zip(df[bench[0]], df[bench[1]], df[bench[2]])]
    for k, col in enumerate(MARKET_COLS):
        df[col] = [r[k] for r in devigged]

    # The prices a bet is placed and settled at, from the benchmark's moment.
    # Selection reads only these, so it cannot drift onto a different moment.
    for o, b, a in zip(OUTCOMES, best, avg):
        df[f"odds_best_{o}"] = df[b]
        df[f"odds_avg_{o}"] = df[a]

    matches = long_table(df, {"GF": ("FTHG", "FTAG"), "GA": ("FTAG", "FTHG")})

    # --- rest days, computed on the long table ---------------------------
    matches["rest"] = (matches.groupby("Team")["Date"].diff().dt.days
                              .fillna(14).clip(upper=14))

    # --- rolling goals ----------------------------------------------------
    for span in (5, 10):
        rolling_by_team(matches, ("GF", "GA"), span)
        matches[f"GD_{span}"] = matches[f"GF_{span}"] - matches[f"GA_{span}"]

    split_back(df, matches, ["rest"] + [f"{c}_{s}" for s in (5, 10)
                                        for c in ("GF", "GA", "GD")])

    df["diff_rest_days"] = df["home_rest"] - df["away_rest"]
    df["diff_roll_GF_5"] = df["home_GF_5"] - df["away_GF_5"]
    df["diff_roll_GA_5"] = df["home_GA_5"] - df["away_GA_5"]

    if "FTR" in df.columns:
        df["target"] = df["FTR"].map(TARGET_MAP)
    df["match_key"] = match_key(df)
    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build market movement and bookmaker disagreement features.

    Two quantities are computed here and they do OPPOSITE jobs.

    DRIFT (-> MARKET_FEATURE_COLS) is how far the price moved between opening
    and closing: a candidate predictor. Expect little from it. Our benchmark is
    the closing price, which already reflects whatever caused the move, so drift
    looks strongly related to the error in the OPENING price -- and that is
    circular, since the price moved because the opening price was wrong.

    DISPERSION (-> UNCERTAINTY_COLS) is how far the books disagree with each
    other. NOT predictive, and it must never enter a feature list. The
    correction step needs a per-match answer to "how much should we trust our
    own deviation here", and where the books disagree the true price is less
    settled, our deviation is more likely to be noise, and it should be shrunk
    harder. That is the winner's curse made measurable.

    No shifting is needed -- closing odds ARE the kickoff price. But a missing
    value is not a zero: zero drift claims the price did not move, which is a
    different statement from not knowing whether it moved.
    """
    df = df.copy()

    for o in OUTCOMES:
        open_col, close_col = f"Avg{o}", f"AvgC{o}"
        if open_col in df.columns and close_col in df.columns:
            df[f"drift_{o}"] = np.log(df[open_col]) - np.log(df[close_col])
        else:
            df[f"drift_{o}"] = np.nan
    df["abs_drift"] = df[[f"drift_{o}" for o in OUTCOMES]].abs().sum(axis=1)

    for o in OUTCOMES:
        max_col, avg_col = f"Max{o}", f"Avg{o}"
        if max_col in df.columns and avg_col in df.columns:
            df[f"spread_{o}"] = df[max_col] / df[avg_col] - 1.0
        else:
            df[f"spread_{o}"] = np.nan
    df["spread_mean"] = df[[f"spread_{o}" for o in OUTCOMES]].mean(axis=1)

    # How much more than a whole the closing prices add up to: the margin.
    bench = pick_benchmark_columns(df)
    df["overround_close"] = sum(1.0 / df[c] for c in bench) - 1.0

    # How uncertain the market itself is. An evenly priced match is genuinely
    # open; a heavily one-sided one is not.
    p_ref = probs(df, "p_ref")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["p_ref_entropy"] = -np.nansum(
            p_ref * np.log(np.clip(p_ref, 1e-12, 1)), axis=1)
    return df


def add_shot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling shot and shots-on-target features.

    Shots on target is the standard cheap stand-in for expected goals, which
    football-data.co.uk does not carry. It beats goals for the same reason
    expected goals does -- goals are rare, so a team's recent goal count is a
    noisy measure of how well it played, while its shot count is less noisy.

    Coverage trap: at least one division carries no match statistics at all.
    Those rows are left missing rather than filled, so the decision to drop or
    keep that division is made explicitly downstream instead of silently here.
    """
    df = df.copy()
    if not all(c in df.columns for c in ["HS", "AS", "HST", "AST"]):
        for c in SHOT_FEATURE_COLS:
            df[c] = np.nan
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    long = long_table(df, {"SF": ("HS", "AS"), "SA": ("AS", "HS"),
                           "STF": ("HST", "AST"), "STA": ("AST", "HST")})
    rolling_by_team(long, ("SF", "SA", "STF", "STA"), 5)
    long["shot_quality_5"] = long["STF_5"] / long["SF_5"].replace(0, np.nan)

    return split_back(df, long, ["SF_5", "SA_5", "STF_5", "STA_5",
                                 "shot_quality_5"])


# =========================================================================
# PART 4 -- ESTIMATORS
# =========================================================================


def make_classifier(seed: int = 42):
    """
    The outcome classifier, returned UNFITTED so every fold gets a fresh one.

    Settings are deliberately small: the training windows are short and a larger
    model would fit noise. They are fixed here so every fold and every
    comparison uses identical settings, and differences come from the pipeline
    rather than from quiet retuning.
    """
    if ENSEMBLE_K > 1:
        return BaggedClassifier(learner=LEARNER, k=ENSEMBLE_K, seed=seed)
    return base_classifier(LEARNER, seed)


def base_classifier(learner: str, seed: int):
    """One stage 2 classifier. Only CatBoost and LightGBM replace sklearn's boosting."""
    if learner == "lgb":
        return _lightgbm().LGBMClassifier(
            n_estimators=100, learning_rate=0.01, num_leaves=15, max_depth=3,
            subsample=0.7, colsample_bytree=0.7, random_state=seed, verbosity=-1)
    if learner == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(loss_function="MultiClass", random_seed=seed,
                                  **{**CATBOOST_PARAMS, "depth": 3})
    return HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.01, max_leaf_nodes=15,
        max_depth=3, random_state=seed)


def base_count_regressor(learner: str, seed: int):
    """One stage 1 regressor for goals and shots, always with a Poisson loss."""
    if learner == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function="Poisson", random_seed=seed,
                                 **CATBOOST_PARAMS)
    if learner == "lgb":
        return _lightgbm().LGBMRegressor(
            objective="poisson", n_estimators=200, learning_rate=0.05,
            num_leaves=15, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=-1)
    if learner == "hgb":
        return HistGradientBoostingRegressor(
            loss="poisson", max_iter=200, learning_rate=0.05,
            max_leaf_nodes=15, max_depth=4, random_state=seed)
    return TorchCountRegressor(seed)


class _Bagged(BaseEstimator):
    """
    K copies of one model, each on its own subsample and seed.

    WHY RESAMPLE AND NOT JUST RESEED. sklearn's boosting without subsampling is
    deterministic on windows this size -- early stopping is off below 10,000
    rows, so random_state changes nothing. K reseeded copies would be K
    identical models. Resampling rows gives every learner something real to
    disagree about.

    WITHOUT REPLACEMENT, NOT A BOOTSTRAP. A bootstrap repeats rows, and the MLP
    carves its early-stopping set out of whatever it is given: copies of the
    same match then sat on both sides of that split, early stopping watched a
    score it had trained on, and the ensemble came out worse than one copy.
    Each copy takes a random SUBSAMPLE share of the rows instead, each row once.

    It is one estimator to the outside, so cross_val_predict clones and fits the
    whole ensemble inside each fold. The out-of-fold predictions that w, rho and
    tau are fitted on therefore come from the ensemble too, not from a single
    copy standing in for it.
    """

    def __init__(self, learner: str = "torch", k: int = 5, seed: int = 42,
                 subsample: float = 0.8):
        self.learner = learner
        self.k = k
        self.seed = seed
        self.subsample = subsample

    def _new(self, seed):                                   # pragma: no cover
        raise NotImplementedError

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        self.members_ = []
        n = max(int(round(len(X) * self.subsample)), 1)
        for i in range(self.k):
            idx = np.sort(rng.choice(len(X), size=n, replace=False))
            self.members_.append(self._new(self.seed + 1000 * (i + 1))
                                 .fit(X[idx], y[idx]))
        return self


class BaggedCountRegressor(RegressorMixin, _Bagged):
    """Stage 1: the average of K Poisson rates."""

    def _new(self, seed):
        return base_count_regressor(self.learner, seed)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        rate = np.mean([m.predict(X) for m in self.members_], axis=0)
        return np.clip(rate, 1e-4, 25.0)


class BaggedClassifier(ClassifierMixin, _Bagged):
    """Stage 2: the average of K probability vectors."""

    def _new(self, seed):
        return base_classifier(self.learner, seed)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return super().fit(X, y)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros((len(X), len(self.classes_)))
        for m in self.members_:
            # A resample can miss a class; line each copy's columns up by label.
            cols = np.searchsorted(self.classes_, np.asarray(m.classes_).ravel())
            out[:, cols] += m.predict_proba(X)
        out /= len(self.members_)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchCountRegressor(BaseEstimator, RegressorMixin):
    """
    A small MLP fitted with a Poisson loss, wearing an sklearn-shaped coat.

    It inherits the sklearn base classes because out-of-fold predictions go
    through cross_val_predict, which clones the estimator, and cloning needs
    get_params. The coat has to be real.

    POISSON, NOT SQUARED ERROR, for the same reason the tree version uses it:
    goals and shots are counts, and the scoreline grid downstream assumes a
    Poisson. The network predicts a log rate and exponentiates it, so the
    prediction cannot come out negative -- which the grid requires and squared
    error does not guarantee.

    MISSING VALUES ARE FILLED AND FLAGGED. Trees take nulls natively; a
    network cannot. Each column is filled with the training median and paired
    with a binary column saying it was filled. The flag matters: these
    features are missing by division rather than at random, so "no previous
    season on record" is information, not noise to be papered over. This is
    the one genuine difference in kind between the two learners, and part of
    what the comparison measures.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.net = None
        self.median_ = None
        self.mean_ = None
        self.std_ = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        missing = np.isnan(X).astype(np.float32)
        filled = np.where(np.isnan(X), self.median_, X)
        z = (filled - self.mean_) / self.std_
        return np.hstack([z, missing]).astype(np.float32)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.median_ = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_)
        filled = np.where(np.isnan(X), self.median_, X)
        self.mean_ = filled.mean(axis=0)
        self.std_ = np.where(filled.std(axis=0) < 1e-8, 1.0, filled.std(axis=0))

        torch, nn = _torch()
        torch.manual_seed(self.seed)
        design = self._design(X)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(design))
        cut = max(int(len(order) * (1 - TORCH_VAL_SHARE)), 1)
        tr_idx, va_idx = order[:cut], order[cut:]

        xt = torch.from_numpy(design[tr_idx])
        yt = torch.from_numpy(y[tr_idx].astype(np.float32))
        xv = torch.from_numpy(design[va_idx])
        yv = torch.from_numpy(y[va_idx].astype(np.float32))

        layers, width = [], design.shape[1]
        for size in TORCH_HIDDEN:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

        opt = torch.optim.Adam(self.net.parameters(), lr=TORCH_LR,
                               weight_decay=1e-4)
        loss_fn = nn.PoissonNLLLoss(log_input=True, full=False)
        best, best_state, waited = np.inf, None, 0

        for _ in range(TORCH_EPOCHS):
            self.net.train()
            for start in range(0, len(xt), TORCH_BATCH):
                xb, yb = xt[start:start + TORCH_BATCH], yt[start:start + TORCH_BATCH]
                opt.zero_grad()
                loss_fn(self.net(xb).squeeze(-1), yb).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                score = float(loss_fn(self.net(xv).squeeze(-1), yv))
            if score < best - 1e-5:
                best, waited = score, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= TORCH_PATIENCE:      # stop before it memorises
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        torch, _ = _torch()
        self.net.eval()
        with torch.no_grad():
            log_rate = self.net(torch.from_numpy(
                self._design(np.asarray(X, dtype=float)))).squeeze(-1)
        return np.clip(np.exp(log_rate.numpy()), 1e-4, 25.0)


def make_regressor(seed: int = 42, count: bool = False):
    """
    The regressor used for physical targets and the uncertainty head.

    Given more capacity than the classifier: predicting a continuous quantity
    from structured inputs is an easier, more structured problem than predicting
    a football result.

    count=True fits a Poisson loss instead of squared error. Goals, shots and
    shots on target are counts, and their predictions are then fed into a
    Poisson scoreline grid -- so squared error was fitting one distribution and
    the grid assuming another. Poisson also keeps predictions positive, which
    the grid requires and squared error does not guarantee.
    """
    if count:                      # the physical targets -- this is the swap
        if ENSEMBLE_K > 1:
            return BaggedCountRegressor(learner=LEARNER, k=ENSEMBLE_K, seed=seed)
        return base_count_regressor(LEARNER, seed)
    if LEARNER == "lgb":           # the uncertainty head and the other helpers,
        return _lightgbm().LGBMRegressor(   # exactly as two_stage.py had them
            objective="regression", n_estimators=200, learning_rate=0.05,
            num_leaves=15, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=-1)
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        max_depth=4, random_state=seed)


# =========================================================================
# PART 5 -- STAGE 1: THE MATCH ENGINE
# =========================================================================


def scoreline_grid(lam_home: np.ndarray, lam_away: np.ndarray,
                   rho: float = 0.0, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    The joint scoreline distribution, with the Dixon-Coles low-score correction.

    Independent Poisson understates draws, because real scorelines are
    correlated at 0-0 and 1-1 in a way two independent rates cannot express.
    Dixon-Coles multiplies the four lowest scorelines by

        tau(0,0) = 1 - lam_h lam_a rho     tau(0,1) = 1 + lam_h rho
        tau(1,0) = 1 + lam_a rho           tau(1,1) = 1 - rho

    and leaves everything else alone. A NEGATIVE rho raises 0-0 and 1-1 while
    lowering 1-0 and 0-1, which is the direction that adds draw probability.
    rho = 0 gives plain independent Poisson back exactly.

    Returns an (n, G, G) array indexed [match, home goals, away goals],
    renormalised, since the correction does not preserve total mass.
    """
    lam_home = np.clip(np.asarray(lam_home, float), 1e-6, None)
    lam_away = np.clip(np.asarray(lam_away, float), 1e-6, None)

    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[None, :], lam_home[:, None])      # (n, G)
    pa = poisson.pmf(goals[None, :], lam_away[:, None])      # (n, G)
    joint = ph[:, :, None] * pa[:, None, :]                  # (n, G, G)

    if rho:
        tau = np.ones_like(joint)
        tau[:, 0, 0] = 1.0 - lam_home * lam_away * rho
        tau[:, 0, 1] = 1.0 + lam_home * rho
        tau[:, 1, 0] = 1.0 + lam_away * rho
        tau[:, 1, 1] = 1.0 - rho
        # tau must stay positive or the "probabilities" go negative. Clipping
        # rather than raising keeps an over-ambitious rho from killing a fold.
        joint = joint * np.clip(tau, 1e-9, None)

    return joint / joint.sum(axis=(1, 2), keepdims=True)


def outcome_probs(lam_home: np.ndarray, lam_away: np.ndarray,
                  rho: float = 0.0, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Turn expected goals into P(home), P(draw), P(away) by summing the three
    triangles of the scoreline grid.

    WHY THIS RATHER THAN A CLASSIFIER. This is what makes stage 1 a match
    engine. It predicts how a match is played -- how many goals each side
    creates -- and the outcome probability falls out of that, rather than being
    fitted directly. The same grid also gives over/under and both-teams-to-score
    for free; see scoreline_markets().
    """
    joint = scoreline_grid(lam_home, lam_away, rho, max_goals)
    goals = np.arange(joint.shape[1])
    home_goals, away_goals = goals[:, None], goals[None, :]
    p = np.stack([
        (joint * (home_goals > away_goals)).sum(axis=(1, 2)),
        (joint * (home_goals == away_goals)).sum(axis=(1, 2)),
        (joint * (home_goals < away_goals)).sum(axis=(1, 2)),
    ], axis=1)
    return p / p.sum(axis=1, keepdims=True)


def scoreline_markets(lam_home, lam_away, rho=0.0, line=2.5,
                      max_goals=MAX_GOALS) -> pd.DataFrame:
    """
    Over/under and both-teams-to-score, read off the same grid.

    Not used by the main pipeline. It exists because the grid is already built,
    and because being able to price a second market is the practical argument
    for a scoreline engine over a three-class classifier.
    """
    joint = scoreline_grid(lam_home, lam_away, rho, max_goals)
    goals = np.arange(joint.shape[1])
    total = goals[:, None] + goals[None, :]
    btts = (goals[:, None] > 0) & (goals[None, :] > 0)
    return pd.DataFrame({
        f"p_over_{line}": (joint * (total > line)).sum(axis=(1, 2)),
        f"p_under_{line}": (joint * (total < line)).sum(axis=(1, 2)),
        "p_btts": (joint * btts).sum(axis=(1, 2)),
    })


def fit_dixon_coles_rho(lam_home: np.ndarray, lam_away: np.ndarray,
                        y_val: np.ndarray) -> float:
    """
    Fit rho by minimising validation log loss of the resulting H/D/A triple.

    WHY VALIDATION AND NOT TRAINING, and why log loss rather than the scoreline
    likelihood. Same reasoning as fit_static_shrinkage(): the quantity is a
    single scalar tuned to make the OUTPUT better, and tuning it on data the
    engine was fitted on would make the correction too weak. Log loss is used
    because the outcome triple is what the pipeline consumes -- fitting rho to
    the full scoreline likelihood would optimise a distribution we then throw
    most of away.
    """
    def objective(rho):
        return log_loss(y_val, outcome_probs(lam_home, lam_away, rho),
                        labels=[0, 1, 2])

    return float(minimize_scalar(objective, bounds=RHO_BOUNDS,
                                 method="bounded").x)


@dataclass
class MatchEngine:
    """
    Stage 1. One regressor per physical quantity, plus a scoreline layer.

    Fitted on form only. Holds no knowledge of any price.
    """

    seed: int = 42
    use_dixon_coles: bool = True
    targets: dict = field(default_factory=lambda: dict(PHYSICAL_TARGETS))
    features: list = None
    models: dict = field(default_factory=dict)
    rho: float = 0.0
    rho_by_div: dict = field(default_factory=dict)
    # Swap the estimator without touching the harness. Any callable taking a
    # seed and returning an unfitted sklearn-style regressor works.
    regressor_factory: object = None
    fitted_cols_: list = None

    def _new_regressor(self):
        if self.regressor_factory:
            return self.regressor_factory(self.seed)
        return make_regressor(self.seed, count=True)

    def _cols(self, df: pd.DataFrame) -> list:
        if self.fitted_cols_ is not None:              # predicting: fixed set
            return self.fitted_cols_
        return self._train_cols(df)

    def _train_cols(self, df: pd.DataFrame) -> list:
        """
        The features to fit on, decided once from the TRAINING data and kept.

        A column with no values at all in the training window is dropped: the
        previous-season block is empty in the first season on file, and a
        learner cannot bin a column that holds nothing. Keeping the choice
        means prediction always sees the same columns the model was fitted on.
        """
        feats = self.features or STAGE1_FEATURES
        cols = [c for c in feats if c in df.columns and df[c].notna().any()]
        self.fitted_cols_ = assert_market_blind(cols)
        return self.fitted_cols_

    def _usable_targets(self, df: pd.DataFrame) -> dict:
        """Drop targets whose source column is absent -- EC has no shot data."""
        return {name: col for name, col in self.targets.items()
                if col in df.columns}

    def fit(self, train_df: pd.DataFrame,
            val_df: pd.DataFrame = None) -> "MatchEngine":
        """
        Fit one regressor per physical target, then rho on the validation fold.

        val_df is optional so the engine can be used alone, but without it rho
        stays at zero and the engine is plain independent Poisson.
        """
        cols = self._train_cols(train_df)
        for name, target_col in self._usable_targets(train_df).items():
            ok = train_df[target_col].notna()
            model = self._new_regressor()
            model.fit(train_df.loc[ok, cols], train_df.loc[ok, target_col])
            self.models[name] = model

        if self.use_dixon_coles and val_df is not None:
            self.fit_rho(self.predict_physical(val_df), val_df)
        return self

    def fit_rho(self, phys: pd.DataFrame, df: pd.DataFrame) -> "MatchEngine":
        """
        Fit the draw correction from physical predictions made elsewhere.

        Split out so rho can be fitted on OUT-OF-FOLD predictions over the
        whole training window rather than on one held-out season. Both are
        honest -- neither has seen the match it is predicting -- but the
        out-of-fold version has five times the data behind it.
        """
        if not self.use_dixon_coles:
            return self
        y = df["target"].astype(int).to_numpy()
        h = phys["s1_goals_home"].to_numpy()
        a = phys["s1_goals_away"].to_numpy()
        self.rho = fit_dixon_coles_rho(h, a, y)
        self.rho_by_div = self._fit_rho_by_division(df, phys, y)
        return self

    def _fit_rho_by_division(self, val_df: pd.DataFrame, lam: pd.DataFrame,
                             y: np.ndarray, prior_weight: int = 300) -> dict:
        """
        A draw correction per league, pulled toward the global one.

        How often matches end level differs by league -- a defensive Greek
        division and a high-scoring German one do not share a number, and one
        global rho splits the difference badly for both. But a single league's
        validation season is small, so each league's own estimate is shrunk
        toward the global value in proportion to how little of it there is:
        prior_weight matches of doubt against however many the league brings.
        """
        if "Div" not in val_df.columns:
            return {}
        out = {}
        h = lam["s1_goals_home"].to_numpy()
        a = lam["s1_goals_away"].to_numpy()
        for div, idx in val_df.groupby("Div").indices.items():
            if len(idx) < 100:                      # too few to say anything
                continue
            local = fit_dixon_coles_rho(h[idx], a[idx], y[idx])
            weight = len(idx) / (len(idx) + prior_weight)
            out[div] = weight * local + (1 - weight) * self.rho
        return out

    def predict_physical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicted goals, shots and shots on target, one column each."""
        cols = self._cols(df)
        return pd.DataFrame(
            {f"s1_{name}": model.predict(df[cols])
             for name, model in self.models.items()},
            index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Market-blind P(H, D, A). This is the estimate that gets corrected."""
        return self.probs_from_physical(self.predict_physical(df), df)

    def probs_from_physical(self, phys: pd.DataFrame,
                            df: pd.DataFrame) -> np.ndarray:
        """
        The scoreline layer, applied to physical predictions made elsewhere.

        Split out so out-of-fold predictions over the training window can be
        turned into probabilities with the same per-league draw correction the
        test predictions get.
        """
        h = phys["s1_goals_home"].to_numpy()
        a = phys["s1_goals_away"].to_numpy()
        if not self.rho_by_div or "Div" not in df.columns:
            return outcome_probs(h, a, self.rho)
        out = np.empty((len(df), 3))
        for div, idx in df.groupby("Div").indices.items():
            out[idx] = outcome_probs(h[idx], a[idx],
                                     self.rho_by_div.get(div, self.rho))
        return out

    def fit_predict_oof(self, train_df: pd.DataFrame, cv: int = 3) -> pd.DataFrame:
        """
        Out-of-fold physical predictions, for stage 2 to train on.

        WHY OUT-OF-FOLD. Given stage 1's in-sample predictions, stage 2 would
        learn to over-trust them, because they look far better than stage 1 will
        manage on data it has not seen. This is the detail that decides whether
        the architecture works at all.

        Plain KFold, so within the training window a fold can be predicted from
        later matches. Everything here is still strictly earlier than the test
        unit, so it does not leak across the seam.
        """
        cols = self._train_cols(train_df)
        out = {}
        for name, target_col in self._usable_targets(train_df).items():
            ok = train_df[target_col].notna()
            preds = pd.Series(np.nan, index=train_df.index)
            preds.loc[ok] = cross_val_predict(
                self._new_regressor(),
                train_df.loc[ok, cols], train_df.loc[ok, target_col], cv=cv)
            out[f"s1_{name}"] = preds
        return pd.DataFrame(out, index=train_df.index)


# =========================================================================
# PART 6 -- STAGE 2: THE MARKET & EDGE MODEL
# =========================================================================


@dataclass
class MarketModel:
    """
    Stage 2. Stage 1's physical view, plus what the market is doing.

    TWO TARGETS, BOTH RETURNING A PROBABILITY.

    "probability" fits P(H, D, A) with an isotonic-calibrated classifier.
    Mispricing is then read off afterwards as p_model - p_ref.

    "deviation" fits the mispricing DIRECTLY: one regressor per outcome, trained
    on (onehot - p_ref). That target is unbiased for the true mispricing, since
    E[onehot] = p_true, so E[onehot - p_ref] is exactly p_true - p_ref.

    THE RECONSTRUCTION IS WHAT MAKES "deviation" USABLE. A raw mispricing
    prediction is not a probability, so log loss, Brier, RPS and ECE would all
    stop applying and the market baseline would have nothing to compare against.
    So the predicted deviation is added back to the market price and
    renormalised, which returns a probability while still having fitted the
    mispricing. The two targets are then scored by identical metrics.

    What to expect: "deviation" should be better calibrated near the market and
    worse in the tails, because it is anchored to p_ref by construction. If it
    produces a much lower w, that is the anchoring showing up rather than a
    better independent view -- which is why stage 2 is not the arm that gets
    shrunk either way.
    """

    seed: int = 42
    target: str = "probability"
    model: object = None
    market_features: list = None

    def _market_cols(self, df: pd.DataFrame) -> list:
        feats = self.market_features or MARKET_FEATURES
        return [c for c in feats if c in df.columns]

    def _inputs(self, df: pd.DataFrame, s1_physical: pd.DataFrame) -> np.ndarray:
        market = np.nan_to_num(df[self._market_cols(df)].to_numpy(float))
        return np.hstack([np.nan_to_num(s1_physical.to_numpy(float)), market])

    def fit(self, train_df: pd.DataFrame, s1_oof: pd.DataFrame) -> "MarketModel":
        X = self._inputs(train_df, s1_oof)
        y = train_df["target"].astype(int).to_numpy()

        if self.target == "probability":
            self.model = CalibratedClassifierCV(
                estimator=make_classifier(self.seed), method="isotonic", cv=3)
            self.model.fit(X, y)

        elif self.target == "deviation":
            onehot = np.zeros((len(y), 3))
            onehot[np.arange(len(y)), y] = 1.0
            residual = onehot - probs(train_df, "p_ref")
            self.model = [make_regressor(self.seed).fit(X, residual[:, k])
                          for k in range(3)]
        else:
            raise ValueError(f"unknown target {self.target!r}; "
                             "use 'probability' or 'deviation'")
        return self

    def predict_proba(self, df: pd.DataFrame,
                      s1_physical: pd.DataFrame) -> np.ndarray:
        X = self._inputs(df, s1_physical)
        if self.target == "probability":
            return self.model.predict_proba(X)

        deviation = np.column_stack([m.predict(X) for m in self.model])
        p = np.clip(probs(df, "p_ref") + deviation, 1e-6, 1.0)
        return p / p.sum(axis=1, keepdims=True)

    def predict_mispricing(self, df: pd.DataFrame,
                           s1_physical: pd.DataFrame) -> np.ndarray:
        """How far this model thinks the market price is wrong, per outcome."""
        return self.predict_proba(df, s1_physical) - probs(df, "p_ref")


# =========================================================================
# PART 7 -- AUXILIARY HEADS
# =========================================================================


@dataclass
class MarketReactionModel:
    """
    Predicts how the price MOVES between opening and closing.

    WHAT IT MAY SEE, and why the restriction is the whole design. The target is
    drift = log(Avg) - log(AvgC), the move from opening to closing. The closing
    price is therefore the answer, so this model gets the OPENING price and
    stage 1's physical view and nothing else. Let AvgC in and it scores
    perfectly while predicting nothing.

    WHAT THIS ANSWERS. "Given only the opening price and a market-blind view of
    the teams, can we tell which way the price will move?" A legitimate,
    leak-free question you could act on when the opening price appears.

    WHAT TO EXPECT: very little, since the market absorbs form quickly. A
    near-zero correlation is the expected result and worth reporting as one.

    WHY IT FEEDS NOTHING. Its output deliberately does not reach stage 2 or the
    selection rule. "Will this price move" is a different question from "is this
    price wrong", and mixing them would make the threshold mean two things.
    """

    seed: int = 42
    models: dict = field(default_factory=dict)

    def _inputs(self, df: pd.DataFrame, s1_physical: pd.DataFrame) -> np.ndarray:
        opening = [c for c in OPENING_COLS if c in df.columns]
        return np.hstack([np.nan_to_num(s1_physical.to_numpy(float)),
                          np.nan_to_num(df[opening].to_numpy(float))])

    def fit(self, train_df: pd.DataFrame,
            s1_oof: pd.DataFrame) -> "MarketReactionModel":
        X = self._inputs(train_df, s1_oof)
        for col in DRIFT_COLS:
            if col not in train_df.columns:
                continue
            ok = train_df[col].notna().to_numpy()
            if ok.sum() < 100:
                continue
            self.models[col] = make_regressor(self.seed).fit(
                X[ok], train_df.loc[ok, col])
        return self

    def predict(self, df: pd.DataFrame, s1_physical: pd.DataFrame) -> pd.DataFrame:
        X = self._inputs(df, s1_physical)
        return pd.DataFrame({f"pred_{col}": model.predict(X)
                             for col, model in self.models.items()},
                            index=df.index)

    def score(self, df: pd.DataFrame, s1_physical: pd.DataFrame) -> dict:
        """Correlation between predicted and realised drift, per outcome."""
        pred = self.predict(df, s1_physical)
        out = {}
        for col in self.models:
            realised = df[col].to_numpy(float)
            predicted = pred[f"pred_{col}"].to_numpy(float)
            ok = np.isfinite(realised) & np.isfinite(predicted)
            out[f"corr_{col}"] = (
                float(np.corrcoef(predicted[ok], realised[ok])[0, 1])
                if ok.sum() > 2 else np.nan)
        return out


@dataclass
class StakeModel:
    """
    Learned stake sizing, applied to bets that have ALREADY been selected.

    WHERE THIS SITS, and why it is here rather than inside the model. A network
    trained to choose stakes from raw matches would have learned which bets to
    place -- selection would move inside the model, there would be no separable
    threshold, and the corrected and uncorrected risk-coverage curves could not
    be produced from one set of predictions. So this never sees an unselected
    match. It takes the output of select_bets() and sizes it, which is the same
    slot stake_rule="quarter_kelly" already occupies. Selection is untouched and
    the headline figure is unaffected.

    WHAT IT LEARNS. Profit per unit staked, regressed on properties of the bet
    itself. Fitted on the VALIDATION fold's bets only -- fitting on test bets
    would be choosing stakes with knowledge of the results they produce.

    HOW THE OUTPUT IS SCALED. Predicted returns are clipped at zero, then
    rescaled to average one unit, so total exposure matches flat staking and the
    comparison isolates ALLOCATION rather than leverage. Without that a rule
    could look better purely by betting more.

    WHAT TO EXPECT, and the warning that goes with it. Sizing by a learned edge
    lets a bad estimate do damage twice -- once by getting the bet selected and
    again by sizing it large. If learned staking beats flat, check it is not
    simply concentrating stake on longer odds before reporting it.
    """

    seed: int = 42
    max_stake: float = 3.0
    model: object = None
    features: tuple = STAKE_FEATURES

    def _X(self, bets: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.features if c in bets.columns]
        return np.nan_to_num(bets[cols].to_numpy(float))

    def fit(self, val_bets: pd.DataFrame) -> "StakeModel":
        if len(val_bets) < 50:
            raise ValueError(f"only {len(val_bets)} validation bets; "
                             "too few to fit a stake rule on")
        stake = val_bets["Stake"].replace(0, np.nan)
        y = (val_bets["PnL"] / stake).to_numpy(float)
        ok = np.isfinite(y)
        self.model = make_regressor(self.seed).fit(self._X(val_bets)[ok], y[ok])
        return self

    def stakes(self, bets: pd.DataFrame) -> np.ndarray:
        raw = np.clip(self.model.predict(self._X(bets)), 0.0, None)
        mean = raw.mean()
        if not np.isfinite(mean) or mean <= 0:
            return np.ones(len(bets))          # model sees no value anywhere
        return np.clip(raw / mean, 0.0, self.max_stake)


def apply_learned_stakes(bets: pd.DataFrame, stake_model: StakeModel) -> pd.DataFrame:
    """
    Re-stake an already-selected bet frame and re-settle it.

    Settles at the best price and again at the average price, exactly as
    select_bets() does, so bootstrap_roi() works on the result unchanged.
    """
    out = bets.copy()
    out["Stake"] = stake_model.stakes(out)

    def settle(odds):
        return np.where(out["IsWin"] == 1,
                        out["Stake"] * (odds - 1.0), -out["Stake"])

    out["PnL"] = settle(out["Odds"])
    out["PnL_Avg"] = np.where(
        out["Odds_Avg"].notna() & (out["Odds_Avg"] > 1.0),
        settle(out["Odds_Avg"]), np.nan)
    return out


# =========================================================================
# PART 8 -- SHRINKAGE: THE CORRECTION
#
# The winner's curse happens because our estimation error is not the same size
# for every match. Selection picks the extreme tail of our estimate, and that
# tail fills up with the matches where our error was biggest, not where our edge
# was biggest. Correcting it means pulling the estimate back toward the market
# BEFORE the threshold sees it.
# =========================================================================


def geometric_blend(p_ref: np.ndarray, p_model: np.ndarray, w) -> np.ndarray:
    """
    The correction itself: pull the model's probabilities toward the market's.

        p_shrunk  proportional to  p_ref^(1-w) * p_model^w

    w = 0 ignores the model and uses the market; w = 1 ignores the market and
    uses the model; in between blends them. `w` may be a scalar (one weight for
    every match) or one weight per match.

    WHY THE BLEND IS GEOMETRIC -- a weighted average of the logs rather than of
    the probabilities. It keeps the result positive, behaves sensibly near zero
    and one, and corresponds to averaging odds rather than probabilities.

    All three shrinkage variants go through here, so the correction is defined
    once and the three differ only in where w comes from.
    """
    eps = 1e-12
    p_ref_c = np.clip(p_ref, eps, 1.0 - eps)
    p_model_c = np.clip(p_model, eps, 1.0 - eps)

    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = w[:, None]

    log_p = (1.0 - w) * np.log(p_ref_c) + w * np.log(p_model_c)
    # Subtract the row maximum before exponentiating so nothing overflows.
    p = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


def apply_static_shrinkage(p_ref: np.ndarray, p_model: np.ndarray,
                           w: float) -> np.ndarray:
    """
    Shrink by one weight, the same for every match.

    Sanity check worth applying: a fitted w above one half means the model is
    being trusted more than the market. Investigate rather than carry on.
    """
    return geometric_blend(p_ref, p_model, np.clip(w, 0.0, 1.0))


def selection_band(p_ref: np.ndarray, p_model: np.ndarray,
                   coverage: float = 0.10) -> np.ndarray:
    """
    The matches a selective strategy would actually consider.

    Where the model disagrees most with the market -- which is exactly where
    the winner's curse lives, and the only place the correction has any work
    to do. The band is defined by the UNCORRECTED disagreement, so it does not
    move as the weight being fitted moves; otherwise the fit would chase its
    own selection around.
    """
    deviation = np.abs(np.asarray(p_model) - np.asarray(p_ref)).max(axis=1)
    return deviation >= np.quantile(deviation, 1.0 - coverage)


def apply_linear_shrinkage(p_ref: np.ndarray, p_model: np.ndarray,
                           c: float) -> np.ndarray:
    """
    James-Stein style: keep a fraction of the deviation from the market.

        p = p_ref + c * (p_model - p_ref)

    The geometric blend shrinks in log-odds space, which pulls hardest on the
    extremes; this one shrinks the deviation itself, evenly. They disagree most
    about long shots, which is where the bets are, so the two are worth
    carrying side by side rather than assuming the first form is right.
    """
    p = np.asarray(p_ref) + c * (np.asarray(p_model) - np.asarray(p_ref))
    p = np.clip(p, 1e-6, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def fit_linear_shrinkage(p_ref: np.ndarray, p_model: np.ndarray,
                         y: np.ndarray, mask: np.ndarray = None) -> float:
    """The James-Stein coefficient, fitted the same way as the static weight."""
    if mask is not None:
        p_ref, p_model, y = p_ref[mask], p_model[mask], y[mask]

    def objective(c):
        return log_loss(y, apply_linear_shrinkage(p_ref, p_model, c),
                        labels=[0, 1, 2])

    return float(minimize_scalar(objective, bounds=(0.0, 1.0),
                                 method="bounded").x)


def fit_static_shrinkage(p_ref_val: np.ndarray, p_model_val: np.ndarray,
                         y_val: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Find the weight that predicts validation results best.

    WHY VALIDATION DATA AND NEVER TRAINING DATA. On training data the model
    looks better than it is, so the weight would come out too high and the
    correction too weak.

    Refitted once per fold rather than once globally, so the weight is allowed
    to differ over time rather than being assumed constant. Whether it moves is
    itself one of the questions worth answering.

    WHERE IT IS FITTED MATTERS AS MUCH AS ON WHAT. Averaged over every match,
    log loss is dominated by the ones no strategy would ever bet, and the
    weight that suits those is not the weight that suits the handful the
    strategy actually acts on. Pass `mask` to fit on the selection band alone.
    """
    if mask is not None:
        p_ref_val, p_model_val, y_val = (p_ref_val[mask], p_model_val[mask],
                                         y_val[mask])

    def objective(w):
        return log_loss(y_val, apply_static_shrinkage(p_ref_val, p_model_val, w),
                        labels=[0, 1, 2])

    return float(minimize_scalar(objective, bounds=(0.0, 1.0),
                                 method="bounded").x)


def calibrate_probabilities(p_fit: np.ndarray, y_fit: np.ndarray,
                            p_apply: np.ndarray) -> np.ndarray:
    """
    Recalibrate stage 1 against its own record, with no market price involved.

    THE RIVAL EXPLANATION. Shrinkage fixes overconfidence using the market's
    opinion of each individual match. Plain calibration fixes overconfidence
    using the model's own track record -- one curve for every match, fitted on
    held-out data. Both attack the same fault with different information, so
    if this does as well, the per-match market anchor was not earning its
    place and the paper should say so.

    One isotonic curve per outcome, then renormalised. Isotonic cannot reorder
    matches, so the ranking is untouched and only the numbers move -- which is
    still enough to change which bets clear a threshold, and how many.
    """
    from sklearn.isotonic import IsotonicRegression

    out = np.empty_like(np.asarray(p_apply, dtype=float))
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_fit[:, k], (y_fit == k).astype(float))
        out[:, k] = iso.predict(p_apply[:, k])
    out = np.clip(out, 1e-6, 1.0)
    return out / out.sum(axis=1, keepdims=True)


@dataclass
class LearnedShrinkage:
    """
    w(x): the shrinkage weight as a FITTED FUNCTION of the match, not a constant.

    WHY A CONSTANT IS NOT ENOUGH. The winner's curse happens because estimation
    error is not the same size for every match. A single w therefore
    over-shrinks the matches we understand well and under-shrinks the ones we do
    not, which is precisely the heterogeneity the correction is supposed to
    address. The uncertainty head already shows the error IS predictable --
    read uncertainty_corr in the fold diagnostics before deciding whether this
    is worth running.

    THE FORM.  w(x) = sigmoid(theta . x), with x the uncertainty inputs
    standardised. Five parameters for four features plus an intercept. That is
    deliberately tiny: it is fitted on one validation unit of a few hundred
    matches, and anything richer would fit that fold rather than the problem.

    HOW IT DIFFERS FROM shrink_by_uncertainty(). That one also produces a
    per-match weight, but from a hand-written rule -- predict the error, clip it
    at percentiles, invert it. The shape of the mapping was chosen, not
    measured. This fits the mapping directly against the thing we actually care
    about, which is validation log loss of the blended probability.

    WHY IT STARTS AT THE STATIC SOLUTION. theta is initialised with the
    intercept at logit(w_static) and every slope at zero, so the optimiser
    begins exactly at the constant-w answer and descends from there. On
    validation data it therefore cannot do worse than static shrinkage. That is
    a property of the fit, not of the test data -- it can still lose out of
    sample, which is the comparison worth reporting.
    """

    theta: np.ndarray = None
    mean_: np.ndarray = None
    std_: np.ndarray = None
    w_static: float = np.nan

    def _design(self, X: np.ndarray) -> np.ndarray:
        z = (np.asarray(X, float) - self.mean_) / self.std_
        return np.hstack([np.ones((len(z), 1)), z])

    def fit(self, p_ref: np.ndarray, p_model: np.ndarray, X: np.ndarray,
            y: np.ndarray, w_static: float = None) -> "LearnedShrinkage":
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0

        if w_static is None:
            w_static = fit_static_shrinkage(p_ref, p_model, y)
        self.w_static = w_static

        design = self._design(X)
        w0 = float(np.clip(w_static, 1e-6, 1 - 1e-6))
        init = np.zeros(design.shape[1])
        init[0] = np.log(w0 / (1.0 - w0))          # start at the static answer

        def objective(theta):
            w = 1.0 / (1.0 + np.exp(-np.clip(design @ theta, -30, 30)))
            return log_loss(y, geometric_blend(p_ref, p_model, w),
                            labels=[0, 1, 2])

        self.theta = minimize(objective, init, method="L-BFGS-B").x
        return self

    def weights(self, X: np.ndarray) -> np.ndarray:
        """The per-match shrinkage weight."""
        return 1.0 / (1.0 + np.exp(-np.clip(self._design(X) @ self.theta, -30, 30)))

    def apply(self, p_ref: np.ndarray, p_model: np.ndarray,
              X: np.ndarray) -> np.ndarray:
        return geometric_blend(p_ref, p_model, self.weights(X))


def uncertainty_inputs(df: pd.DataFrame, p_model: np.ndarray = None) -> np.ndarray:
    """
    Assemble the inputs to the uncertainty model.

    Every one is about OUR confidence, not about the teams: how much the books
    disagree, how much margin the book is taking, how open the match is on the
    market's own numbers, and how far we have strayed from the market. Rest days
    are a fact about the fixture, not about how wrong we are likely to be, and
    cannot do this job.
    """
    d = df.copy()
    if p_model is not None:
        d["abs_deviation"] = np.abs(p_model - probs(d, "p_ref")).sum(axis=1)
    elif "abs_deviation" not in d:
        d["abs_deviation"] = 0.0

    cols = [c for c in UNCERTAINTY_COLS if c in d.columns]
    return np.nan_to_num(d[cols].to_numpy(dtype=float),
                         nan=0.0, posinf=0.0, neginf=0.0)


def fit_uncertainty_head(df_val: pd.DataFrame, p_model_val: np.ndarray,
                         y_val: np.ndarray, seed: int = 42):
    """
    Train a model to predict our own error, match by match.

    For each validation match, measure how badly the model did on that single
    match, then fit a regressor predicting that from the uncertainty inputs.
    Matches predicted to go badly get shrunk harder toward the market.

    WHY VALIDATION DATA ONLY: on training data the model's error is artificially
    small, so the head would learn a confidence that does not survive contact
    with new data.

    Returns the head and a check. The check reports whether predicted error
    actually tracks realised error -- if it does not, the head learned nothing,
    and the honest move is to keep the simpler version and say so. It is
    IN-SAMPLE, so it flatters the head; fit_fold() reports the same correlation
    on the test unit, and that is the one to read.
    """
    X = uncertainty_inputs(df_val, p_model_val)
    eps = 1e-12
    realised = -np.log(np.clip(p_model_val[np.arange(len(y_val)), y_val], eps, 1.0))

    head = make_regressor(seed)
    head.fit(X, realised)

    pred = head.predict(X)
    r = float(np.corrcoef(pred, realised)[0, 1]) if len(pred) > 2 else np.nan
    return head, {"in_sample_corr": r, "n": len(realised),
                  "mean_realised_loss": float(realised.mean())}


def shrink_by_uncertainty(p_ref: np.ndarray, p_model: np.ndarray,
                          predicted_error: np.ndarray,
                          w_max: float = 1.0) -> np.ndarray:
    """
    Shrink each match by how badly we expect to do on it.

    The same geometric blend as apply_static_shrinkage(), but the weight comes
    from predicted error rather than a single fitted scalar: high (trust the
    model) where predicted error is low, low (trust the market) where it is
    high.

    THE DIRECTION IS ASSERTED because getting it backwards inverts the whole
    correction and would still produce a plausible-looking curve, so the mistake
    would not announce itself.

    The weight is clipped at percentiles rather than the extremes: a few unusual
    matches would otherwise squash every other match into a narrow band and the
    weight would stop discriminating.
    """
    e = np.asarray(predicted_error, dtype=float)
    lo, hi = np.nanpercentile(e, 5), np.nanpercentile(e, 95)
    if not np.isfinite(hi - lo) or hi - lo < 1e-12:
        w = np.full(len(e), 0.5)
    else:
        w = 1.0 - np.clip((e - lo) / (hi - lo), 0.0, 1.0)
    w = np.clip(w * w_max, 0.0, 1.0)

    assert np.corrcoef(w, e)[0, 1] <= 0, \
        "w must fall as predicted error rises -- the correction is inverted"

    return geometric_blend(p_ref, p_model, w)


# =========================================================================
# PART 9 -- WALK-FORWARD FOLDS
# =========================================================================

Fold = namedtuple("Fold", "test_unit val_unit train val test")


def assign_blocks(clean: pd.DataFrame, n_blocks: int = 10) -> pd.Series:
    """
    Cut one season into equal-sized chronological blocks.

    WHY THIS EXISTS. Walk-forward needs a time axis to step along. With many
    seasons that axis is the season. With one season there is no such axis, and
    the alternative -- a single train/test split -- throws away most of the
    data's use as test data, which is the one thing in short supply.

    WHAT THIS DOES NOT FIX, and it must not be claimed to. Blocks are not
    seasons. They are correlated with each other far more than seasons are --
    the same teams, the same squads, the same market regime all the way through.
    So the folds are closer to dependent than the season version's are, and
    pooling them does NOT buy the independent test data that more seasons would.

    What it does buy is honest: more of the season used as test data, several
    estimates of w rather than one, and a visible trajectory of w within the
    season. The interval stays wide. Read it, do not round it away.

    Equal counts, not equal calendar time: the first and last month of a season
    are thin, and calendar blocks would produce folds too small to fit on.
    """
    order = clean["Date"].rank(method="first")
    return pd.qcut(order, n_blocks, labels=False).astype(int)


def make_folds(df: pd.DataFrame, features: list, min_train_units: int = 1,
               min_test_matches: int = 300, split_col: str = None,
               n_blocks: int = 10, verbose: bool = True,
               required: list = None) -> tuple:
    """
    Drop unusable rows, pick the time axis, and cut the data into folds.

    Returns (folds, split_col). Every fold trains on data strictly earlier than
    the unit it validates on, which is strictly earlier than the unit it tests
    on. Nothing ever trains on its own future.

    WHY THIS IS ONE FUNCTION. More TRAINING data does not narrow the confidence
    interval; it may improve the model slightly, but it adds no test data. Only
    more TEST data does. Walking forward turns one test period into many, which
    is the only change here capable of moving the result toward significance.

    Folds too small to mean anything are dropped here rather than downstream. A
    test unit with very few matches returns an interval so wide it cannot be
    told from any other number, and pooling it in adds noise while looking like
    extra evidence.
    """
    # Only the required features must be present. The rest may be null, which
    # the tree models handle natively; demanding every column would throw away
    # every division a newer feature does not reach.
    needed = (list(required) if required is not None else list(features))
    needed = [c for c in needed if c in df.columns]
    needed += ["target"] + outcome_cols("odds_best")
    clean = df[df[needed].notna().all(axis=1)].copy()
    clean["Season"] = clean["Season"].astype(str)
    clean = clean.sort_values("Date").reset_index(drop=True)

    if split_col is None:
        n_seasons = clean["Season"].nunique()
        split_col = "Season" if n_seasons >= min_train_units + 2 else "Block"
        if split_col == "Block" and verbose:
            warnings.warn(
                f"Only {n_seasons} season(s) present, so walk-forward is "
                f"stepping along {n_blocks} chronological blocks within the "
                "season rather than across seasons. Blocks share teams, squads "
                "and market regime, so the folds are not independent the way "
                "seasons are. Treat every interval as a lower bound on the "
                "true uncertainty.")
    if split_col == "Block":
        clean["Block"] = assign_blocks(clean, n_blocks)

    units = sorted(clean[split_col].unique())
    if len(units) < min_train_units + 2:
        raise ValueError(
            f"Walk-forward needs at least {min_train_units + 2} units along "
            f"'{split_col}'; the data has {len(units)}: {units}.")

    folds = []
    for i in range(min_train_units + 1, len(units)):
        tr = clean[clean[split_col].isin(units[:i - 1])]
        va = clean[clean[split_col] == units[i - 1]]
        te = clean[clean[split_col] == units[i]].copy()
        # Seasons are labels, not dates, and two of them can overlap: 2019/20
        # ran to 7 August 2020 because of the covid shutdown, while 2020/21
        # kicked off on the 1st. Six matches, one fold, no test season touched
        # -- but the rule is that nothing trains on its own future, so the
        # tail is cut rather than explained away.
        if len(va) and len(te):
            tr = tr[tr["Date"] < va["Date"].min()]
            va = va[va["Date"] < te["Date"].min()]
        if len(tr) < 400 or len(va) < 100 or len(te) < min_test_matches:
            if verbose:
                print(f"  skipping fold {units[i]}: train {len(tr)}, "
                      f"val {len(va)}, test {len(te)} -- too small")
            continue
        folds.append(Fold(units[i], units[i - 1], tr, va, te))

    if not folds:
        raise ValueError(
            f"No fold along '{split_col}' had enough data. Lower n_blocks or "
            "min_test_matches, or check how many rows survive the feature "
            "completeness filter.")
    return folds, split_col


# =========================================================================
# PART 10 -- SELECTION
# =========================================================================


def bet_score(preds_df: pd.DataFrame, prefix: str, outcome: str,
              criterion: str = "ev") -> pd.Series:
    """
    The quantity the threshold is applied to, for one outcome.

    WHY THIS IS ONE FUNCTION. Four places need this number -- the selection rule
    itself, the risk-coverage sweep, the ROI table and the non-vacuousness check
    -- and if any computed it differently the threshold axis would quietly mean
    different things in different tables while every one still printed.

    THE TWO CRITERIA ARE NOT INTERCHANGEABLE.

    "ev" multiplies the probability by the odds, so the odds sit inside the
    score. Raising the threshold then selects longer and longer prices whatever
    the model thinks, and a model with no opinion at all shows the same falling
    return -- which is the favourite-longshot bias, not selection acting on
    estimation error.

    "edge" is how far our probability sits above the market's. No odds in it, so
    a tighter threshold means more disagreement rather than longer prices. That
    is what the claim is about. Sweep both before believing either.
    """
    p = preds_df[f"{prefix}_{outcome}"]
    if criterion == "ev":
        return p * preds_df[f"odds_best_{outcome}"] - 1.0
    if criterion == "edge":
        return p - preds_df[f"p_ref_{outcome}"]
    raise ValueError(f"unknown criterion {criterion!r}")


def score_pool(preds_df: pd.DataFrame, prefix: str, allowed=("H", "A"),
               criterion: str = "ev") -> pd.Series:
    """
    Every outcome's score in one series, for choosing thresholds by quantile.

    Picking thresholds from the scores that actually occur is what makes the
    curve span real coverage rather than an arbitrary range, and what lets two
    arms on different scales be compared at matched coverage.
    """
    vals = [bet_score(preds_df, prefix, o, criterion).dropna()
            for o in allowed if f"odds_best_{o}" in preds_df]
    return pd.concat(vals) if vals else pd.Series([0.0])


def count_candidates(preds_df: pd.DataFrame, allowed=("H", "A"),
                     max_odds: float = None) -> int:
    """
    How many bets COULD have been placed: the denominator of coverage.

    Without it the curve can only show bet counts, and a bet count says nothing
    about how selective the rule is being.
    """
    n = 0
    for o in allowed:
        odds = preds_df.get(f"odds_best_{o}")
        if odds is None:
            continue
        ok = odds.notna() & (odds > 1.0)
        if max_odds is not None:
            ok &= odds <= max_odds
        n += int(ok.sum())
    return n


def select_bets(preds_df: pd.DataFrame, tau: float,
                prob_prefix: str = "p_corr", allowed=("H", "A"),
                max_odds: float = None, stake_rule: str = "flat",
                kelly_fraction: float = 0.25, max_stake: float = 3.0,
                criterion: str = "ev") -> pd.DataFrame:
    """
    THE SELECTION RULE. Everything else calls this.

    Given probability estimates and a threshold, return the bets that clear it.
    In selective-prediction terms, we act on these and abstain on everything
    else.

    Bets are priced at the best available price and also settled at the average
    price, so the gain from simply shopping around can be reported separately
    from any model edge.

    ONE THRESHOLD, NOT TWO. Applying a threshold to expected value AND a second
    one to the probability edge makes the coverage axis meaningless, because you
    can no longer say what a given threshold selected.

    NO ODDS CAP BY DEFAULT. Long odds are exactly where estimation error is
    largest and where the winner's curse should bite hardest, so capping the
    price cuts off the thing being studied. If you reintroduce a cap, sweep it
    and report the sensitivity rather than fixing it silently.

    FLAT STAKES BY DEFAULT. Sizing by estimated edge lets a bad estimate do
    damage twice -- once by getting the bet selected and again by sizing it
    large. The claim is about selection, so the default isolates it. Pass
    stake_rule="quarter_kelly", or use StakeModel, to vary it.
    """
    frames = []
    for o in allowed:
        odds = preds_df.get(f"odds_best_{o}")
        if odds is None:
            continue
        p_est = preds_df[f"{prob_prefix}_{o}"]

        ok = odds.notna() & (odds > 1.0) & p_est.notna()
        if max_odds is not None:
            ok &= odds <= max_odds

        ev = p_est * odds - 1.0
        score = bet_score(preds_df, prob_prefix, o, criterion)

        chosen = ok & (score >= tau)
        if not chosen.any():
            continue

        sub = preds_df.loc[chosen]
        odds_ref = sub.get(f"odds_avg_{o}", pd.Series(np.nan, index=sub.index))
        frames.append(pd.DataFrame({
            "match_key": sub["match_key"].values,
            "Date": sub["Date"].values,
            "Season": sub.get("Season", pd.Series("", index=sub.index)).values,
            "Div": sub.get("Div", pd.Series("", index=sub.index)).values,
            "Selection": o,
            "Odds": odds[chosen].values,
            "Odds_Avg": odds_ref.values,
            "p_est": p_est[chosen].values,
            "p_ref": sub[f"p_ref_{o}"].values,
            "EV": ev[chosen].values,
            "Score": score[chosen].values,
            "IsWin": (sub["target"].values == TARGET_MAP[o]).astype(int),
        }))

    if not frames:
        return pd.DataFrame(columns=BET_COLS)

    bets = pd.concat(frames, ignore_index=True)

    if stake_rule == "flat":
        bets["Stake"] = 1.0
    elif stake_rule == "quarter_kelly":
        full_kelly = bets["EV"] / (bets["Odds"] - 1.0)
        bets["Stake"] = (full_kelly * kelly_fraction * 100).clip(0.0, max_stake)
    else:
        raise ValueError(f"unknown stake_rule {stake_rule!r}")

    def settle(odds):
        return np.where(bets["IsWin"] == 1,
                        bets["Stake"] * (odds - 1.0), -bets["Stake"])

    bets["PnL"] = settle(bets["Odds"])
    bets["PnL_Avg"] = np.where(
        bets["Odds_Avg"].notna() & (bets["Odds_Avg"] > 1.0),
        settle(bets["Odds_Avg"]), np.nan)
    return bets


# =========================================================================
# PART 11 -- EVALUATION
# =========================================================================


def bootstrap_roi(bets_df: pd.DataFrame, n_boot: int = 10000,
                  alpha: float = 0.05, seed: int = 42,
                  pnl_col: str = "PnL") -> dict:
    """
    Return on investment with a confidence interval.

    WHY EVERY REPORTED RETURN GOES THROUGH HERE: a return without an interval is
    not a result. It cannot be compared with another return, and it cannot be
    distinguished from zero.

    WHY IT RESAMPLES MATCHES, NOT BETS. Two bets on the same match share an
    outcome, so they are correlated. Resampling individual bets pretends they
    are independent and produces an interval that is too narrow -- an error that
    is dangerous precisely because a too-narrow interval looks confident rather
    than obviously broken.
    """
    if len(bets_df) == 0:
        return {"roi": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_le_zero": np.nan, "n_bets": 0, "n_matches": 0, "dropped": 0}

    rng = np.random.default_rng(seed)

    # Total stake and profit per MATCH first, then resample whole matches.
    g = bets_df.groupby("match_key")
    stake_g = g["Stake"].sum().to_numpy()
    pnl_g = g[pnl_col].sum().to_numpy()
    keep = np.isfinite(stake_g) & np.isfinite(pnl_g)
    stake_g, pnl_g = stake_g[keep], pnl_g[keep]
    n_matches = len(stake_g)

    roi = float(pnl_g.sum() / stake_g.sum() * 100) if stake_g.sum() > 0 else np.nan

    idx = rng.integers(0, n_matches, size=(n_boot, n_matches))
    boot_stake = stake_g[idx].sum(axis=1)
    boot_pnl = pnl_g[idx].sum(axis=1)

    ok = boot_stake > 0
    boot_roi = boot_pnl[ok] / boot_stake[ok] * 100

    return {
        "roi": roi,
        "ci_low": float(np.percentile(boot_roi, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot_roi, 100 * (1 - alpha / 2))),
        "p_le_zero": float(np.mean(boot_roi <= 0)),
        "n_bets": int(len(bets_df)),
        "n_matches": n_matches,
        "dropped": int((~ok).sum()),
    }


def report_roi(bets_df: pd.DataFrame, label: str = "",
               n_boot: int = 10000) -> dict:
    """
    Print a return with its interval and the line-shopping gain.

    WHY THE SECOND NUMBER: the gap between settling at the best price and at the
    average price is mechanical. It is what shopping around is worth to somebody
    with no model at all, so it must never be presented as model edge. Printing
    both makes that impossible to confuse.
    """
    b = bootstrap_roi(bets_df, n_boot=n_boot)
    print(f"\n{label}")
    print(f"  bets {b['n_bets']:,} over {b['n_matches']:,} matches")
    print(f"  ROI  {b['roi']:+.2f}%   95% CI [{b['ci_low']:+.2f}%, "
          f"{b['ci_high']:+.2f}%]   P(ROI<=0) = {b['p_le_zero']:.3f}")

    if len(bets_df) and bets_df["PnL_Avg"].notna().any():
        ref = bootstrap_roi(bets_df.dropna(subset=["PnL_Avg"]),
                            n_boot=n_boot, pnl_col="PnL_Avg")
        print(f"  same bets at the average price: {ref['roi']:+.2f}%")
        print(f"  line-shopping gain: {b['roi'] - ref['roi']:+.2f} points "
              f"(mechanical, not model edge)")
    return b


def scoring_report(y_true: np.ndarray, p: np.ndarray, label: str = "") -> dict:
    """
    Every scoring metric that matters, in one row.

    WHY ACCURACY IS EXCLUDED FROM RANKING: a model can be more accurate and
    worse calibrated at the same time. We bet on the probabilities themselves,
    not on which outcome is most likely, so calibration is what matters and
    accuracy can actively mislead. It is reported because refusing to print it
    reads as hiding something, and named so it cannot be quoted as a ranking.

    WHY THE RANKED PROBABILITY SCORE IS INCLUDED: home, draw and away are
    ordered, so being wrong by a lot should cost more than being wrong by a
    little, and plain log loss does not know that.

    Use the de-vigged market as the baseline. A dummy model returning the market
    price should reproduce the market's own score exactly, which is the sanity
    check for this whole harness.
    """
    y_true = np.asarray(y_true, int)
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    onehot = np.zeros_like(p)
    onehot[np.arange(len(y_true)), y_true] = 1.0

    rps = float(np.mean(np.sum((np.cumsum(p, axis=1) - np.cumsum(onehot, axis=1)) ** 2,
                               axis=1) / (p.shape[1] - 1)))

    # Classwise expected calibration error, over ten probability bins.
    ece = 0.0
    for k in range(p.shape[1]):
        bins = np.clip((p[:, k] * 10).astype(int), 0, 9)
        for b in range(10):
            m = bins == b
            if m.sum():
                gap = abs(p[m, k].mean() - onehot[m, k].mean())
                ece += gap * m.sum() / (len(p) * p.shape[1])

    return {
        "label": label,
        "log_loss": float(log_loss(y_true, p, labels=[0, 1, 2])),
        "rps": rps,
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "ece": float(ece),
        "accuracy_do_not_rank_on_this": float(np.mean(p.argmax(axis=1) == y_true)),
    }


def accuracy_report(y_true: np.ndarray, p: np.ndarray,
                    label: str = "") -> pd.DataFrame:
    """
    Accuracy broken out per class, with the baselines that make it readable.

    WHY IT IS REPORTED AND STILL NOT RANKED ON. Accuracy is the number everybody
    asks for first, so refusing to print it reads as hiding something. Printing
    it with its baselines makes it interpretable and makes its uselessness here
    visible at the same time. The ALL row's base_rate is what you score by
    predicting the commonest class every time, with no model at all.

    WHERE ACCURACY ACTIVELY MISLEADS HERE: draws. A draw is almost never the
    single most likely outcome, so argmax nearly never predicts one, and a model
    can score well on accuracy while being systematically wrong about draw
    probability -- exactly the kind of error that loses money on a home or away
    bet priced off that same draw probability. The per-class rows make that
    visible; one accuracy figure hides it.
    """
    y_true = np.asarray(y_true, int)
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    pred = p.argmax(axis=1)

    rows = []
    for k, name in enumerate(OUTCOMES):
        n_pred, n_true = int((pred == k).sum()), int((y_true == k).sum())
        hit = int(((pred == k) & (y_true == k)).sum())
        rows.append({
            "label": label, "class": name,
            "n_actual": n_true, "n_predicted": n_pred,
            "precision": hit / n_pred if n_pred else np.nan,
            "recall": hit / n_true if n_true else np.nan,
            "base_rate": n_true / len(y_true),
            "mean_p": float(p[:, k].mean()),
        })

    rows.append({
        "label": label, "class": "ALL",
        "n_actual": len(y_true), "n_predicted": len(y_true),
        "precision": float((pred == y_true).mean()),
        "recall": float((pred == y_true).mean()),
        "base_rate": float(np.bincount(y_true, minlength=3).max() / len(y_true)),
        "mean_p": np.nan,
    })
    return pd.DataFrame(rows)


def candidate_bets(preds_df: pd.DataFrame, allowed=("H", "A")) -> pd.DataFrame:
    """Every bet that COULD have been placed: one row per outcome per match."""
    frames = []
    for o in allowed:
        odds = preds_df.get(f"odds_best_{o}")
        if odds is None:
            continue
        ok = odds.notna() & (odds > 1.0)
        sub = preds_df.loc[ok]
        frames.append(pd.DataFrame({
            "Odds": sub[f"odds_best_{o}"].to_numpy(float),
            "IsWin": (sub["target"].to_numpy() == OUTCOMES.index(o)).astype(float),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def odds_matched_random(preds_df: pd.DataFrame, bets: pd.DataFrame,
                        n_draws: int = 500, allowed=("H", "A"),
                        seed: int = 42, bins: int = 10) -> dict:
    """
    What a chooser with no skill returns, betting at the same prices.

    WHY NOT PLAIN RANDOM. Returns depend heavily on the odds you bet, because
    the bookmaker's margin is far bigger on long shots. A model that drifts
    toward outsiders as it gets pickier will lose more, and that says nothing
    about its judgement. So the random picks are drawn to match the odds
    profile of the model's own bets, decile by decile: same prices, same
    number of bets, no opinion about which match.

    What is left when the model beats this is selection skill. What is left
    when it does not is the favourite-longshot bias wearing a model's clothes.
    """
    pool = candidate_bets(preds_df, allowed)
    if pool.empty or bets.empty:
        return {}

    edges = np.unique(np.quantile(pool["Odds"], np.linspace(0, 1, bins + 1)))
    pool_bin = np.clip(np.digitize(pool["Odds"], edges[1:-1]), 0, len(edges) - 2)
    bet_bin = np.clip(np.digitize(bets["Odds"], edges[1:-1]), 0, len(edges) - 2)
    by_bin = {b: pool.index[pool_bin == b].to_numpy() for b in np.unique(pool_bin)}
    wanted = pd.Series(bet_bin).value_counts().to_dict()

    rng = np.random.default_rng(seed)
    profit = pool["IsWin"].to_numpy() * (pool["Odds"].to_numpy() - 1.0) \
        - (1.0 - pool["IsWin"].to_numpy())
    draws = np.empty(n_draws)
    for d in range(n_draws):
        # Without replacement where the decile is big enough, which it almost
        # always is; with it only when a thin decile would otherwise run out.
        # Drawing with replacement narrows the spread of the random
        # distribution and flatters whatever is being compared against it.
        picked = [rng.choice(by_bin[b], size=n, replace=len(by_bin[b]) < n)
                  for b, n in wanted.items() if b in by_bin and len(by_bin[b])]
        draws[d] = profit[np.concatenate(picked)].mean() * 100 if picked else np.nan

    model_roi = float((bets["IsWin"] * (bets["Odds"] - 1.0)
                       - (1.0 - bets["IsWin"])).mean() * 100)
    return {"n_bets": len(bets), "model_roi": model_roi,
            "random_mean": float(np.nanmean(draws)),
            "random_lo": float(np.nanpercentile(draws, 5)),
            "random_hi": float(np.nanpercentile(draws, 95)),
            "model_beats_random": float((draws < model_roi).mean())}


def rankable(preds_df: pd.DataFrame, prefix: str, criterion: str,
             allowed=("H", "A")) -> bool:
    """
    Can this arm be ranked by this criterion at all?

    An arm's edge is measured against the market, so the market's own edge is
    zero on every match and an arm that has collapsed onto the market inherits
    that. Ranking those by edge sorts a column of zeros: the threshold selects
    everything, the coverage column lies, and the row still prints. Better to
    leave it out and say why.
    """
    if criterion != "edge":
        return True
    scores = np.abs(score_pool(preds_df, prefix, allowed, criterion))
    # Not "does anything differ" but "does enough differ to rank on". An arm
    # whose weight was zero in four folds out of five has an edge of exactly
    # zero on those seasons, and its "top 10%" would be drawn almost entirely
    # from the one season that moved -- while the coverage column claimed it
    # spanned them all.
    return float((scores > 1e-6).mean()) > 0.5


def random_baseline_table(preds_df: pd.DataFrame, prefixes, coverage: float,
                          criterion: str = "edge", allowed=("H", "A"),
                          n_draws: int = 500) -> pd.DataFrame:
    """One row per arm: its return against odds-matched random picks."""
    rows = []
    for prefix in prefixes:
        if not has_arm(preds_df, prefix) or not rankable(preds_df, prefix, criterion):
            continue
        scores = score_pool(preds_df, prefix, allowed, criterion)
        tau = float(np.quantile(scores, 1.0 - coverage))
        bets = select_bets(preds_df, tau, prefix, allowed, criterion=criterion)
        stats = odds_matched_random(preds_df, bets, n_draws=n_draws,
                                    allowed=allowed)
        if stats:
            rows.append({"arm": prefix, **stats})
    return pd.DataFrame(rows)


def betting_report(preds_df: pd.DataFrame, prefixes=("p_indep", "p_corr"),
                   coverages=(0.50, 0.25, 0.10, 0.05), allowed=("H", "A"),
                   stake_rule: str = "flat", criterion: str = "ev",
                   n_boot: int = 5000) -> pd.DataFrame:
    """
    The ROI table: what each arm returns at matched coverage.

    WHY COVERAGE IS MATCHED RATHER THAN THE THRESHOLD. Corrected and uncorrected
    probabilities live on different scales, so the same tau selects different
    numbers of bets from each and the two rows are not comparable. Each arm's
    threshold is picked from ITS OWN score distribution so both bet on the same
    fraction of matches. A difference in return is then a difference in which
    matches were chosen, which is the claim.

    WHAT EACH COLUMN IS FOR.

      win_rate    the betting analogue of accuracy. Falls as coverage tightens
                  because the surviving bets are longer-priced -- arithmetic,
                  not failure.
      avg_odds    watch alongside win_rate. If it climbs as coverage falls, the
                  rule is selecting longshots, and any change in return may be
                  favourite-longshot bias rather than anything the model did.
                  Use criterion="edge" to take the odds out and check.
      roi, ci     the result. The interval is the result; the middle is not.
      p_le_zero   bootstrap probability the true return is at or below zero.
      roi_avg     the same bets settled at the AVERAGE price.
      shop_gain   roi minus roi_avg: what shopping for the best price is worth
                  to somebody with no model whatsoever. Mechanical, and never to
                  be reported as model edge.

    Read shop_gain first. If it accounts for most of roi, the model is not what
    is making money.
    """
    rows = []
    for prefix in prefixes:
        pooled = score_pool(preds_df, prefix, allowed, criterion)
        for cov in coverages:
            tau = float(np.quantile(pooled, 1.0 - cov))
            bets = select_bets(preds_df, tau, prefix, allowed,
                               stake_rule=stake_rule, criterion=criterion)
            if len(bets) == 0:
                continue
            b = bootstrap_roi(bets, n_boot=n_boot)

            ref = {"roi": np.nan}
            sub = bets.dropna(subset=["PnL_Avg"])
            if len(sub):
                ref = bootstrap_roi(sub, n_boot=n_boot, pnl_col="PnL_Avg")

            rows.append({
                "arm": prefix, "target_cov": cov, "tau": tau,
                "n_bets": b["n_bets"], "n_matches": b["n_matches"],
                "win_rate": float(bets["IsWin"].mean()),
                "avg_odds": float(bets["Odds"].mean()),
                "roi": b["roi"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                "p_le_zero": b["p_le_zero"],
                "roi_avg": ref["roi"],
                "shop_gain": b["roi"] - ref["roi"],
            })
    return pd.DataFrame(rows)


def risk_coverage_curve(preds_df: pd.DataFrame, prob_prefix: str = "p_corr",
                        taus: np.ndarray = None, allowed=("H", "A"),
                        stake_rule: str = "flat", n_boot: int = 2000,
                        min_bets: int = 30, max_odds: float = None,
                        criterion: str = "ev") -> pd.DataFrame:
    """
    THE HEADLINE FIGURE.

    Sweeps the threshold and records, at each value, how many bets were selected
    (coverage) and what they returned (risk), with a bootstrapped interval.

    Coverage is an explicit axis rather than implicit in the bet count. That is
    what makes this a risk-coverage curve and connects it to the selective
    classification literature, where the same curve evaluates classifiers
    allowed to abstain.

    WHAT TO LOOK FOR. Plot two curves, uncorrected and corrected. The uncorrected
    one is predicted to get WORSE as coverage falls, because tightening the
    threshold selects harder for estimation error, so the surviving bets are
    increasingly those whose edge was overestimated. The corrected one is
    predicted to stay flat.

    They must differ in SHAPE, not merely in position along the threshold axis.
    A corrected curve that is the uncorrected one shifted sideways is a
    relabelled axis, not a correction. check_non_vacuousness() tests that
    formally; this is where it becomes visible.

    All three outcomes are reportable: curves differing in shape supports the
    claim; curves coinciding means the correction is vacuous, which is a finding
    about the method rather than a failed project; and two flat curves with wide
    intervals means there is not enough test data yet.
    """
    if taus is None:
        taus = np.quantile(score_pool(preds_df, prob_prefix, allowed, criterion),
                           np.linspace(0.50, 0.995, 25))

    denom = count_candidates(preds_df, allowed, max_odds)
    rows = []
    for tau in taus:
        bets = select_bets(preds_df, tau, prob_prefix, allowed,
                           max_odds=max_odds, stake_rule=stake_rule,
                           criterion=criterion)
        coverage = len(bets) / denom if denom else 0
        if len(bets) < min_bets:
            # Below this the interval is meaningless. Record the point and move
            # on rather than drawing a curve into noise.
            rows.append({"tau": tau, "coverage": coverage, "n_bets": len(bets),
                         "roi": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue
        b = bootstrap_roi(bets, n_boot=n_boot)
        rows.append({"tau": tau, "coverage": coverage, "n_bets": len(bets),
                     "avg_odds": bets["Odds"].mean(),
                     "win_rate": bets["IsWin"].mean(), "roi": b["roi"],
                     "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
    return pd.DataFrame(rows)


def aurc(curve_df: pd.DataFrame) -> float:
    """
    Summarise the whole risk-coverage curve as one number.

    The standard metric in selective prediction, and it removes the temptation
    to quote whichever threshold happened to look best. A metric defined over
    the entire curve makes tuning the threshold on test data impossible.
    """
    d = curve_df.dropna(subset=["roi"]).sort_values("coverage")
    if len(d) < 2:
        return np.nan
    span = d["coverage"].iloc[-1] - d["coverage"].iloc[0]
    if span <= 0:
        return np.nan
    return float(np.trapezoid(d["roi"].values, d["coverage"].values) / span)


def check_non_vacuousness(preds_df: pd.DataFrame,
                          corrected_prefix: str = "p_corr",
                          raw_prefix: str = "p_indep",
                          taus: np.ndarray = None,
                          n_boot: int = 1000,
                          criterion: str = None) -> pd.DataFrame:
    """
    Test whether the correction is real or just a relabelled threshold.

    THE RISK IT GUARDS AGAINST: shrinking and then thresholding might be exactly
    the same as thresholding at a different level and not shrinking at all. If
    so we renamed an axis, and the headline figure is an artefact. This is the
    first thing a sceptical examiner will test, so much better to have tested it
    ourselves.

    HOW IT WORKS: for each threshold used with correction, find the threshold
    that picks the SAME NUMBER of bets without correction, then compare the two
    sets directly. Equal sizes make the comparison fair.

    HOW TO READ IT: a high overlap with intervals sitting on top of each other
    means the correction is vacuous, and that should be said plainly. An overlap
    well below one means the correction REORDERS which matches look attractive,
    and no change of threshold can do that.
    """
    # ON THE SAME CRITERION AS THE CLAIM. This used to rank by expected value
    # while the operating point and the headline rank by edge, so the gate was
    # testing a different selection rule from the one being reported.
    criterion = criterion or OP_CRITERION
    if not rankable(preds_df, corrected_prefix, criterion):
        return pd.DataFrame()          # collapsed onto the market: nothing to test
    if taus is None:
        taus = np.quantile(score_pool(preds_df, corrected_prefix,
                                      criterion=criterion),
                           np.linspace(0.60, 0.99, 12))

    rows = []
    for tau in taus:
        bets_c = select_bets(preds_df, tau, corrected_prefix, criterion=criterion)
        n_target = len(bets_c)
        if n_target < 20:
            continue

        # Bisect so the uncorrected rule picks the same number of bets. Bet
        # count falls as the threshold rises, so bisection is safe here.
        lo, hi = -1.0, 5.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if len(select_bets(preds_df, mid, raw_prefix,
                               criterion=criterion)) > n_target:
                lo = mid
            else:
                hi = mid
        tau_prime = (lo + hi) / 2
        bets_r = select_bets(preds_df, tau_prime, raw_prefix, criterion=criterion)

        id_c = set(zip(bets_c["match_key"], bets_c["Selection"]))
        id_r = set(zip(bets_r["match_key"], bets_r["Selection"]))

        bc = bootstrap_roi(bets_c, n_boot=n_boot)
        br = bootstrap_roi(bets_r, n_boot=n_boot)
        rows.append({
            "tau": tau, "tau_prime": tau_prime,
            "n_corrected": len(bets_c), "n_raw": len(bets_r),
            "overlap": len(id_c & id_r) / len(id_c) if id_c else np.nan,
            "roi_corrected": bc["roi"],
            "ci_corrected": (bc["ci_low"], bc["ci_high"]),
            "roi_raw": br["roi"], "ci_raw": (br["ci_low"], br["ci_high"]),
        })
    return pd.DataFrame(rows)


def era_breakdown(preds_df: pd.DataFrame, folds_df: pd.DataFrame,
                  prob_prefix: str = "p_corr", coverage: float = 0.10,
                  unit_col: str = None, criterion: str = None) -> pd.DataFrame:
    """
    Break the results down by test unit, and track the correction weight.

    WHY NEVER REPORT ONE POOLED FIGURE: the market changes over time, so a single
    number can hide the effect being alive early and dead later. Splitting it is
    the difference between describing a trend and averaging one away.

    Within one season the unit is a block, and what the table shows is drift
    through the season -- whether w and the return move from August to May. That
    is a weaker question than the across-season one and should be read as a
    diagnostic: the blocks are not independent of each other, so a trend across
    them can be one run of results rather than a change in the market.
    """
    if unit_col is None:
        unit_col = "Block" if "Block" in preds_df.columns else "Season"

    criterion = criterion or OP_CRITERION
    # An arm collapsed onto the market has no edge to rank, and its "top 10%"
    # is a sort of rounding residue. Same rule as the operating point.
    if not rankable(preds_df, prob_prefix, criterion):
        return pd.DataFrame()
    # Where the fold diagnostics carry a threshold read off VALIDATION for this
    # arm, use it, so a test unit's own scores never choose its bets.
    tau_col = {"p_corr": "op_tau", "p_indep": "op_tau_indep"}.get(prob_prefix)
    if criterion != OP_CRITERION or tau_col not in folds_df.columns:
        tau_col = None
    rows = []
    for unit, part in preds_df.groupby(preds_df[unit_col]):
        # BY COVERAGE, AND ON THE SAME CRITERION AS EVERYTHING ELSE. This was
        # a fixed "expected value of at least 2%", which is a number in EV
        # units: under edge ranking it selected nothing useful, and when the
        # weight collapses to zero it scored the market's own picks -- so two
        # different models printed byte-identical rows here.
        fold_tau = (folds_df.loc[folds_df["test_unit"] == unit, tau_col]
                    if tau_col else pd.Series(dtype=float))
        tau = (float(fold_tau.iloc[0]) if len(fold_tau) else
               float(np.quantile(score_pool(part, prob_prefix, criterion=criterion),
                                 1.0 - coverage)))
        bets = select_bets(part, tau, prob_prefix, criterion=criterion)
        b = bootstrap_roi(bets, n_boot=2000)
        w = folds_df.loc[folds_df["test_unit"] == unit, "w"]
        rows.append({
            unit_col.lower(): unit,
            "from": part["Date"].min().date(), "to": part["Date"].max().date(),
            "w": float(w.iloc[0]) if len(w) else np.nan,
            "n_bets": b["n_bets"], "roi": b["roi"],
            "ci_low": b["ci_low"], "ci_high": b["ci_high"],
        })
    return pd.DataFrame(rows)


def arms_present(preds_df: pd.DataFrame) -> list:
    """
    The (prefix, label) pairs to report on, in the order they should be read.

    Market first, because it is the number everything else has to beat. One list
    drives every table, so the scoring, accuracy and ROI reports can never end
    up describing different sets of arms.
    """
    pairs = [("p_ref", "market")]
    if has_arm(preds_df, "p_indep"):
        pairs.append(("p_indep", "stage 1 (market-blind)"))
    pairs += [("p_model", "stage 2 (market-aware)"), ("p_corr", "corrected")]
    if has_arm(preds_df, "p_band"):
        pairs.append(("p_band", "corrected, w fitted on the bet band"))
    if has_arm(preds_df, "p_js"):
        pairs.append(("p_js", "corrected, James-Stein"))
    if has_arm(preds_df, "p_cal"):
        pairs.append(("p_cal", "calibrated, no market anchor"))
    if has_arm(preds_df, "p_unc"):
        pairs.append(("p_unc", "per-match w, hand-written"))
    if has_arm(preds_df, "p_wfit"):
        pairs.append(("p_wfit", "per-match w(x), fitted"))
    return pairs


# =========================================================================
# PART 12 -- ORCHESTRATION
# =========================================================================


def fit_fold(fold, seed: int = 42, stage2_target: str = "probability",
             use_dixon_coles: bool = True, use_learned_uncertainty: bool = True,
             learn_w: bool = True, learn_stakes: bool = True,
             stake_coverage: float = 0.25, op_coverage: float = 0.10) -> dict:
    """
    Fit every stage on one walk-forward fold and apply the correction.

    ORDER OF OPERATIONS, which nothing may rearrange:
        stage 1 -> scoreline -> deviate from market -> shrink -> SELECT -> stake

    ONE STAGE 1, NOT TWO. Everything downstream -- rho, the shrinkage weight,
    the calibration curve, the uncertainty head, w(x), stage 2, the operating
    threshold -- is fitted against out-of-fold predictions over the whole
    training window, and the models that face the test season are fitted on
    that same window. An earlier version fitted those pieces against a model
    trained on part of the window and then applied them to a model trained on
    all of it: two different stage 1s mixed into one row of results.

    The window is train PLUS validation. Nothing here has seen the test
    season, and nothing is fitted on predictions of matches its own model was
    trained on.
    """
    tr, va, te = fold.train, fold.val, fold.test.copy()
    window = pd.concat([tr, va], ignore_index=True)

    # --- stage 1, market-blind -------------------------------------------
    engine = MatchEngine(seed=seed, use_dixon_coles=use_dixon_coles)
    s1_oof = engine.fit_predict_oof(window)   # honest across the whole window
    engine.fit(window)                        # the model the test season meets
    engine.fit_rho(s1_oof, window)            # rho from out-of-fold, not in-sample

    phys_te = engine.predict_physical(te)
    p1_window = engine.probs_from_physical(s1_oof, window)
    p1_test = engine.predict_proba(te)

    # --- stage 2, market-aware -------------------------------------------
    stage2 = MarketModel(seed=seed, target=stage2_target).fit(window, s1_oof)
    p2_test = stage2.predict_proba(te, phys_te)

    # --- market reaction, a diagnostic and nothing more -------------------
    reaction = {}
    try:
        rx = MarketReactionModel(seed=seed).fit(window, s1_oof)
        if rx.models:
            reaction = rx.score(te, phys_te)
    except Exception as exc:                                # pragma: no cover
        warnings.warn(f"market reaction model failed on {fold.test_unit}: {exc}")

    # --- THE SEAM: correct stage 1, never stage 2 ------------------------
    #
    # The correction pulls our estimate toward the market. That only means
    # anything if our estimate was formed independently of the market. Stage 2
    # has the market price among its inputs, so its output is mostly a copy of
    # the market, there is almost nothing left to pull, and the fitted weight
    # goes to one -- the correction switches itself off while still appearing to
    # run. So stage 1 is what gets shrunk, and stage 2 is kept as the comparison.
    p_ref_test = probs(te, "p_ref")

    # THE WEIGHT IS FITTED ON EVERYTHING EARLIER, NOT JUST LAST SEASON.
    # One validation season is a few thousand matches, and a weight read off
    # it moves with that season's noise. The whole window can be used because
    # its predictions are out-of-fold: honest, in the only sense that matters
    # here, since no model ever saw the match it is predicting.
    fit_ref = probs(window, "p_ref")
    fit_model = p1_window
    fit_y = window["target"].astype(int).to_numpy()

    w = fit_static_shrinkage(fit_ref, fit_model, fit_y)

    # ...and a second weight fitted only where the strategy would bet. Average
    # log loss is dominated by matches no threshold would ever select, so the
    # weight it prefers is not the weight the selected bets need.
    band = selection_band(fit_ref, fit_model, coverage=op_coverage)
    w_band = fit_static_shrinkage(fit_ref, fit_model, fit_y, mask=band)

    # ...and the other functional form, fitted the same way.
    c_js = fit_linear_shrinkage(fit_ref, fit_model, fit_y)

    te[outcome_cols("p_indep")] = p1_test
    te[outcome_cols("p_model")] = p2_test
    te[outcome_cols("p_corr")] = apply_static_shrinkage(p_ref_test, p1_test, w)
    te[outcome_cols("p_band")] = apply_static_shrinkage(p_ref_test, p1_test, w_band)
    te[outcome_cols("p_cal")] = calibrate_probabilities(fit_model, fit_y, p1_test)
    te[outcome_cols("p_js")] = apply_linear_shrinkage(p_ref_test, p1_test, c_js)
    te[outcome_cols("p_mispricing")] = stage2.predict_mispricing(te, phys_te)

    # --- per-match shrinkage, two ways -----------------------------------
    # Both produce a weight that varies by match. shrink_by_uncertainty() gets
    # it from a hand-written rule applied to predicted error; LearnedShrinkage
    # fits the mapping against validation log loss directly. Carrying both means
    # the hand-written shape can be judged rather than assumed.
    corr, corr_in = np.nan, np.nan
    if use_learned_uncertainty:
        try:
            head, check = fit_uncertainty_head(window, p1_window, fit_y, seed)
            err = head.predict(uncertainty_inputs(te, p1_test))
            te[outcome_cols("p_unc")] = shrink_by_uncertainty(
                p_ref_test, p1_test, err)
            # Scored on the TEST unit. The head's own check is in-sample and
            # flatters it, so it is kept only for comparison.
            y_te = te["target"].astype(int).to_numpy()
            realised = -np.log(np.clip(p1_test[np.arange(len(y_te)), y_te],
                                       1e-12, 1.0))
            corr = float(np.corrcoef(err, realised)[0, 1])
            corr_in = check["in_sample_corr"]
        except Exception as exc:                            # pragma: no cover
            warnings.warn(f"uncertainty head failed on {fold.test_unit}: {exc}")

    w_stats = {}
    if learn_w:
        try:
            X_window = uncertainty_inputs(window, p1_window)
            X_test = uncertainty_inputs(te, p1_test)
            wx = LearnedShrinkage().fit(fit_ref, fit_model, X_window, fit_y, w)
            weights = wx.weights(X_test)
            te[outcome_cols("p_wfit")] = wx.apply(p_ref_test, p1_test, X_test)
            w_stats = {"wx_mean": float(weights.mean()),
                       "wx_sd": float(weights.std()),
                       "wx_min": float(weights.min()),
                       "wx_max": float(weights.max())}
        except Exception as exc:                            # pragma: no cover
            warnings.warn(f"learned w(x) failed on {fold.test_unit}: {exc}")

    # --- validation bets: the operating threshold and the stake model ------
    # The validation fold needs corrected probabilities of its own before it can
    # produce bets to learn from. They are never pooled into preds.
    va_scored = window.copy()
    va_scored[outcome_cols("p_corr")] = apply_static_shrinkage(
        fit_ref, fit_model, w)
    # The uncorrected arm gets a threshold of its own. If the weight collapses,
    # the corrected arm cannot rank anything and the report falls back to this
    # one -- a threshold read off the corrected arm would then be a quantile of
    # rounding error, and applying it to real edges would select almost
    # everything.
    va_scored[outcome_cols("p_indep")] = fit_model

    # The operating point's threshold is read off VALIDATION scores, so the test
    # unit's own score distribution never decides which of its bets are placed.
    # RANKED BY EDGE, NOT EXPECTED VALUE. Expected value multiplies by the
    # odds, so tightening the threshold selects longer prices whatever the
    # model thinks: the EV-ranked picks here average 7.45 odds against the
    # edge-ranked ones' 4.45, and the apparent decay with coverage is mostly
    # that. Both thresholds are reported; this is the one the claim rests on.
    op_tau = float(np.quantile(
        score_pool(va_scored, "p_corr", criterion=OP_CRITERION),
        1.0 - op_coverage))
    op_tau_indep = float(np.quantile(
        score_pool(va_scored, "p_indep", criterion=OP_CRITERION),
        1.0 - op_coverage))

    stake_model = None
    if learn_stakes:
        try:
            # BY COVERAGE, NOT BY A FIXED NUMBER. This threshold used to be
            # "expected value of at least 2%", which is a quantity in EV units
            # and meant nothing once bets were ranked by edge -- an edge of
            # 0.02 is two percentage points of disagreement with the market,
            # and an arm shrunk onto the market never has that, so every fold
            # produced zero bets to learn from. A quantile always yields bets.
            stake_tau = float(np.quantile(
                score_pool(va_scored, "p_corr", criterion=OP_CRITERION),
                1.0 - stake_coverage))
            stake_model = StakeModel(seed=seed).fit(
                select_bets(va_scored, stake_tau, "p_corr",
                            criterion=OP_CRITERION))
        except Exception as exc:                            # pragma: no cover
            warnings.warn(f"stake model failed on {fold.test_unit}: {exc}")
            stake_model = None

    for col, values in phys_te.items():
        te[col] = values

    y_test = te["target"].astype(int)
    diagnostics = {
        "test_unit": fold.test_unit, "val_unit": fold.val_unit,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "learner": LEARNER, "ensemble_k": ENSEMBLE_K,
        "w": w, "w_band": w_band, "c_js": c_js, "op_tau_indep": op_tau_indep,
        "rho": engine.rho, "n_rho_div": len(engine.rho_by_div), "op_tau": op_tau,
        "uncertainty_corr": corr, "uncertainty_corr_in": corr_in,
        **{f"ll_{name}": log_loss(y_test, p, labels=[0, 1, 2])
           for name, p in (("market", p_ref_test), ("stage1", p1_test),
                           ("stage2", p2_test), ("corrected", probs(te, "p_corr")))},
        "mae_goals_home": float((phys_te["s1_goals_home"] - te["FTHG"]).abs().mean()),
        **w_stats,
        **reaction,
    }
    if has_arm(te, "p_wfit"):
        diagnostics["ll_wfit"] = log_loss(y_test, probs(te, "p_wfit"),
                                          labels=[0, 1, 2])
    return {"test": te, "diagnostics": diagnostics, "stake_model": stake_model}


def run(df: pd.DataFrame, n_blocks: int = 10, min_test_matches: int = 300,
        seed: int = 42, stage2_target: str = "probability",
        use_dixon_coles: bool = True, use_learned_uncertainty: bool = True,
        learn_w: bool = True, learn_stakes: bool = True) -> tuple:
    """Walk forward with the two-stage architecture. Returns (preds, folds, stakes)."""
    features = assert_market_blind([c for c in STAGE1_FEATURES if c in df.columns])
    fold_list, split_col = make_folds(
        df, features, min_test_matches=min_test_matches, n_blocks=n_blocks,
        required=STAGE1_REQUIRED)

    preds, diagnostics, stake_models = [], [], {}
    for fold in fold_list:
        fold = with_fold_elo(df, fold)
        out = fit_fold(fold, seed, stage2_target, use_dixon_coles,
                       use_learned_uncertainty, learn_w, learn_stakes)
        preds.append(out["test"])
        diagnostics.append({"unit": split_col, **out["diagnostics"]})
        if out["stake_model"] is not None:
            stake_models[fold.test_unit] = out["stake_model"]

        if out["diagnostics"]["w"] > W_WARNING_LEVEL:
            warnings.warn(
                f"fold {fold.test_unit}: w = {out['diagnostics']['w']:.3f} "
                f"exceeds {W_WARNING_LEVEL}. With a market-blind stage 1 this is "
                "more surprising than it would otherwise be -- investigate.")

    return (pd.concat(preds, ignore_index=True), pd.DataFrame(diagnostics),
            stake_models)


def staking_comparison(preds: pd.DataFrame, stake_models: dict, unit_col: str,
                       coverages=(0.25, 0.10, 0.05),
                       n_boot: int = 2000) -> pd.DataFrame:
    """
    Flat against quarter-Kelly against the learned rule, at matched coverage.

    Every rule sizes the SAME selected bets, so any difference is allocation
    rather than selection. Total exposure is normalised to one unit per bet for
    the learned rule, so it cannot win by simply staking more -- read
    total_staked before believing any row.

    select_bets() does not carry the fold column, so each bet is mapped back to
    its test unit through match_key. That matters: every bet must be sized by
    the stake model fitted on the fold BEFORE it, never by one fitted later.
    """
    unit_of = preds.drop_duplicates("match_key").set_index("match_key")[unit_col]
    pooled = score_pool(preds, "p_corr")
    rows = []
    for cov in coverages:
        tau = float(np.quantile(pooled, 1.0 - cov))
        base = select_bets(preds, tau, "p_corr")
        if len(base) == 0:
            continue

        variants = {"flat": base,
                    "quarter_kelly": select_bets(preds, tau, "p_corr",
                                                 stake_rule="quarter_kelly")}
        if stake_models:
            units = base["match_key"].map(unit_of)
            learned = [apply_learned_stakes(base[units == unit], model)
                       for unit, model in stake_models.items()
                       if (units == unit).any()]
            if learned:
                variants["learned"] = pd.concat(learned, ignore_index=True)

        for name, bets in variants.items():
            b = bootstrap_roi(bets, n_boot=n_boot)
            rows.append({
                "coverage": cov, "rule": name, "n_bets": b["n_bets"],
                "total_staked": float(bets["Stake"].sum()),
                "roi": b["roi"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                "p_le_zero": b["p_le_zero"],
            })
    return pd.DataFrame(rows)


# =========================================================================
# PART 13 -- REPORT
# =========================================================================


def main(source: str = "raw/all-euro-data-2025-2026.csv", mode: str = "export",
         n_blocks: int = 10, stage2_target: str = "probability",
         use_dixon_coles: bool = True, learn_w: bool = True,
         learn_stakes: bool = True, export: str = None):
    """
    Run the pipeline end to end.

    What it prints, in order: what was loaded, how the market scores (the number
    any model has to beat), one line per fold, out-of-sample scores, accuracy
    with its baselines, ROI at matched coverage, the risk-coverage curves, the
    non-vacuousness gate, and a breakdown by test unit.

    Read them in that order. A model that does not beat the market baseline has
    nothing to correct, and a curve whose intervals all overlap says nothing
    regardless of where its middle sits.
    """
    print("=" * 70)
    print("TWO-STAGE SELECTIVE BETTING PIPELINE")
    print(f"  stage 2 target : {stage2_target}")
    print(f"  Dixon-Coles    : {'on' if use_dixon_coles else 'off'}")
    print(f"  fitted w(x)    : {'on' if learn_w else 'off'}")
    print(f"  learned stakes : {'on' if learn_stakes else 'off'}")
    print(f"  learner        : {LEARNER}, ensemble of {ENSEMBLE_K}")
    print("=" * 70)

    df = load_and_prepare(source) if mode == "export" else load_multiseason(source)

    complete = df[BASE_FEATURE_COLS].notna().all(axis=1).sum()
    print(f"\nFeatures built. {len(df):,} matches, {complete:,} complete.")

    ok = np.isfinite(probs(df, "p_ref")).all(axis=1) & df["target"].notna()
    show("Market baseline (the score to beat):",
         pd.DataFrame([scoring_report(df.loc[ok, "target"].astype(int),
                                      probs(df, "p_ref")[ok.to_numpy()], "market")]))

    preds, folds, stake_models = run(
        df, n_blocks=n_blocks, stage2_target=stage2_target,
        use_dixon_coles=use_dixon_coles, learn_w=learn_w,
        learn_stakes=learn_stakes)
    show("Per-fold diagnostics:", folds,
         "rho < 0 means the engine is adding draw probability.",
         "wx_sd is how much the fitted weight varies by match. Near zero means",
         "  w(x) collapsed back to the constant, and the constant was enough.",
         "corr_drift_* is the market-reaction model. Near zero is expected.")

    y = preds["target"].astype(int).to_numpy()
    arms = arms_present(preds)

    show("Out-of-sample scoring:",
         pd.DataFrame([scoring_report(y, probs(preds, p), lab) for p, lab in arms]))

    show("Accuracy (reported, NOT ranked on):",
         pd.concat([accuracy_report(y, probs(preds, p), lab) for p, lab in arms],
                   ignore_index=True),
         "base_rate on the ALL row is the always-predict-the-commonest-class score.",
         "Watch the draw rows -- that is what rho is meant to buy.")

    # The market arm is the null: shrinkage with w = 0 reproduces it exactly, so
    # a corrected arm that merely matches it has replaced the model, not
    # corrected it. Carrying it makes that visible instead of inferred.
    w_median = float(folds["w"].median())
    if w_median < W_COLLAPSE_LEVEL:
        print(f"\n  WARNING: median fitted w = {w_median:.3f}. The corrected arm "
              "has collapsed onto the market\n  price, so p_corr's rows are the "
              "market's rows, not evidence for a correction.")

    roi_arms = (["p_ref", "p_indep", "p_corr", "p_band", "p_js", "p_cal"]
                + (["p_wfit"] if has_arm(preds, "p_wfit") else []))

    # BOTH CRITERIA, EVERY TIME. Ranking on expected value puts the odds inside
    # the score, so a tighter threshold selects longer prices whatever the model
    # thinks -- a model with no opinion shows the same falling return. Ranking
    # on edge holds the prices still and tightens the disagreement instead.
    # Only the second one is evidence about selection; the first is kept
    # because the difference between them is itself the diagnosis.
    for criterion, note in (("edge", "the claim: more disagreement, same prices"),
                            ("ev", "the confound: tighter also means longer odds")):
        arms_here = [a for a in roi_arms if rankable(preds, a, criterion)]
        dropped = [a for a in roi_arms if a not in arms_here]
        notes = ["shop_gain is mechanical. Subtract it before calling any roi "
                 "an edge.",
                 "p_ref is the market alone: the row a correction has to beat."]
        if dropped:
            notes.append(f"no edge to rank on, so left out: {', '.join(dropped)} "
                         "-- these ARE the market price.")
        show(f"ROI at matched coverage, ranked by {criterion.upper()} "
             f"({note}):",
             betting_report(preds, prefixes=tuple(arms_here), criterion=criterion),
             *notes)

    # C2: what the same bets return if nobody chooses them.
    show("Against odds-matched random picks (top 10% by edge):",
         random_baseline_table(preds, roi_arms, coverage=0.10, criterion="edge"),
         "random picks share the arm's own odds profile, decile by decile.",
         "model_beats_random is the share of 500 draws the arm beat.")

    # Each test unit is thresholded at the tau its own validation unit gave,
    # so no test score helps decide which test bets are placed.
    unit_col = "Block" if "Block" in preds.columns else "Season"

    # WHICH ARM THE OPERATING POINT RUNS ON. When the weight collapses to zero
    # the corrected arm IS the market, its edge is zero on every match, and the
    # only thing left to rank is floating-point residue from renormalising the
    # blend -- around 2e-07. Sorting that and taking the top 10% picks matches
    # at random while looking exactly like a strategy, and the resulting ROI
    # reads as a result. It is not one. So the operating point runs on the
    # corrected arm only while that arm still differs from the market, and
    # falls back to the uncorrected one, which always has opinions of its own.
    op_prefix = "p_corr" if rankable(preds, "p_corr", OP_CRITERION) else "p_indep"
    if op_prefix != "p_corr":
        print("\n  NOTE: the corrected arm has collapsed onto the market, so it "
              "cannot\n  rank anything -- its edge is zero on every match. The "
              "operating point\n  below is the UNCORRECTED arm, which still "
              "selects on its own opinion.")

    tau_col = "op_tau" if op_prefix == "p_corr" else "op_tau_indep"
    op_bets = pd.concat(
        [select_bets(preds[preds[unit_col] == unit], tau, op_prefix,
                     criterion=OP_CRITERION)
         for unit, tau in zip(folds["test_unit"], folds[tau_col])],
        ignore_index=True)
    op_label = "corrected" if op_prefix == "p_corr" else "uncorrected"
    report_roi(op_bets, f"Operating point: {op_label}, top 10% by "
                        f"{OP_CRITERION.upper()} on the training window "
                        f"(per-fold tau in the diagnostics)")

    op_stats = odds_matched_random(preds, op_bets)
    if op_stats:
        print(f"  odds-matched random on the same prices: "
              f"{op_stats['random_mean']:+.2f}% "
              f"[{op_stats['random_lo']:+.2f}, {op_stats['random_hi']:+.2f}], "
              f"model beat {op_stats['model_beats_random']:.0%} of 500 draws")

    if stake_models:
        show("Staking rules on identical bets:",
             staking_comparison(preds, stake_models, unit_col),
             "total_staked shows whether a rule won by allocating or leveraging.")

    curve_arms = [("p_ref", "market only"), ("p_indep", "uncorrected"),
                  ("p_band", "corrected, w fitted on the bet band"),
                  ("p_js", "corrected, James-Stein"),
                  ("p_cal", "calibrated, no market anchor"),
                  ("p_corr", "corrected, static w")]
    if has_arm(preds, "p_wfit"):
        curve_arms.append(("p_wfit", "corrected, fitted w(x)"))
    for criterion in ("edge", "ev"):
        for prefix, label in curve_arms:
            if not rankable(preds, prefix, criterion):
                continue
            curve = risk_coverage_curve(preds, prefix, criterion=criterion)
            show(f"Risk-coverage by {criterion}, {label}:  "
                 f"AURC = {aurc(curve):+.3f}", curve)

    nv = check_non_vacuousness(preds, "p_corr", "p_indep")
    if len(nv):
        show(f"Non-vacuousness check (ranked by {OP_CRITERION}):", nv)
    else:
        print(f"\nNon-vacuousness check: skipped. The corrected arm has no "
              f"{OP_CRITERION} to rank on\n  -- it collapsed onto the market -- "
              "so there is no corrected selection to compare.")
    if len(nv) and nv["overlap"].mean() > 0.95:
        print("\n  WARNING: the corrected and uncorrected rules are selecting "
              "almost the same bets.\n  The correction may be a relabelled "
              "threshold rather than a correction.")

    show(f"By {unit_col.lower()} ({op_label} arm, validation tau):",
         era_breakdown(preds, folds, prob_prefix=op_prefix, unit_col=unit_col))

    if export:
        keep = ["match_key", "Date", "Season", "Div", "Tier", "target"]
        keep += [c for c in preds.columns
                 if c.startswith(("odds_best_", "odds_avg_"))
                 or any(c.startswith(f"{p}_") for p in
                        ("p_ref", "p_indep", "p_model", "p_corr", "p_band",
                         "p_js", "p_cal", "p_wfit", "p_unc"))]
        slim = preds[[c for c in keep if c in preds.columns]]
        slim.to_csv(export, index=False)
        print(f"\nwrote {export}: {len(slim):,} rows x {slim.shape[1]} columns "
              f"-- per-match predictions, so any season can be sliced out "
              f"without running this again")

    print("\n" + "=" * 70)
    print("Read the intervals, not the middles. Overlapping intervals mean the")
    print("difference is not established, whatever the point estimates say.")
    if "Block" in preds.columns:
        print()
        print("This run used ONE SEASON, split into blocks. The folds share")
        print("teams, squads and one market regime, so they are not independent")
        print("and these intervals understate the true uncertainty. Nothing")
        print("here supports a claim about a real edge.")
    print("=" * 70)
    return df, preds, folds


# =========================================================================
# PART 14 -- ELO
#
# Elo ratings built from results alone, with every constant fitted rather than
# chosen. Textbook Elo hides half a dozen hand-set numbers: the update size K,
# how much a home ground is worth, how much a three-goal win counts over a
# one-goal win, how much of a rating survives the summer. Here they are
# parameters, fitted on training seasons by log loss and then held fixed.
#
# The ratings are strictly pre-match: a match is predicted from the ratings the
# two teams carried INTO it, and only afterwards do those ratings move.
#
# Teams are keyed by country and name, not by division, so a promoted side
# carries its rating up with it. A team seen for the first time starts at the
# current average of its division, not at a fixed 1500.
# =========================================================================

ELO_START = 1500.0            # the scale's origin; any constant would do

# (low, high) for each fitted parameter, in the order of EloParams
ELO_BOUNDS = [(1.0, 120.0),   # k          update size
          (0.0, 250.0),   # home_adv   rating points a home ground is worth
          (0.0, 1.5),     # gamma      how much a bigger margin counts
          (0.0, 1.0),     # carry      fraction of a rating kept over a summer
          (0.0005, 0.05), # beta       rating points -> latent strength
          (0.05, 3.0)]    # delta      half-width of the drawing band


@dataclass
class EloParams:
    k: float = 20.0
    home_adv: float = 60.0
    gamma: float = 0.5
    carry: float = 0.75
    beta: float = 0.0058          # ln(10)/400, the textbook scale
    delta: float = 0.5

    def clipped(self) -> "EloParams":
        return EloParams(*[float(np.clip(v, lo, hi))
                           for v, (lo, hi) in zip(astuple(self), ELO_BOUNDS)])


def elo_outcome_probs(strength: np.ndarray, p: EloParams) -> np.ndarray:
    """
    Turn a rating difference into home/draw/away probabilities.

    An ordered logit, which is the natural choice: the three results are
    ordered, and a draw is what happens when the latent difference in strength
    is too small to decide the match either way. delta is the width of that
    indecisive band, fitted like everything else.
    """
    s = p.beta * strength
    p_away = expit(-p.delta - s)
    p_home = 1.0 - expit(p.delta - s)
    return np.column_stack([p_home, 1.0 - p_home - p_away, p_away])


def add_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure a Country column exists, because that is half the team key.

    Divisions carry it implicitly (E0 and E1 are both England), and teams move
    between divisions inside a country every summer. Keying on the division
    instead would reset a promoted team to its new division's average and throw
    away the very thing we want to measure.
    """
    if "Country" in df.columns and df["Country"].notna().all():
        return df
    out = df.copy()
    out["Country"] = out["Div"].map(lambda x: TIERS.get(x, (str(x), 0))[0])
    return out


def run_elo(df: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    """
    Walk the fixtures in date order and return the PRE-match state of each one.

    Everything returned for a match is known before it kicks off, so these
    columns can go straight into a market-blind model without leaking.
    """
    p = params.clipped()
    d = add_country(df).sort_values(["Date", "Div", "HomeTeam"], kind="mergesort")

    home_key = d["Country"].astype(str) + "|" + d["HomeTeam"].astype(str)
    away_key = d["Country"].astype(str) + "|" + d["AwayTeam"].astype(str)
    divs = d["Div"].astype(str).to_numpy()
    seasons = d["Season"].astype(str).to_numpy()
    gh = d["FTHG"].to_numpy(dtype=float)
    ga = d["FTAG"].to_numpy(dtype=float)

    rating: dict[str, float] = {}
    div_members: dict[str, set] = {}
    eh = np.empty(len(d))
    ea = np.empty(len(d))
    season_now = seasons[0] if len(d) else ""

    def rating_of(team: str, div: str) -> float:
        if team not in rating:
            peers = [rating[t] for t in div_members.get(div, ()) if t in rating]
            rating[team] = float(np.mean(peers)) if peers else ELO_START
        div_members.setdefault(div, set()).add(team)
        return rating[team]

    for i, (h, a, div, season) in enumerate(zip(home_key, away_key, divs, seasons)):
        if season != season_now:            # summer: regress toward the mean
            season_now = season
            if rating:
                m = float(np.mean(list(rating.values())))
                for t in rating:
                    rating[t] = m + p.carry * (rating[t] - m)

        rh, ra = rating_of(h, div), rating_of(a, div)
        eh[i], ea[i] = rh, ra

        if np.isnan(gh[i]) or np.isnan(ga[i]):       # unplayed: no update
            continue
        expected = 1.0 / (1.0 + 10.0 ** (-(rh + p.home_adv - ra) / 400.0))
        score = 1.0 if gh[i] > ga[i] else (0.5 if gh[i] == ga[i] else 0.0)
        move = p.k * (1.0 + abs(gh[i] - ga[i])) ** p.gamma * (score - expected)
        rating[h], rating[a] = rh + move, ra - move

    strength = eh + p.home_adv - ea
    p3 = elo_outcome_probs(strength, p)
    return pd.DataFrame({
        "elo_home": eh,
        "elo_away": ea,
        "elo_diff": strength,
        "elo_p_H": p3[:, 0],
        "elo_p_D": p3[:, 1],
        "elo_p_A": p3[:, 2],
    }, index=d.index).reindex(df.index)


def log_loss_of(df: pd.DataFrame, params: EloParams) -> float:
    """Mean log loss of the Elo probabilities against the actual results."""
    out = run_elo(df, params)
    played = df["FTR"].isin(["H", "D", "A"]).to_numpy()
    if not played.any():
        return np.inf
    col = df["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()[played].astype(int)
    p3 = out[["elo_p_H", "elo_p_D", "elo_p_A"]].to_numpy()[played]
    p_actual = p3[np.arange(len(col)), col]
    return float(-np.mean(np.log(np.clip(p_actual, 1e-12, 1.0))))


def fit_elo(train: pd.DataFrame, seed_params: EloParams = EloParams(),
            verbose: bool = False) -> tuple[EloParams, float]:
    """
    Fit the six constants on training matches by log loss.

    Nelder-Mead, because the walk is not differentiable in any useful sense:
    a change in K changes every later rating through a chain of updates. The
    search is small enough (six parameters) that this is not a problem.
    """
    def objective(x: np.ndarray) -> float:
        loss = log_loss_of(train, EloParams(*x))
        if verbose:
            print(f"    {np.round(x, 4)} -> {loss:.5f}", flush=True)
        return loss

    res = minimize(objective, np.array(astuple(seed_params)),
                   method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-6})
    return EloParams(*res.x).clipped(), float(res.fun)


def elo_report(path: str = "matches_multiseason.csv",
               verbose: bool = False) -> None:
    """Does Elo know anything the base rate does not? Fit, then test on the last season."""
    df = read_match_table(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Season"].astype(str)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    counts = df["Season"].value_counts()
    df = df[df["Season"].isin(counts[counts >= 1000].index)]
    df = add_country(df).sort_values("Date").reset_index(drop=True)

    seasons = sorted(df["Season"].unique())
    test_season = seasons[-1]
    train = df[df["Season"] < test_season]
    print(f"{len(df):,} matches, seasons {', '.join(seasons)}; "
          f"fitting on {', '.join(seasons[:-1])}, testing on {test_season}")

    params, train_loss = fit_elo(train, verbose=verbose)
    print(f"\nfitted: k={params.k:.1f}  home_adv={params.home_adv:.1f}  "
          f"gamma={params.gamma:.2f}  carry={params.carry:.2f}  "
          f"beta={params.beta:.5f}  delta={params.delta:.2f}")
    print(f"train log loss {train_loss:.4f}")

    # Ratings must be carried through the training seasons to be right for the
    # test season, so the walk runs over everything and is then sliced.
    out = run_elo(df, params)
    mask = (df["Season"] == test_season).to_numpy()
    actual = df.loc[mask, "FTR"].to_numpy()
    onehot = np.column_stack([actual == "H", actual == "D", actual == "A"]).astype(float)

    def loss(p3: np.ndarray) -> float:
        return float(-np.mean(np.log(np.clip((p3 * onehot).sum(1), 1e-12, 1.0))))

    elo_p = out.loc[mask, ["elo_p_H", "elo_p_D", "elo_p_A"]].to_numpy()
    base = train["FTR"].value_counts(normalize=True)
    base_p = np.tile([base.get("H", 0), base.get("D", 0), base.get("A", 0)],
                     (mask.sum(), 1))

    rows = [("base rate (train frequencies)", loss(base_p)),
            ("elo", loss(elo_p))]

    cols = ["AvgCH", "AvgCD", "AvgCA"]
    if all(c in df.columns for c in cols):
        sub = df.loc[mask, cols]
        ok = sub.notna().all(axis=1).to_numpy()
        market = np.array([shin_de_vig(h, d, a)
                           for h, d, a in sub[ok].to_numpy()])
        m_onehot = onehot[ok]
        m_loss = float(-np.mean(np.log(np.clip((market * m_onehot).sum(1), 1e-12, 1.0))))
        rows.append((f"market closing ({ok.sum():,} of {mask.sum():,} matches)", m_loss))

    print(f"\nout-of-sample log loss on {test_season} ({mask.sum():,} matches)")
    for label, value in rows:
        print(f"  {label:<40} {value:.4f}")


# =========================================================================
# PART 15 -- SEED SWEEP
#
# One seed gives one number. Whether that number is the pipeline or the draw
# cannot be read off it, so this runs the whole walk-forward N times, changing
# nothing but the seed, and reports distributions instead of points:
#
#   1. THE WEIGHT. w and w_band for every seed and fold. Is w zero every time?
#   2. THE TAIL. Log loss on the betting band alone -- the matches where
#      stage 1 disagrees most with the market, where a small edge could hide.
#   3. LUCK, three kinds: across seeds, a bootstrap over whole WEEKS, and the
#      share of odds-matched random draws each seed beats.
#
# Nothing in the pipeline changes. Each seed calls run() as is, with the stake
# model switched off (it does not affect selection).
# =========================================================================

SWEEP_COVERAGE = 0.10
SWEEP_ARMS = ("p_indep", "p_corr", "p_band", "p_js", "p_cal")


def week_bootstrap(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """ROI interval resampling whole weeks rather than single matches."""
    if bets.empty:
        return {"wk_lo": np.nan, "wk_hi": np.nan, "wk_p_le_zero": np.nan}
    week = pd.to_datetime(bets["Date"]).dt.to_period("W").astype(str)
    g = bets.groupby(week.values)
    stake, pnl = g["Stake"].sum().to_numpy(), g["PnL"].sum().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(stake), (n_boot, len(stake)))
    roi = pnl[idx].sum(1) / stake[idx].sum(1) * 100
    return {"wk_lo": float(np.percentile(roi, 2.5)),
            "wk_hi": float(np.percentile(roi, 97.5)),
            "wk_p_le_zero": float((roi <= 0).mean())}


def band_log_loss(te: pd.DataFrame) -> dict:
    """Log loss on the whole test unit and on its betting band, per arm."""
    y = te["target"].astype(int).to_numpy()
    p_ref, p_ind = probs(te, "p_ref"), probs(te, "p_indep")
    band = selection_band(p_ref, p_ind, coverage=SWEEP_COVERAGE)
    out = {"n_band": int(band.sum())}
    for name, prefix in (("market", "p_ref"), ("stage1", "p_indep"),
                         ("corr", "p_corr"), ("band", "p_band")):
        p = probs(te, prefix)
        out[f"ll_all_{name}"] = log_loss(y, p, labels=[0, 1, 2])
        out[f"ll_band_{name}"] = log_loss(y[band], p[band], labels=[0, 1, 2])
    return out


def roi_of(bets: pd.DataFrame) -> float:
    stake = bets["Stake"].sum() if len(bets) else 0.0
    return float(bets["PnL"].sum() / stake * 100) if stake > 0 else np.nan


def one_seed(df: pd.DataFrame, seed: int, learner: str, k: int,
             n_blocks: int, learn_w: bool) -> tuple:
    """The full walk-forward under one seed, reduced to what the sweep needs."""
    configure(learner=learner, ensemble_k=k)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, folds, _ = run(df, n_blocks=n_blocks, seed=seed,
                                 learn_w=learn_w, learn_stakes=False)
    unit_col = "Block" if "Block" in preds.columns else "Season"

    # The operating point, chosen exactly as main() chooses it: the corrected
    # arm while it can still rank anything, the uncorrected one once it cannot.
    op_prefix = ("p_corr" if rankable(preds, "p_corr", OP_CRITERION)
                 else "p_indep")
    tau_col = "op_tau" if op_prefix == "p_corr" else "op_tau_indep"

    fold_rows, op_frames = [], []
    for _, f in folds.iterrows():
        te = preds[preds[unit_col] == f["test_unit"]]
        bets = select_bets(te, f[tau_col], op_prefix, criterion=OP_CRITERION)
        op_frames.append(bets.assign(fold=f["test_unit"]))
        fold_rows.append({
            "seed": seed, "fold": f["test_unit"],
            "w": f["w"], "w_band": f["w_band"], "c_js": f["c_js"],
            **band_log_loss(te),
            "op_bets": len(bets), "op_pnl": float(bets["PnL"].sum()) if len(bets) else 0.0,
            "op_stake": float(bets["Stake"].sum()) if len(bets) else 0.0,
            "op_roi": roi_of(bets)})
    op_bets = pd.concat(op_frames, ignore_index=True).assign(seed=seed)

    fold_pnl = pd.Series([r["op_pnl"] for r in fold_rows])
    total = fold_pnl.sum()
    rnd = odds_matched_random(preds, op_bets, n_draws=300, seed=seed)

    summary = {"seed": seed, "learner": learner, "ensemble_k": k,
               "op_arm": op_prefix, "op_bets": len(op_bets),
               "op_roi": roi_of(op_bets),
               **week_bootstrap(op_bets, seed=seed),
               "random_mean": rnd.get("random_mean", np.nan),
               "beats_random": rnd.get("model_beats_random", np.nan),
               # Above 1 means one fold made more than the whole run did.
               "best_fold_share": float(fold_pnl.max() / total) if total > 0 else np.nan,
               "w_median": float(folds["w"].median()),
               "w_band_median": float(folds["w_band"].median()),
               "seconds": time.time() - t0}

    # Every arm at matched coverage, thresholded on the pooled test scores the
    # way betting_report() does it. Comparable across arms; not a strategy.
    for prefix in SWEEP_ARMS:
        if has_arm(preds, prefix) and rankable(preds, prefix, "edge"):
            tau = float(np.quantile(score_pool(preds, prefix, criterion="edge"),
                                    1.0 - SWEEP_COVERAGE))
            summary[f"roi10_{prefix}"] = roi_of(
                select_bets(preds, tau, prefix, criterion="edge"))
        else:
            summary[f"roi10_{prefix}"] = np.nan        # collapsed onto the market

    keep = ["seed", "fold", "match_key", "Date", "Div", "Selection", "Odds",
            "p_est", "p_ref", "Score", "IsWin", "Stake", "PnL"]
    return fold_rows, summary, op_bets[[c for c in keep if c in op_bets.columns]]


# -------------------------------------------------------------------------
def _q(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "n/a"
    return (f"{s.median():+.3f}  [{s.quantile(0.05):+.3f}, "
            f"{s.quantile(0.95):+.3f}]")


def sweep_report(fold_df: pd.DataFrame, seed_df: pd.DataFrame) -> str:
    out = []
    say = out.append
    n_seeds = seed_df["seed"].nunique()
    say("=" * 72)
    say(f"SEED SWEEP  {n_seeds} seeds x {fold_df['fold'].nunique()} folds   "
        f"learner={seed_df['learner'].iloc[0]}  "
        f"ensemble_k={seed_df['ensemble_k'].iloc[0]}")
    say("Figures are median [5th, 95th percentile] unless marked otherwise.")
    say("=" * 72)

    # 1. the weight -------------------------------------------------------
    say("\n1. THE SHRINKAGE WEIGHT  (0 = trust the market, 1 = trust stage 1)")
    for col, label in (("w", "w, fitted on every match"),
                       ("w_band", "w_band, fitted on the betting band only")):
        s = fold_df[col]
        say(f"  {label}")
        say(f"     all seed x fold : {_q(s)}   mean {s.mean():.3f}")
        say(f"     share < 0.05    : {(s < W_COLLAPSE_LEVEL).mean():.0%}   "
            f"share > 0.5 : {(s > W_WARNING_LEVEL).mean():.0%}")
    gap = fold_df["w_band"] - fold_df["w"]
    say(f"  w_band - w        : {_q(gap)}   w_band higher in "
        f"{(gap > 0.05).mean():.0%} of seed x fold")

    per_fold = fold_df.groupby("fold").agg(
        w_med=("w", "median"), w_p5=("w", lambda s: s.quantile(0.05)),
        w_p95=("w", lambda s: s.quantile(0.95)),
        wb_med=("w_band", "median"), wb_p5=("w_band", lambda s: s.quantile(0.05)),
        wb_p95=("w_band", lambda s: s.quantile(0.95)),
        roi_med=("op_roi", "median"))
    say("\n  By fold (across seeds):")
    say(per_fold.round(3).to_string())

    # 2. the tail ---------------------------------------------------------
    say("\n2. WHERE THE STRATEGY BETS  (log loss, lower is better; mean over "
        "seed x fold)")
    for scope in ("all", "band"):
        row = {n: fold_df[f"ll_{scope}_{n}"].mean()
               for n in ("market", "stage1", "corr", "band")}
        say(f"  {'every match' if scope == 'all' else 'betting band':13s}"
            + "  ".join(f"{n} {v:.4f}" for n, v in row.items()))
    d = fold_df["ll_band_stage1"] - fold_df["ll_band_market"]
    db = fold_df["ll_band_band"] - fold_df["ll_band_market"]
    say(f"  stage 1 minus market, band : {_q(d)}   stage 1 better in "
        f"{(d < -1e-4).mean():.0%}")
    say(f"  w_band arm minus market    : {_q(db)}   better in {(db < -1e-4).mean():.0%}")
    say("  A negative number would be an edge the all-match log loss hides.")

    # 3. luck ---------------------------------------------------------------
    say("\n3. LUCK CHECKS  (operating point: top 10% by edge, tau from validation)")
    arms = seed_df["op_arm"].value_counts().to_dict()
    say(f"  arm the operating point ran on : {arms}")
    r = seed_df["op_roi"]
    say(f"  seed    ROI across seeds       : {_q(r)}  (min {r.min():+.2f}, "
        f"max {r.max():+.2f})")
    say(f"          seeds with ROI > 0     : {(r > 0).mean():.0%}")
    if 42 in set(seed_df["seed"]):
        r42 = float(seed_df.loc[seed_df["seed"] == 42, "op_roi"].iloc[0])
        say(f"          seed 42 (the default)  : {r42:+.2f}%, percentile "
            f"{(r < r42).mean():.0%} of the sweep")
    say(f"  sample  week-bootstrap P(ROI<=0): {_q(seed_df['wk_p_le_zero'])}")
    say(f"          seeds whose 95% CI > 0  : {(seed_df['wk_lo'] > 0).mean():.0%}")
    say(f"  select  share of random beaten : {_q(seed_df['beats_random'])}")
    say(f"          seeds beating > 95%    : {(seed_df['beats_random'] > 0.95).mean():.0%}")
    say(f"  folds   best fold's share of profit (profitable seeds only): "
        f"{_q(seed_df['best_fold_share'])}")
    say("          above 1.0 means one fold earned more than the whole run.")

    # 4. every arm ------------------------------------------------------------
    say("\n4. EVERY ARM AT 10% SWEEP_COVERAGE BY EDGE  (pooled test threshold; for "
        "comparison, not a strategy)")
    for prefix in SWEEP_ARMS:
        s = seed_df.get(f"roi10_{prefix}")
        if s is None or s.isna().all():
            say(f"  {prefix:8s} collapsed onto the market in every seed")
            continue
        say(f"  {prefix:8s} {_q(s)}   ROI > 0 in {(s > 0).mean():.0%}   "
            f"(rankable in {s.notna().mean():.0%} of seeds)")

    say(f"\n  {seed_df['seconds'].median():.0f}s per seed (median)")
    return "\n".join(out)


def _importable_self():
    """
    This file as an importable module, for the sweep's worker processes.

    configure() changes module-level settings, so each worker must hold a real
    module of its own rather than a pickled copy of functions from __main__.
    """
    if __name__ != "__main__":
        return sys.modules[__name__]
    here = Path(__file__).resolve()
    if str(here.parent) not in sys.path:
        sys.path.insert(0, str(here.parent))
    return importlib.import_module(here.stem)


def sweep_main(args) -> None:
    from joblib import Parallel, delayed

    source = args.source or DEFAULT_SOURCES[args.mode]
    out = Path(args.out) if args.out else (
        DATA / "sweeps" /
        f"{args.mode}_{args.learner}_k{args.ensemble_k}_s{args.seeds}")
    out.mkdir(parents=True, exist_ok=True)

    df = (load_and_prepare(source) if args.mode == "export"
          else load_multiseason(source))
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"{len(seeds)} seeds on {args.jobs} workers -> {out}")

    worker = _importable_self().one_seed
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
        delayed(worker)(df, s, args.learner, args.ensemble_k, args.blocks,
                        args.learned_w)
        for s in seeds)

    fold_df = pd.DataFrame([row for rows, _, _ in results for row in rows])
    seed_df = pd.DataFrame([summary for _, summary, _ in results])
    bets_df = pd.concat([b for _, _, b in results], ignore_index=True)
    fold_df.to_csv(out / "folds.csv", index=False)
    seed_df.to_csv(out / "seeds.csv", index=False)
    bets_df.to_csv(out / "op_bets.csv", index=False)

    text = sweep_report(fold_df, seed_df)
    text += f"\n  {time.time() - t0:.0f}s wall clock\n"
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"wrote folds.csv, seeds.csv, op_bets.csv, summary.txt to {out}")



# =========================================================================
# PART 17 -- THE OPENING-PRICE STRATEGY  (closing-line value)
#
# The main pipeline bets INTO the closing price, the most efficient number in
# the market, and on this data it loses. This asks a different question: when
# the opening price first appears, can we tell which prices are too long --
# which ones the market will shorten by kickoff -- and take them before it does?
#
# THE MEASURE IS CLOSING-LINE VALUE, NOT PROFIT. A bet's CLV is the price taken
# times the de-vigged CLOSING probability, minus one: what the bet was worth
# by the market's own final opinion. It is far less noisy than profit, because
# it does not wait for the ball to go in, and positive CLV over many bets is the
# standard evidence of a real edge. Profit is reported too, with its interval.
#
# WHAT THE MODEL MAY SEE. Only what exists when the opening price is quoted:
# the opening prices themselves, stage 1's market-blind view, and league
# context. Every closing price, and anything built from one, is a LABEL here.
# assert_pre_close() enforces it, the same way assert_market_blind() guards
# stage 1, because a closing price reaching the inputs would score perfectly
# while predicting nothing.
#
# THE STALE-QUOTE TRAP. Max (the best opening price across books) includes
# quotes one book forgot to move. A model told to maximise CLV at Max learns to
# find those, which looks like an edge and cannot be bet at size. So the model
# is trained on CLV at the AVERAGE opening price -- a price most books offer --
# and results are reported at both. If the edge only exists at Max, it is the
# trap, not a model.
#
# WHEN "OPENING" IS. In the football-data files the non-closing prices are
# collected a day or two before the match (Friday for a weekend fixture). Read
# "opening" as "that snapshot", not as the first price ever quoted.
# =========================================================================

CLOSE_TOKENS = ("close", "AvgC", "MaxC", "PSC", "B365C", "BWC", "IWC", "WHC",
                "VCC", "p_ref", "drift", "odds_best", "odds_avg", "spread_mean")
OPEN_MARKET_FEATURES = ([f"{p}_{o}" for p in ("p_open", "log_open", "open_spread")
                         for o in OUTCOMES] + ["overround_open"])
CLV_CONTEXT_FEATURES = ["Tier", "min_played", "league_goals_20",
                        "league_home_rate_20"]
CLV_COVERAGES = (0.05, 0.10, 0.20)


def assert_pre_close(features: list) -> list:
    """Fail loudly if anything known only at kickoff reached the inputs."""
    leaked = [c for c in features
              if any(t.lower() in c.lower() for t in CLOSE_TOKENS)]
    if leaked:
        raise ValueError(f"Closing-time data leaked into the opening model: {leaked}")
    return features


def add_opening_market(df: pd.DataFrame) -> pd.DataFrame:
    """Opening-price features, plus the closing probability as a LABEL only."""
    d = df.copy()
    op = np.array([shin_de_vig(*r) for r in d[["AvgH", "AvgD", "AvgA"]].to_numpy()])
    cl = np.array([shin_de_vig(*r) for r in d[["AvgCH", "AvgCD", "AvgCA"]].to_numpy()])
    for k, o in enumerate(OUTCOMES):
        d[f"p_open_{o}"] = op[:, k]
        d[f"log_open_{o}"] = np.log(d[f"Avg{o}"])
        d[f"open_spread_{o}"] = d[f"Max{o}"] / d[f"Avg{o}"] - 1.0
        d[f"q_close_{o}"] = cl[:, k]                       # label, never input
    d["overround_open"] = sum(1.0 / d[f"Avg{o}"] for o in OUTCOMES) - 1.0
    return d


def _attach_stage1(d: pd.DataFrame, phys: pd.DataFrame, p1: np.ndarray) -> pd.DataFrame:
    d = d.copy()
    for k, o in enumerate(OUTCOMES):
        d[f"p1_{o}"] = p1[:, k]
        d[f"gap_{o}"] = p1[:, k] - d[f"p_open_{o}"]       # stage 1 vs opening price
    for c in phys.columns:
        d[c] = phys[c].to_numpy()
    return d


def _clv_feature_sets(d: pd.DataFrame) -> dict:
    s1 = ([f"p1_{o}" for o in OUTCOMES] + [f"gap_{o}" for o in OUTCOMES]
          + [c for c in d.columns if c.startswith("s1_")])
    market = [c for c in OPEN_MARKET_FEATURES + CLV_CONTEXT_FEATURES if c in d.columns]
    return {"with stage 1": assert_pre_close(market + s1),
            "market only": assert_pre_close(market)}


def _col(d: pd.DataFrame, c: str) -> np.ndarray:
    """A price column if this file has it, otherwise all missing."""
    return d[c].to_numpy(float) if c in d.columns else np.full(len(d), np.nan)


def _candidates(d: pd.DataFrame, pred: dict) -> pd.DataFrame:
    """One row per match per outcome: prices, value at close, result, prediction."""
    frames = []
    for k, o in enumerate(OUTCOMES):
        frames.append(pd.DataFrame({
            "match_key": d["match_key"].to_numpy(), "Date": d["Date"].to_numpy(),
            "Season": d["Season"].to_numpy(), "Div": d["Div"].to_numpy(),
            "outcome": o,
            "odds_avg": d[f"Avg{o}"].to_numpy(float),
            "odds_best": d[f"Max{o}"].to_numpy(float),
            "clv_avg": (d[f"Avg{o}"] * d[f"q_close_{o}"] - 1.0).to_numpy(float),
            "clv_best": (d[f"Max{o}"] * d[f"q_close_{o}"] - 1.0).to_numpy(float),
            "win": (d["target"].to_numpy() == k).astype(int),
            "q_close": d[f"q_close_{o}"].to_numpy(float),
            "odds_ps": _col(d, f"PS{o}"),
            "odds_bfe": _col(d, f"BFE{o}"),
            **{f"pred_{name}": v[o] for name, v in pred.items()},
        }))
    return pd.concat(frames, ignore_index=True)


def clv_fold(window: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> tuple:
    """Fit stage 1 and the CLV models on the window; score the test season."""
    window = window.reset_index(drop=True)
    test = test.reset_index(drop=True)

    engine = MatchEngine(seed=seed)
    oof = engine.fit_predict_oof(window)
    engine.fit(window)
    engine.fit_rho(oof, window)
    W = _attach_stage1(window, oof, engine.probs_from_physical(oof, window))
    phys_te = engine.predict_physical(test)
    T = _attach_stage1(test, phys_te, engine.predict_proba(test))

    pred_w, pred_t = {}, {}
    for name, feats in _clv_feature_sets(W).items():
        pred_w[name], pred_t[name] = {}, {}
        for o in OUTCOMES:
            y = (W[f"Avg{o}"] * W[f"q_close_{o}"] - 1.0).to_numpy(float)
            Xw, Xt = W[feats].to_numpy(float), T[feats].to_numpy(float)
            # Out-of-fold on the window, so the threshold is read off
            # predictions no model made about its own training rows.
            pred_w[name][o] = cross_val_predict(make_regressor(seed), Xw, y, cv=3)
            pred_t[name][o] = make_regressor(seed).fit(Xw, y).predict(Xt)
    return _candidates(W, pred_w), _candidates(T, pred_t)


def _match_boot(b: pd.DataFrame, col: str, n_boot: int = 5000, seed: int = 42):
    g = b.groupby("match_key")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (n_boot, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1) * 100
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m <= 0).mean())


def clv_summary(b: pd.DataFrame, label: str) -> dict:
    """CLV and profit for one set of bets, both with intervals over matches."""
    if b.empty:
        return {"set": label, "n_bets": 0}
    lo, hi, p0 = _match_boot(b, "clv_avg")
    bb = b.assign(Stake=1.0, PnL=np.where(b["win"] == 1, b["odds_best"] - 1.0, -1.0),
                  PnL_Avg=np.nan)
    r = bootstrap_roi(bb, n_boot=5000)
    return {"set": label, "n_bets": len(b), "avg_odds": b["odds_best"].mean(),
            "units_staked": float(len(b)), "units_won": float(bb["PnL"].sum()),
            "clv_avg%": b["clv_avg"].mean() * 100, "clv_lo": lo, "clv_hi": hi,
            "P(clv<=0)": p0, "clv_best%": b["clv_best"].mean() * 100,
            "roi_best%": r["roi"], "roi_lo": r["ci_low"], "roi_hi": r["ci_high"],
            "roi_avg%": np.where(b["win"] == 1, b["odds_avg"] - 1.0, -1.0).mean() * 100}


def clv_random(pool: pd.DataFrame, bets: pd.DataFrame, n_draws: int = 300,
               seed: int = 42) -> dict:
    """Random picks at the same opening odds, decile by decile."""
    if bets.empty:
        return {}
    edges = np.unique(np.quantile(pool["odds_best"], np.linspace(0, 1, 11)))
    pb = np.clip(np.digitize(pool["odds_best"], edges[1:-1]), 0, len(edges) - 2)
    bb = np.clip(np.digitize(bets["odds_best"], edges[1:-1]), 0, len(edges) - 2)
    by = {k: np.flatnonzero(pb == k) for k in np.unique(pb)}
    want = pd.Series(bb).value_counts()
    clv = pool["clv_avg"].to_numpy()
    prof = np.where(pool["win"] == 1, pool["odds_best"] - 1.0, -1.0)
    rng = np.random.default_rng(seed)
    rc, rr = np.empty(n_draws), np.empty(n_draws)
    for i in range(n_draws):
        idx = np.concatenate([rng.choice(by[k], n, replace=len(by[k]) < n)
                              for k, n in want.items() if k in by])
        rc[i], rr[i] = clv[idx].mean() * 100, prof[idx].mean() * 100
    m_clv = bets["clv_avg"].mean() * 100
    m_roi = np.where(bets["win"] == 1, bets["odds_best"] - 1.0, -1.0).mean() * 100
    return {"model_clv%": m_clv, "random_clv%": rc.mean(),
            "random_clv_p95": np.percentile(rc, 95), "beats_random_clv": (rc < m_clv).mean(),
            "model_roi%": m_roi, "random_roi%": rr.mean(),
            "beats_random_roi": (rr < m_roi).mean()}


# --- where the bet is placed ------------------------------------------------
#
# The opening-price model predicts value at the AVERAGE price, and on its own
# that is not enough to beat the average book's margin. A cheaper venue changes
# that arithmetic without changing the model: its price is known when the bet
# is placed, so the prediction converts exactly --
#
#     predicted value at venue X  =  X / Avg * (1 + predicted value at Avg) - 1
#
# -- and the bet is placed when that clears a margin FIXED IN ADVANCE (0, 1, 2
# percent). Fixed, not tuned: the exchange columns only start in 2024/25, so
# the early folds have no exchange history to tune a threshold on, and a
# threshold chosen on the test seasons would be the old pipeline's mistake.
#
# Exchange winnings pay commission, so its price is cut to what you actually
# receive: 1 + (odds - 1)(1 - commission). Losses pay none.
#
# Judged as before: value against the de-vigged closing AVERAGE, and profit,
# both with intervals over matches, and against random picks at the same
# prices on the same venue.
VENUES = (("Average (reference)", "odds_avg", 0.0),
          ("Best price (optimistic)", "odds_best", 0.0),
          ("Pinnacle", "odds_ps", 0.0),
          ("Betfair Exchange, 2% commission", "odds_bfe", 0.02),
          ("Betfair Exchange, 5% commission", "odds_bfe", 0.05))
VENUE_MARGINS = (0.0, 0.01, 0.02)


def at_venue(pool: pd.DataFrame, col: str, commission: float,
             name: str = "with stage 1") -> pd.DataFrame:
    """The candidates that venue quoted, re-priced at what it actually pays."""
    v = pool[pool[col].notna() & (pool[col] > 1.0)].copy()
    v["odds_x"] = 1.0 + (v[col] - 1.0) * (1.0 - commission)
    v["pred_x"] = v["odds_x"] / v["odds_avg"] * (1.0 + v[f"pred_{name}"]) - 1.0
    v["clv_x"] = v["odds_x"] * v["q_close"] - 1.0
    return v


def venue_summary(b: pd.DataFrame, label: str) -> dict:
    if b.empty:
        return {"rule": label, "n_bets": 0}
    lo, hi, p0 = _match_boot(b, "clv_x")
    bb = b.assign(Stake=1.0, PnL=np.where(b["win"] == 1, b["odds_x"] - 1.0, -1.0))
    r = bootstrap_roi(bb, n_boot=5000)
    return {"rule": label, "n_bets": len(b),
            "seasons": ",".join(sorted(b["Season"].astype(str).unique())),
            "avg_odds": b["odds_x"].mean(),
            "units_staked": float(len(b)), "units_won": float(bb["PnL"].sum()),
            "clv%": b["clv_x"].mean() * 100, "clv_lo": lo, "clv_hi": hi,
            "P(clv<=0)": p0, "roi%": r["roi"], "roi_lo": r["ci_low"],
            "roi_hi": r["ci_high"], "P(roi<=0)": r["p_le_zero"]}


def venue_random(v: pd.DataFrame, bets: pd.DataFrame) -> dict:
    ren = {"odds_x": "odds_best", "clv_x": "clv_avg"}
    pool = v.drop(columns=["odds_best", "clv_avg"]).rename(columns=ren)
    r = clv_random(pool, bets.drop(columns=["odds_best", "clv_avg"]).rename(columns=ren))
    return {"random_clv%": r.get("random_clv%"),
            "beats_random_clv": r.get("beats_random_clv"),
            "random_roi%": r.get("random_roi%"),
            "beats_random_roi": r.get("beats_random_roi")}


ODDS_BANDS = (1.0, 1.6, 2.2, 3.0, 4.5, 7.0, 1000.0)


def by_odds_band(v: pd.DataFrame, bands=ODDS_BANDS) -> pd.DataFrame:
    """Value and profit side by side, per odds band."""
    v = v.assign(band=pd.cut(v["odds_x"], bands).astype(str),
                 roi=np.where(v["win"] == 1, v["odds_x"] - 1.0, -1.0))
    g = v.groupby("band", sort=False)
    out = pd.DataFrame({"n": g.size(), "clv%": g["clv_x"].mean() * 100,
                        "roi%": g["roi"].mean() * 100,
                        "roi_se": g["roi"].std() / np.sqrt(g.size()) * 100,
                        "win_rate": g["win"].mean(),
                        "closing_prob": g["q_close"].mean()})
    order = [str(b) for b in pd.cut(pd.Series([1.3]), bands).cat.categories]
    return out.reindex([b for b in order if b in out.index]).reset_index()


def benchmark_check(pool: pd.DataFrame, exec_view: pd.DataFrame, label: str,
                    bands=ODDS_BANDS, show_benchmark: bool = True) -> None:
    """
    Is the closing price a fair judge at every price? Checked BEFORE trusting CLV.

    CLV scores a bet against the de-vigged closing probability, which is only
    honest where that probability matches how often things actually happen.
    The first table answers that with no model involved at all -- every
    candidate, closing probability against realised win rate -- so whatever it
    shows cannot have been tuned to flatter the model. Where closing_prob sits
    clearly above win_rate, the closing price overrates those outcomes, and a
    model that buys them will show CLV that never turns into profit.
    """
    if show_benchmark:
      show("BENCHMARK CHECK (no model): does the closing probability match reality?",
         by_odds_band(at_venue(pool, "odds_avg", 0.0), bands),
         "closing_prob above win_rate = the closing price overrates that band;",
         "  CLV measured there is not real value.")
    show(f"The model's bets ({label}), by odds band:", by_odds_band(exec_view, bands),
         "Trust clv% only in bands where the benchmark check holds.")


def venue_report(pool: pd.DataFrame) -> None:
    rows, seasons = [], []
    for venue, col, comm in VENUES:
        v = at_venue(pool, col, comm)
        if v.empty:
            print(f"\n{venue}: no prices in this file")
            continue
        rows.append({"venue": venue, **venue_summary(v, "every outcome, no model")})
        for m in VENUE_MARGINS:
            b = v[v["pred_x"] > m]
            rows.append({"venue": venue, **venue_summary(b, f"predicted value > {m:.0%}"),
                         **(venue_random(v, b) if len(b) else {})})
        main = v[v["pred_x"] > 0.0]
        for sname, g in main.groupby("Season"):
            seasons.append({"venue": venue, **venue_summary(g, f"season {sname}")})
    show("WHERE TO BET: the same model, settled at each venue's opening price.",
         pd.DataFrame(rows),
         "clv% is value against the closing average; its interval is the steadier",
         "  evidence. roi% is what was actually won, and is far noisier.",
         "beats_random_* is the share of 300 odds-matched random draws beaten.",
         "Pinnacle covers 2023/24-2024/25 here; the exchange 2024/25-2025/26.",
         "  Each venue is judged only on the seasons it was quoted in.")
    show("Rule 'predicted value > 0%', by season and venue:", pd.DataFrame(seasons),
         "One good season and one bad is not an edge. Look for the same sign in both.")
    ex = at_venue(pool, "odds_bfe", 0.02)
    if len(ex):
        benchmark_check(pool, ex[ex["pred_x"] > 0.0],
                        "exchange, 2% commission, predicted value > 0%")


def build_clv_pool(source: str, mode: str = "multiseason", seed: int = 42) -> tuple:
    """
    Every test-season candidate of the opening-price strategy, with predictions.

    Returns (pool, tau): one row per match per outcome over every test season,
    and the coverage thresholds read off each fold's training window.
    """
    df = load_and_prepare(source) if mode == "export" else load_multiseason(source)
    full = df
    need = (STAGE1_REQUIRED + ["target"]
            + [f"{p}{o}" for p in ("Avg", "Max", "AvgC") for o in OUTCOMES])
    df = df[df[need].notna().all(axis=1)]
    df = df[(df[[f"{p}{o}" for p in ("Avg", "Max", "AvgC") for o in OUTCOMES]] > 1.0).all(axis=1)]
    df = add_opening_market(df).sort_values("Date").reset_index(drop=True)
    df["Season"] = df["Season"].astype(str)
    seasons = sorted(df["Season"].unique())

    test_frames, tau = [], {}
    for i in range(1, len(seasons)):
        te = df[df["Season"] == seasons[i]]
        win = df[df["Season"].isin(seasons[:i]) & (df["Date"] < te["Date"].min())]
        if len(win) < 2000 or len(te) < 300:
            print(f"  skipping {seasons[i]}: window {len(win)}, test {len(te)}")
            continue
        win, te = with_window_elo(full, win, te)
        t0 = time.time()
        cand_w, cand_t = clv_fold(win, te, seed)
        # Thresholds from the WINDOW's out-of-fold predictions, never the test's.
        for name in ("with stage 1", "market only"):
            for cov in CLV_COVERAGES:
                tau[(seasons[i], name, cov)] = float(
                    np.quantile(cand_w[f"pred_{name}"], 1.0 - cov))
        cand_t["fold"] = seasons[i]
        test_frames.append(cand_t)
        print(f"  fold {seasons[i]}: trained on {', '.join(seasons[:i])} "
              f"({len(win):,} matches), tested on {len(te):,}  [{time.time()-t0:.0f}s]")

    pool = pd.concat(test_frames, ignore_index=True)
    return pool, tau


def clv_main(source: str, mode: str = "multiseason", seed: int = 42,
             export: str = None):
    print("=" * 70)
    print("OPENING-PRICE STRATEGY  (bet the opening price, judge by closing-line value)")
    print(f"  learner: {LEARNER}, ensemble of {ENSEMBLE_K}")
    print("=" * 70)
    pool, tau = build_clv_pool(source, mode, seed)

    def pick(name, cov):
        if cov == "pred>0":
            return pool[pool[f"pred_{name}"] > 0]
        return pd.concat([g[g[f"pred_{name}"] >= tau[(f, name, cov)]]
                          for f, g in pool.groupby("fold")])

    rows = [clv_summary(pool, "every outcome, no model")]
    for name in ("with stage 1", "market only"):
        for cov in CLV_COVERAGES + ("pred>0",):
            lab = f"{name}, " + (f"top {cov:.0%}" if cov != "pred>0" else "predicted CLV > 0")
            rows.append(clv_summary(pick(name, cov), lab))
    show("All test seasons pooled.  clv_avg% is the headline: value at the AVERAGE "
         "opening price, by the de-vigged closing price.", pd.DataFrame(rows),
         "clv_best% / roi_best% are at the BEST opening price -- optimistic, "
         "stale quotes live there.",
         "roi has a wide interval; clv is the steadier measure. Positive CLV "
         "whose interval clears zero is the signal.")

    main_bets = pick("with stage 1", 0.10)
    show("Top 10% (with stage 1), by test season:",
         pd.DataFrame([clv_summary(g, s) for s, g in main_bets.groupby("Season")]))
    show("Top 10% (with stage 1), by outcome:",
         pd.DataFrame([clv_summary(g, o) for o, g in main_bets.groupby("outcome")]))
    by_div = pd.DataFrame([clv_summary(g, d) for d, g in main_bets.groupby("Div")])
    show("Top 10% (with stage 1), by division (small n is noise -- do NOT pick "
         "leagues from this table):", by_div.sort_values("clv_avg%", ascending=False))
    show("Against random picks at the same opening odds (top 10%):",
         pd.DataFrame([{"set": n, **clv_random(pool, pick(n, 0.10))}
                       for n in ("with stage 1", "market only")]),
         "beats_random_clv is the share of 300 draws the model's CLV beat.")

    venue_report(pool)

    if export:
        pool.to_csv(export, index=False)
        print(f"\nwrote {export}: every test candidate with its predictions")
    print("\n" + "=" * 70)
    print("Read clv_avg% and its interval first. Positive CLV at the AVERAGE")
    print("price, stable across seasons, beating random picks -- that is an edge.")
    print("CLV only at the BEST price is the stale-quote trap, not an edge.")
    print("=" * 70)
    return pool


# =========================================================================
# PART 18 -- OVER/UNDER 2.5 GOALS
#
# The same two questions as for the match result, asked of the totals market.
#
#   A. AT THE CLOSING PRICE. Does stage 1's P(over 2.5) beat the de-vigged
#      closing price? The same shrinkage test as the main pipeline: a weight
#      w fitted on out-of-fold predictions over the training window, 0 meaning
#      "trust the market", and bets on the top 10% by edge with the threshold
#      read off the window.
#
#   B. AT THE OPENING PRICE. Can the model find over/under prices the market
#      will move before kickoff? The same closing-line-value setup as PART 17,
#      the same venues and the same odds-band benchmark check.
#
# WHY THIS MARKET MIGHT DIFFER. Stage 1 is a goals model, so the totals market
# is its native question: P(over 2.5) comes straight off the scoreline grid,
# with no draw to get wrong. The market is two-way with prices between about
# 1.5 and 3, so the longshot distortion that sank the exchange result in
# PART 17 has little room to operate. Neither is a reason to expect an edge;
# both are reasons this test is not a repeat of the last one.
# =========================================================================

OU_SIDES = (("over", ">2.5"), ("under", "<2.5"))
OU_BANDS = (1.0, 1.6, 1.8, 2.0, 2.2, 2.6, 100.0)


def shin_two(o1: float, o2: float) -> tuple:
    """Shin de-vigging for a two-way market -- the same model as shin_de_vig()."""
    odds = (o1, o2)
    if any(o is None or not np.isfinite(o) or o <= 1.0 for o in odds):
        return (np.nan, np.nan)
    q = np.array([1.0 / o for o in odds])
    booksum = q.sum()
    if booksum <= 1.0:
        return tuple(q / booksum)

    def implied(z):
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / booksum) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.99
    for _ in range(30):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if implied(mid).sum() > 1.0 else (lo, mid)
    p = implied((lo + hi) / 2.0)
    return tuple(p / p.sum())


def add_ou_market(df: pd.DataFrame) -> pd.DataFrame:
    """Opening and closing totals prices, de-vigged. Closing ones are labels."""
    d = df.copy()
    for tag, pre in (("open", "Avg"), ("close", "AvgC")):
        pr = np.array([shin_two(a, b) for a, b in
                       d[[f"{pre}>2.5", f"{pre}<2.5"]].to_numpy(float)])
        d[f"ou_p_{tag}_over"], d[f"ou_p_{tag}_under"] = pr[:, 0], pr[:, 1]
    for side, suf in OU_SIDES:
        d[f"ou_log_open_{side}"] = np.log(d[f"Avg{suf}"])
        d[f"ou_spread_{side}"] = d[f"Max{suf}"] / d[f"Avg{suf}"] - 1.0
    d["ou_overround_open"] = 1.0 / d["Avg>2.5"] + 1.0 / d["Avg<2.5"] - 1.0
    d["y_over"] = ((d["FTHG"] + d["FTAG"]) > 2.5).astype(int)
    return d


def p_over_from_physical(engine: "MatchEngine", phys: pd.DataFrame,
                         d: pd.DataFrame) -> np.ndarray:
    """P(over 2.5) off the scoreline grid, with each league's own draw correction."""
    h = phys["s1_goals_home"].to_numpy()
    a = phys["s1_goals_away"].to_numpy()
    out = np.empty(len(d))
    for div, idx in d.groupby("Div").indices.items():
        rho = engine.rho_by_div.get(div, engine.rho)
        out[idx] = scoreline_markets(h[idx], a[idx], rho)["p_over_2.5"].to_numpy()
    return np.clip(out, 1e-6, 1 - 1e-6)


def _two_col(p_over: np.ndarray) -> np.ndarray:
    return np.column_stack([p_over, 1.0 - p_over])


def fit_ou_weight(p_ref: np.ndarray, p_model: np.ndarray, y: np.ndarray) -> float:
    """The shrinkage weight for a two-way market, fitted exactly like w."""
    def objective(w):
        return log_loss(y, geometric_blend(_two_col(p_ref), _two_col(p_model), w)[:, 0],
                        labels=[0, 1])
    return float(minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded").x)


def ou_fold(window: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> tuple:
    """Stage 1 once; both tests read off the same predictions."""
    window = window.reset_index(drop=True)
    test = test.reset_index(drop=True)
    engine = MatchEngine(seed=seed)
    oof = engine.fit_predict_oof(window)
    engine.fit(window)
    engine.fit_rho(oof, window)
    phys_te = engine.predict_physical(test)

    W, T = window.copy(), test.copy()
    W["ou_p1_over"] = p_over_from_physical(engine, oof, W)
    T["ou_p1_over"] = p_over_from_physical(engine, phys_te, T)
    for d, ph in ((W, oof), (T, phys_te)):
        d["ou_gap_over"] = d["ou_p1_over"] - d["ou_p_open_over"]
        for c in ("s1_goals_home", "s1_goals_away"):
            d[c] = ph[c].to_numpy()

    # --- A: shrink toward the CLOSING price, fitted on the window ---------
    w = fit_ou_weight(W["ou_p_close_over"].to_numpy(), W["ou_p1_over"].to_numpy(),
                      W["y_over"].to_numpy())
    for d in (W, T):
        d["ou_pcorr_over"] = geometric_blend(_two_col(d["ou_p_close_over"].to_numpy()),
                                             _two_col(d["ou_p1_over"].to_numpy()), w)[:, 0]

    # --- B: predict value at the OPENING average price ---------------------
    feats = assert_pre_close(
        [f"ou_p_open_{s}" for s, _ in OU_SIDES] + [f"ou_log_open_{s}" for s, _ in OU_SIDES]
        + [f"ou_spread_{s}" for s, _ in OU_SIDES] + ["ou_overround_open"]
        + [c for c in CLV_CONTEXT_FEATURES if c in W.columns]
        + ["ou_p1_over", "ou_gap_over", "s1_goals_home", "s1_goals_away"])
    for side, suf in OU_SIDES:
        y = (W[f"Avg{suf}"] * W[f"ou_p_close_{side}"] - 1.0).to_numpy(float)
        Xw, Xt = W[feats].to_numpy(float), T[feats].to_numpy(float)
        W[f"ou_pred_{side}"] = cross_val_predict(make_regressor(seed), Xw, y, cv=3)
        T[f"ou_pred_{side}"] = make_regressor(seed).fit(Xw, y).predict(Xt)
    return W, T, w


def ou_candidates(d: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """One row per match per side, in the shape PART 17's helpers read."""
    frames = []
    for side, suf in OU_SIDES:
        p_side = d[prob_col] if side == "over" else 1.0 - d[prob_col]
        frames.append(pd.DataFrame({
            "match_key": d["match_key"].to_numpy(), "Date": d["Date"].to_numpy(),
            "Season": d["Season"].astype(str).to_numpy(), "Div": d["Div"].to_numpy(),
            "outcome": side,
            "odds_avg": d[f"Avg{suf}"].to_numpy(float),
            "odds_best": d[f"Max{suf}"].to_numpy(float),
            "odds_close_avg": d[f"AvgC{suf}"].to_numpy(float),
            "odds_close_best": d[f"MaxC{suf}"].to_numpy(float),
            "odds_ps": _col(d, f"P{suf}"), "odds_bfe": _col(d, f"BFE{suf}"),
            "q_close": d[f"ou_p_close_{side}"].to_numpy(float),
            "clv_avg": (d[f"Avg{suf}"] * d[f"ou_p_close_{side}"] - 1.0).to_numpy(float),
            "clv_best": (d[f"Max{suf}"] * d[f"ou_p_close_{side}"] - 1.0).to_numpy(float),
            "edge": (p_side - d[f"ou_p_close_{side}"]).to_numpy(float),
            "win": (d["y_over"].to_numpy() == (1 if side == "over" else 0)).astype(int),
            "pred_with stage 1": d[f"ou_pred_{side}"].to_numpy(float),
        }))
    return pd.concat(frames, ignore_index=True)


def _settle(b: pd.DataFrame, odds_col: str, label: str) -> dict:
    if b.empty:
        return {"set": label, "n_bets": 0}
    bb = b.assign(Stake=1.0, PnL=np.where(b["win"] == 1, b[odds_col] - 1.0, -1.0))
    r = bootstrap_roi(bb, n_boot=5000)
    return {"set": label, "n_bets": len(b), "avg_odds": b[odds_col].mean(),
            "units_staked": float(len(b)), "units_won": float(bb["PnL"].sum()),
            "win_rate": b["win"].mean(), "roi%": r["roi"], "roi_lo": r["ci_low"],
            "roi_hi": r["ci_high"], "P(roi<=0)": r["p_le_zero"]}


def ou_main(source: str, mode: str = "multiseason", seed: int = 42):
    print("=" * 70)
    print("OVER/UNDER 2.5 GOALS")
    print(f"  learner: {LEARNER}, ensemble of {ENSEMBLE_K}")
    print("=" * 70)
    df = load_and_prepare(source) if mode == "export" else load_multiseason(source)
    full = df
    price = [f"{p}{s}" for p in ("Avg", "Max", "AvgC", "MaxC") for s in (">2.5", "<2.5")]
    missing = [c for c in price if c not in df.columns]
    if missing:
        raise KeyError(f"over/under prices missing from this file: {missing}")
    df = df[df[STAGE1_REQUIRED + ["FTHG", "FTAG"] + price].notna().all(axis=1)]
    df = df[(df[price] > 1.0).all(axis=1)]
    df = add_ou_market(df).sort_values("Date").reset_index(drop=True)
    df["Season"] = df["Season"].astype(str)
    seasons = sorted(df["Season"].unique())

    fold_rows, tests = [], []
    for i in range(1, len(seasons)):
        te = df[df["Season"] == seasons[i]]
        win = df[df["Season"].isin(seasons[:i]) & (df["Date"] < te["Date"].min())]
        if len(win) < 2000 or len(te) < 300:
            continue
        win, te = with_window_elo(full, win, te)
        t0 = time.time()
        W, T, w = ou_fold(win, te, seed)
        y = T["y_over"].to_numpy()
        # A's arm: the corrected one while it still differs from the market,
        # the raw stage 1 once w has collapsed -- the main pipeline's rule.
        arm = "ou_pcorr_over" if w >= W_COLLAPSE_LEVEL else "ou_p1_over"
        cw = ou_candidates(W, arm)
        tau = float(np.quantile(cw["edge"], 0.90))           # from the WINDOW
        ct = ou_candidates(T, arm)
        ct["chosen_A"] = ct["edge"] >= tau
        tests.append(ct)
        fold_rows.append({
            "test": seasons[i], "n_window": len(W), "n_test": len(T),
            "w": w, "arm_A": "corrected" if arm == "ou_pcorr_over" else "stage 1 raw",
            "ll_market": log_loss(y, T["ou_p_close_over"], labels=[0, 1]),
            "ll_stage1": log_loss(y, T["ou_p1_over"], labels=[0, 1]),
            "ll_corrected": log_loss(y, T["ou_pcorr_over"], labels=[0, 1]),
            "over_rate": y.mean(), "seconds": round(time.time() - t0)})
        print(f"  fold {seasons[i]} done [{fold_rows[-1]['seconds']}s]")

    folds = pd.DataFrame(fold_rows)
    pool = pd.concat(tests, ignore_index=True)

    # ---------------- A ----------------
    show("A. CLOSING PRICE -- does stage 1 beat the market on over/under?", folds,
         "log loss, lower is better. w = 0 means trust the market completely.")
    chosen = pool[pool["chosen_A"]]
    rows = [_settle(pool, "odds_close_best", "every side, no model, best closing"),
            _settle(chosen, "odds_close_best", "top 10% by edge, best closing price"),
            _settle(chosen, "odds_close_avg", "same bets, average closing price")]
    show("A. Betting the top 10% by edge at the closing price:", pd.DataFrame(rows))
    r = clv_random(pool.assign(clv_avg=0.0, odds_best=pool["odds_close_best"]),
                   chosen.assign(clv_avg=0.0, odds_best=chosen["odds_close_best"]))
    if r:
        print(f"  odds-matched random picks: {r['random_roi%']:+.2f}%, "
              f"model beat {r['beats_random_roi']:.0%} of 300 draws")
    show("A. By test season:", pd.DataFrame(
        [_settle(g, "odds_close_best", s) for s, g in chosen.groupby("Season")]))

    # ---------------- B ----------------
    rows, seasons_tab = [], []
    for venue, col, comm in VENUES:
        v = at_venue(pool, col, comm)
        if v.empty:
            continue
        rows.append({"venue": venue, **venue_summary(v, "every side, no model")})
        for m in VENUE_MARGINS:
            b = v[v["pred_x"] > m]
            rows.append({"venue": venue, **venue_summary(b, f"predicted value > {m:.0%}"),
                         **(venue_random(v, b) if len(b) else {})})
        for sname, g in v[v["pred_x"] > 0.0].groupby("Season"):
            seasons_tab.append({"venue": venue, **venue_summary(g, f"season {sname}")})
    show("B. OPENING PRICE -- value against the closing price, by venue:",
         pd.DataFrame(rows),
         "clv% is value against the de-vigged closing average; roi% is what was won.",
         "Pinnacle covers 2023/24-2024/25 here; the exchange 2024/25-2025/26.")
    show("B. Rule 'predicted value > 0%', by season and venue:", pd.DataFrame(seasons_tab))
    first = True
    for venue, col, comm in (("Pinnacle", "odds_ps", 0.0),
                             ("exchange, 2% commission", "odds_bfe", 0.02)):
        v = at_venue(pool, col, comm)
        if len(v):
            benchmark_check(pool, v[v["pred_x"] > 0.0],
                            f"{venue}, predicted value > 0%", OU_BANDS,
                            show_benchmark=first)
            first = False

    print("\n" + "=" * 70)
    print("An edge here needs ALL of: positive value whose interval clears zero,")
    print("the same sign in every season, profit consistent with that value,")
    print("beating odds-matched random picks, and a benchmark check that holds.")
    print("=" * 70)
    return folds, pool


# =========================================================================
# PART 19 -- FEATURES BUILT INSIDE THE PIPELINE: Elo and last season's table
#
# These used to come from separate scripts (build_elo.py, build_history.py).
# Built here, they need no extra files and -- for Elo -- can be fitted inside
# each fold, which the separate script could not guarantee.
#
# LAST SEASON'S TABLE has nothing fitted in it: points, goals and tier from a
# season that is over before the current one starts. It can be built once for
# the whole file without leaking anything.
#
# ELO has six fitted constants (PART 14). Fitted on every season, they would
# carry a little of the test seasons into the features. So they are fitted on
# each fold's training window only, and the walk is then run over the whole
# file with those constants. Running it past the window is not a leak: each
# match still gets the ratings its teams carried INTO it, built only from
# matches already played.
# =========================================================================

BUILT_IN_PIPELINE = {"elo/elo_by_match.csv", "history/prev_season.csv",
                     "history/xgproxy_features.csv"}


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Each side's record in the previous season, and how its tier changed."""
    d = add_country(df).copy()
    d["Season"] = d["Season"].astype(str)
    tier = d["Div"].map(lambda x: TIERS.get(x, ("?", 0))[1]).astype(float)
    played = d[d["FTR"].isin(["H", "D", "A"])]
    pts_h = np.select([played["FTR"] == "H", played["FTR"] == "D"], [3, 1], 0)
    pts_a = np.select([played["FTR"] == "A", played["FTR"] == "D"], [3, 1], 0)
    long = pd.concat([
        pd.DataFrame({"Season": played["Season"],
                      "team": played["Country"] + "|" + played["HomeTeam"],
                      "gf": played["FTHG"], "ga": played["FTAG"], "pts": pts_h,
                      "tier": tier[played.index]}),
        pd.DataFrame({"Season": played["Season"],
                      "team": played["Country"] + "|" + played["AwayTeam"],
                      "gf": played["FTAG"], "ga": played["FTHG"], "pts": pts_a,
                      "tier": tier[played.index]})])
    tab = long.groupby(["Season", "team"]).agg(
        prev_ppg=("pts", "mean"), prev_gf_pg=("gf", "mean"),
        prev_ga_pg=("ga", "mean"), prev_tier=("tier", lambda x: x.mode().iloc[0]))
    tab["prev_gd_pg"] = tab["prev_gf_pg"] - tab["prev_ga_pg"]

    seasons = sorted(d["Season"].unique())
    nxt = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}
    tab = tab.reset_index()
    tab["Season"] = tab["Season"].map(nxt)          # last season -> this season
    tab = tab.dropna(subset=["Season"])

    out = d.copy()
    stats = ["prev_ppg", "prev_gd_pg", "prev_gf_pg", "prev_ga_pg", "prev_tier"]
    first = out["Season"] == seasons[0]             # no earlier season on file
    for side, col in (("home", "HomeTeam"), ("away", "AwayTeam")):
        key = out["Country"] + "|" + out[col]
        m = pd.DataFrame({"Season": out["Season"], "team": key}).merge(
            tab, on=["Season", "team"], how="left")
        for st in stats:
            out[f"{side}_{st}"] = m[st].to_numpy()
        found = m["prev_ppg"].notna().to_numpy()
        change = tier.to_numpy() - m["prev_tier"].to_numpy()
        out[f"{side}_tier_change"] = change           # < 0: moved up a division
        out[f"{side}_is_promoted"] = np.where(found, (change < 0).astype(float), np.nan)
        # "New" means absent from a season we HAVE. In the first season on file
        # nobody has a previous season, which says nothing about the team.
        out[f"{side}_is_new"] = np.where(first, np.nan, (~found).astype(float))
    print(f"  + previous season: built in-pipeline, "
          f"{out['home_prev_ppg'].notna().mean():.0%} coverage")
    return out


_ELO_CACHE = {}


def fold_elo(df_all: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    """Elo for every match, with its constants fitted on `window` only."""
    key = (len(window), str(window["Date"].min()), str(window["Date"].max()))
    if key not in _ELO_CACHE:
        t0 = time.time()
        fit_on = window[window["FTR"].isin(["H", "D", "A"])]
        params, loss = fit_elo(fit_on)
        # With one season in the window there is no summer to learn from, so
        # `carry` is not identified -- the fit is flat in it and drifts to a
        # bound. Zero would wipe every rating before the test season; hold it
        # at the textbook value instead of letting noise choose.
        if fit_on["Season"].astype(str).nunique() < 2:
            params = EloParams(**{**params.__dict__, "carry": EloParams().carry})
        tab = run_elo(df_all, params)
        tab["match_key"] = df_all["match_key"].to_numpy()
        tab = tab.drop_duplicates("match_key")
        coefs = fit_pxg(fit_on)                       # proxy xG, same window
        px = pxg_features(df_all, coefs).drop_duplicates("match_key")
        tab = tab.merge(px, on="match_key", how="left")
        if coefs:
            print(f"  proxy xG fitted on the same window: log rate = {coefs[0]:+.3f} "
                  f"{coefs[1]:+.4f} x shots {coefs[2]:+.4f} x on target", flush=True)
        _ELO_CACHE[key] = tab
        print(f"  elo fitted on {len(fit_on):,} matches to "
              f"{pd.Timestamp(window['Date'].max()).date()}: k={params.k:.1f} "
              f"home={params.home_adv:.0f} gamma={params.gamma:.2f} "
              f"carry={params.carry:.2f}  (log loss {loss:.4f}, "
              f"{time.time() - t0:.0f}s)", flush=True)
    return _ELO_CACHE[key]


# --- proxy xG, fitted per fold like Elo --------------------------------------
#
# The files carry shots and shots on target but no expected goals (real xG only
# starts in 2026/27). So expected goals are estimated from them: a Poisson
# regression of a side's goals on its shots and shots on target, fitted on the
# training window only. Each team-match then gets the goals its chances were
# worth, and the features are rolling means over the team's previous 5 and 10
# matches, home and away together:
#
#   pxg_for     chances created          pxg_against   chances conceded
#   pxg_diff    the two together         pxg_luck      goals minus pxg_for: how
#                                                      far finishing ran ahead of
#                                                      (or behind) the chances
#
# Divisions without shot data (the National League) simply get no values.

PXG_SPANS = (5, 10)


def fit_pxg(window: pd.DataFrame) -> tuple:
    """Poisson regression of goals on shots and shots on target (training only)."""
    from sklearn.linear_model import PoissonRegressor
    w = window[window["FTR"].isin(["H", "D", "A"])]
    X = np.vstack([w[["HS", "HST"]].to_numpy(float), w[["AS", "AST"]].to_numpy(float)])
    y = np.concatenate([w["FTHG"].to_numpy(float), w["FTAG"].to_numpy(float)])
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if ok.sum() < 500:
        return None
    m = PoissonRegressor(alpha=1e-6, max_iter=500).fit(X[ok], y[ok])
    return float(m.intercept_), float(m.coef_[0]), float(m.coef_[1])


def pxg_features(df_all: pd.DataFrame, coefs: tuple) -> pd.DataFrame:
    """Rolling proxy-xG features for every match, from previous matches only."""
    if coefs is None or not all(c in df_all.columns for c in ("HS", "AS", "HST", "AST")):
        return pd.DataFrame({"match_key": df_all["match_key"].to_numpy()})
    b0, b1, b2 = coefs
    d = add_country(df_all)

    def rate(shots, sot):
        return np.exp(b0 + b1 * shots + b2 * sot)

    hx = rate(d["HS"].to_numpy(float), d["HST"].to_numpy(float))
    ax = rate(d["AS"].to_numpy(float), d["AST"].to_numpy(float))
    hg = pd.to_numeric(d["FTHG"], errors="coerce").to_numpy(float)
    ag = pd.to_numeric(d["FTAG"], errors="coerce").to_numpy(float)
    long = pd.concat([
        pd.DataFrame({"row": np.arange(len(d)), "side": "home",
                      "team": (d["Country"] + "|" + d["HomeTeam"]).to_numpy(),
                      "Date": d["Date"].to_numpy(), "xf": hx, "xa": ax, "g": hg}),
        pd.DataFrame({"row": np.arange(len(d)), "side": "away",
                      "team": (d["Country"] + "|" + d["AwayTeam"]).to_numpy(),
                      "Date": d["Date"].to_numpy(), "xf": ax, "xa": hx, "g": ag})],
        ignore_index=True).sort_values(["team", "Date", "row"], kind="mergesort")
    long["luck"] = long["g"] - long["xf"]
    g = long.groupby("team", sort=False)
    for span in PXG_SPANS:
        for src, name in (("xf", "for"), ("xa", "against"), ("luck", "luck")):
            # shift(1): a match never sees its own shots or goals
            long[f"pxg_{name}_{span}"] = g[src].transform(
                lambda x: x.shift(1).rolling(span, min_periods=1).mean())
        long[f"pxg_diff_{span}"] = long[f"pxg_for_{span}"] - long[f"pxg_against_{span}"]
    feats = [f"pxg_{n}_{s}" for s in PXG_SPANS for n in ("for", "against", "diff", "luck")]
    out = pd.DataFrame({"match_key": d["match_key"].to_numpy()})
    for side in ("home", "away"):
        part = long[long["side"] == side].set_index("row").sort_index()
        for f in feats:
            out[f"{side}_{f}"] = part[f].reindex(np.arange(len(d))).to_numpy()
    return out


def put_elo(frame: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    """Attach the per-fold columns (Elo, proxy xG) to a frame by match."""
    cols = [c for c in tab.columns if c != "match_key"]
    base = frame.drop(columns=[c for c in cols if c in frame.columns])
    m = base[["match_key"]].merge(tab, on="match_key", how="left")
    for c in cols:
        base[c] = m[c].to_numpy()
    return base


def _elo_external(df: pd.DataFrame) -> bool:
    return "elo_home" in df.columns and df["elo_home"].notna().any()


def with_fold_elo(df_all: pd.DataFrame, fold: "Fold") -> "Fold":
    """The main walk-forward's hook: Elo fitted on this fold's train + val."""
    if _elo_external(df_all):
        return fold
    tab = fold_elo(df_all, pd.concat([fold.train, fold.val]))
    return fold._replace(train=put_elo(fold.train, tab), val=put_elo(fold.val, tab),
                         test=put_elo(fold.test, tab))


def with_window_elo(df_all: pd.DataFrame, window: pd.DataFrame,
                    test: pd.DataFrame) -> tuple:
    """The same hook for the clv and ou tests."""
    if _elo_external(df_all):
        return window, test
    tab = fold_elo(df_all, window)
    return put_elo(window, tab), put_elo(test, tab)


# =========================================================================
# PART 20 -- THE TRAPS, MEASURED
#
# Two mistakes that make a backtest lie, run deliberately so their size is
# visible instead of argued about.
#
# TRAP 1: CHOOSING THE RULE ON THE TEST SEASONS. Search enough combinations of
# league, odds range, outcome and threshold on the data you then report, and
# the best one will look profitable whether or not anything is there. The
# experiment runs that search twice -- on the model's score, and on a RANDOM
# score that knows nothing -- and then does it honestly: the combination is
# chosen on one season and applied, unchanged, to the next.
#
# TRAP 2: CLOSING ODDS AS AN INPUT. The closing price is what the opening-price
# strategy is trying to anticipate. Give it to the model and the model simply
# reads the answer. The experiment scores bets with the closing value itself --
# what any learner given the closing price converges to -- and then checks that
# the pipeline's guards refuse such inputs.
#
# Every number here comes from the opening-price candidate pool of PART 17,
# settled at the best opening price, flat stakes.
# =========================================================================

TRAP_ODDS = [(lo, hi) for lo in (1.0, 1.5, 2.0, 2.5, 3.0)
             for hi in (2.0, 2.5, 3.0, 4.0, 6.0, 1000.0) if hi > lo]
TRAP_OUTCOMES = {"H": ("H",), "D": ("D",), "A": ("A",), "H+A": ("H", "A"),
                 "all": ("H", "D", "A")}
TRAP_TOP = (0.05, 0.10, 0.20, 0.50, 1.0)


def _profit(pool: pd.DataFrame) -> np.ndarray:
    return np.where(pool["win"].to_numpy() == 1, pool["odds_best"].to_numpy() - 1.0, -1.0)


def snoop_grid(pool: pd.DataFrame, score: np.ndarray, min_bets: int = 100) -> pd.DataFrame:
    """
    Every combination of league x odds range x outcome x top-fraction, scored on
    `pool` itself. The threshold for each is the score quantile INSIDE that
    combination, recorded so it can be carried to another season unchanged.
    """
    profit = _profit(pool)
    odds = pool["odds_best"].to_numpy()
    div, out = pool["Div"].to_numpy(), pool["outcome"].to_numpy()
    leagues = {"all": np.ones(len(pool), bool)}
    leagues.update({d: div == d for d in sorted(pool["Div"].unique())})
    outs = {k: np.isin(out, v) for k, v in TRAP_OUTCOMES.items()}
    rows = []
    for lg, m_lg in leagues.items():
        for (lo, hi) in TRAP_ODDS:
            m_odds = m_lg & (odds > lo) & (odds <= hi)
            if m_odds.sum() < min_bets:
                continue
            for ok, m_out in outs.items():
                m = m_odds & m_out
                if m.sum() < min_bets:
                    continue
                sc = score[m]
                for top in TRAP_TOP:
                    thr = -np.inf if top >= 1.0 else float(np.quantile(sc, 1.0 - top))
                    sel = sc >= thr
                    n = int(sel.sum())
                    if n < min_bets:
                        continue
                    rows.append({"league": lg, "odds_lo": lo, "odds_hi": hi,
                                 "outcome": ok, "top": top, "thr": thr, "n": n,
                                 "roi%": profit[m][sel].mean() * 100})
    return pd.DataFrame(rows)


def apply_config(pool: pd.DataFrame, score: np.ndarray, cfg) -> np.ndarray:
    """A combination chosen elsewhere, applied to `pool` with its threshold fixed."""
    m = ((pool["odds_best"] > cfg["odds_lo"]) & (pool["odds_best"] <= cfg["odds_hi"])
         & pool["outcome"].isin(TRAP_OUTCOMES[cfg["outcome"]])).to_numpy().copy()
    if cfg["league"] != "all":
        m = m & (pool["Div"] == cfg["league"]).to_numpy()
    return m & (score >= cfg["thr"])


def _naive_p(profit: np.ndarray, n_boot: int = 4000, seed: int = 0) -> float:
    """The p-value the winning combination would report if you forgot the search."""
    idx = np.random.default_rng(seed).integers(0, len(profit), (n_boot, len(profit)))
    return float((profit[idx].mean(1) <= 0).mean())


def trap_selection(pool: pd.DataFrame, scores: dict) -> None:
    seasons = sorted(pool["Season"].astype(str).unique())
    profit = _profit(pool)
    base = profit.mean() * 100

    rows = []
    for name, score in scores.items():
        g = snoop_grid(pool, score)
        best = g.sort_values("roi%", ascending=False).iloc[0]
        sel = apply_config(pool, score, best)
        rows.append({
            "score": name, "combinations": len(g),
            "share_roi>0": (g["roi%"] > 0).mean(),
            "share_roi>+10%": (g["roi%"] > 10).mean(),
            "best_roi%": best["roi%"], "best_n": int(best["n"]),
            "naive_p(roi<=0)": _naive_p(profit[sel]),
            "best_rule": f"{best['league']}, odds {best['odds_lo']}-{best['odds_hi']}, "
                         f"{best['outcome']}, top {best['top']:.0%}"})
    show("TRAP 1a. Search the test seasons for the best rule (>= 100 bets each).",
         pd.DataFrame(rows),
         f"Every outcome, no selection at all: {base:+.2f}%.",
         "naive_p is what a bootstrap of the winning rule's own bets reports.",
         "A random score with no information finds a 'profitable' rule too --",
         "  so finding one is not evidence of anything.")

    # Honest version: choose on one season, apply unchanged to the next.
    rows = []
    for name, score in scores.items():
        for a, b in zip(seasons[:-1], seasons[1:]):
            ma = (pool["Season"].astype(str) == a).to_numpy()
            mb = (pool["Season"].astype(str) == b).to_numpy()
            pa, pb = pool[ma].reset_index(drop=True), pool[mb].reset_index(drop=True)
            g = snoop_grid(pa, score[ma], min_bets=50).sort_values("roi%", ascending=False)
            if g.empty:
                continue
            out_best = apply_config(pb, score[mb], g.iloc[0])
            prof_b = _profit(pb)
            top20 = [prof_b[apply_config(pb, score[mb], c)].mean() * 100
                     for _, c in g.head(20).iterrows()
                     if apply_config(pb, score[mb], c).sum() > 0]
            rows.append({"score": name, "chosen_on": a, "applied_to": b,
                         "in_sample_roi%": g.iloc[0]["roi%"],
                         "next_season_roi%": prof_b[out_best].mean() * 100
                         if out_best.any() else np.nan,
                         "next_season_n": int(out_best.sum()),
                         "top20_rules_next_season_mean%": np.mean(top20) if top20 else np.nan})
    show("TRAP 1b. The same search done honestly: chosen on one season, applied to the next.",
         pd.DataFrame(rows),
         "in_sample_roi% is what the snooped backtest would have printed;",
         "  next_season_roi% is what the same rule actually did a season later.")

    # The old pipeline's symptom: rerun the random search, get a different winner.
    rows = []
    for seed in range(5):
        noise = np.random.default_rng(100 + seed).random(len(pool))
        best = snoop_grid(pool, noise).sort_values("roi%", ascending=False).iloc[0]
        rows.append({"random_seed": seed, "best_roi%": best["roi%"], "n": int(best["n"]),
                     "rule": f"{best['league']}, odds {best['odds_lo']}-{best['odds_hi']}, "
                             f"{best['outcome']}, top {best['top']:.0%}"})
    show("TRAP 1c. Five random scores, five 'best' rules:", pd.DataFrame(rows),
         "Always strongly positive, never the same rule. This is how a result can",
         "  flip sign between seeds: the search, not the model, made the number.")


def trap_closing(pool: pd.DataFrame) -> None:
    leaky = (pool["odds_best"] * pool["q_close"] - 1.0).to_numpy()
    honest = pool["pred_with stage 1"].to_numpy()
    profit = _profit(pool)
    rows = []
    for name, score in (("honest model (opening data only)", honest),
                        ("leaky model (knows the closing price)", leaky)):
        sel = np.concatenate([
            np.flatnonzero(m)[score[m] >= np.quantile(score[m], 0.90)]
            for m in (pool["Season"] == s for s in pool["Season"].unique())])
        b = pool.iloc[sel]
        rows.append({"model": name, "n_bets": len(b),
                     "clv_at_best%": b["clv_best"].mean() * 100,
                     "roi_at_best%": profit[sel].mean() * 100,
                     "naive_p(roi<=0)": _naive_p(profit[sel])})
    show("TRAP 2. Closing odds as an input -- top 10% per season, bet at the opening price.",
         pd.DataFrame(rows),
         "The leaky row is what any model converges to once the closing price is",
         "  in its inputs. It could never be bet: at the opening moment that",
         "  price does not exist yet.")

    print("\n  Do the guards catch it?")
    for label, fn, feats in (
            ("stage 1 guard, closing average", assert_market_blind, ["AvgCH"]),
            ("stage 1 guard, de-vigged closing", assert_market_blind, ["p_ref_H"]),
            ("opening-model guard, closing prob", assert_pre_close, ["q_close_H"]),
            ("opening-model guard, Pinnacle close", assert_pre_close, ["PSCH"]),
            ("opening-model guard, price drift", assert_pre_close, ["drift_H"])):
        try:
            fn(feats)
            print(f"    {label:38s} NOT CAUGHT  <-- fix the guard")
        except ValueError:
            print(f"    {label:38s} caught")
    for label, fn, feats in (("stage 1 features as shipped", assert_market_blind,
                              STAGE1_FEATURES),
                             ("opening-model features as shipped", assert_pre_close,
                              OPEN_MARKET_FEATURES + CLV_CONTEXT_FEATURES)):
        try:
            fn(feats)
            print(f"    {label:38s} clean")
        except ValueError as exc:
            print(f"    {label:38s} LEAK: {exc}")


def traps_main(source: str, mode: str = "multiseason", seed: int = 42):
    print("=" * 70)
    print("THE TRAPS, MEASURED")
    print("=" * 70)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pool, _ = build_clv_pool(source, mode, seed)
    pool = pool.reset_index(drop=True)
    pool["Season"] = pool["Season"].astype(str)
    print(f"{len(pool):,} candidate bets over seasons "
          f"{', '.join(sorted(pool['Season'].unique()))}")
    scores = {"model (stage 1 + opening prices)": pool["pred_with stage 1"].to_numpy(),
              "random noise": np.random.default_rng(seed).random(len(pool))}
    trap_selection(pool, scores)
    trap_closing(pool)
    print("\n" + "=" * 70)
    print("The only numbers worth reporting are ones whose rule was fixed before")
    print("the seasons it is judged on, using inputs that existed at bet time.")
    print("=" * 70)


# =========================================================================
# PART 21 -- ASIAN HANDICAP
#
# The third market, tested the same two ways as over/under (PART 18):
#   A. at the CLOSING price -- does stage 1 beat the market's handicap price?
#   B. at the OPENING price -- can it find handicap prices the market will
#      move, and does that survive the margin at Pinnacle or the exchange?
#
# WHY THIS MARKET. It carries the lowest margins in the file, at Pinnacle and
# on the exchange alike, so a small skill at predicting price moves has the
# best chance of clearing them here.
#
# SETTLEMENT FOLLOWS THE REAL RULES, because a backtest that treats every
# handicap as win-or-lose is simply wrong on most of the bets:
#   * a whole line (0, -1, ...) can PUSH: the stake comes back
#   * a half line (-0.5, -1.5, ...) cannot push
#   * a quarter line (-0.25, -0.75, ...) is two half-stakes on the lines either
#     side, so a bet can half-win or half-lose
# Every bet is settled as  win_frac * (odds - 1) - lose_frac,  with each
# fraction 0, 1/2 or 1. The model's own view is computed the same way, from
# the full goal-difference distribution off stage 1's scoreline grid.
#
# THE LINE MOVES. In a large share of matches the handicap line at kickoff is
# not the line that opened, and then the closing price is for a different bet.
# So value against the close (CLV) is only measured, and only trained on, where
# the line stayed put. Profit is measured on every bet, settled at the line
# actually taken -- you cannot know at the opening moment whether it will move.
# =========================================================================

AH_SIDES = ("H", "A")
AH_D = np.arange(-MAX_GOALS, MAX_GOALS + 1)
AH_BANDS = (1.0, 1.7, 1.85, 2.0, 2.15, 2.3, 100.0)


def goal_diff_probs(lam_home, lam_away, rho: float) -> np.ndarray:
    """P(home goals - away goals = d) for d in AH_D, off the scoreline grid."""
    joint = scoreline_grid(lam_home, lam_away, rho)
    G = joint.shape[1]
    out = np.zeros((len(joint), len(AH_D)))
    for i in range(G):
        for j in range(G):
            out[:, i - j + MAX_GOALS] += joint[:, i, j]
    return out


def ah_fractions(d_probs: np.ndarray, line: np.ndarray, side: str) -> tuple:
    """
    Expected (win_frac, lose_frac) of a one-unit handicap bet.

    `line` is the HOME handicap. A home bet wins by d + line > 0; an away bet
    wins by -(d + line) > 0. Quarter lines are split into two half-stakes.
    """
    line = np.asarray(line, float)
    quarter = np.isclose(np.abs(line) % 0.5, 0.25)
    w = np.zeros(len(line))
    l = np.zeros(len(line))
    for shift in (-0.25, 0.25):
        sub = np.where(quarter, line + shift, line)
        m = AH_D[None, :] + sub[:, None]
        if side == "A":
            m = -m
        w += 0.5 * (d_probs * (m > 1e-9)).sum(1)
        l += 0.5 * (d_probs * (m < -1e-9)).sum(1)
    return w, l


def ah_settle(diff: np.ndarray, line: np.ndarray, side: str) -> tuple:
    """The same rule applied to the actual result: (win_frac, lose_frac)."""
    onehot = (AH_D[None, :] == np.asarray(diff)[:, None]).astype(float)
    return ah_fractions(onehot, line, side)


def _push_free(w: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Share of the non-refunded stake that wins: comparable to a de-vigged price."""
    return np.where(w + l > 0, w / np.maximum(w + l, 1e-12), 0.5)


AH_PRICE_COLS = ["AHh", "AHCh", "AvgAHH", "AvgAHA", "MaxAHH", "MaxAHA",
                 "AvgCAHH", "AvgCAHA", "MaxCAHH", "MaxCAHA"]


def add_ah_market(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for tag, pre in (("open", "AvgAH"), ("close", "AvgCAH")):
        pr = np.array([shin_two(a, b) for a, b in d[[f"{pre}H", f"{pre}A"]].to_numpy(float)])
        d[f"ah_p_{tag}_H"], d[f"ah_p_{tag}_A"] = pr[:, 0], pr[:, 1]
    for sd in AH_SIDES:
        d[f"ah_log_open_{sd}"] = np.log(d[f"AvgAH{sd}"])
        d[f"ah_spread_{sd}"] = d[f"MaxAH{sd}"] / d[f"AvgAH{sd}"] - 1.0
    d["ah_overround_open"] = 1.0 / d["AvgAHH"] + 1.0 / d["AvgAHA"] - 1.0
    d["ah_line_open"] = d["AHh"]                     # known at the opening moment
    d["ah_same_line"] = np.isclose(d["AHh"], d["AHCh"])
    d["goal_diff"] = (d["FTHG"] - d["FTAG"]).astype(int)
    return d


def _ah_model_view(engine, phys, d: pd.DataFrame) -> pd.DataFrame:
    """The model's push-free win share for each side, at the opening and closing lines."""
    h = phys["s1_goals_home"].to_numpy()
    a = phys["s1_goals_away"].to_numpy()
    dp = np.empty((len(d), len(AH_D)))
    for div, idx in d.groupby("Div").indices.items():
        dp[idx] = goal_diff_probs(h[idx], a[idx], engine.rho_by_div.get(div, engine.rho))
    out = {}
    for tag, col in (("open", "AHh"), ("close", "AHCh")):
        for sd in AH_SIDES:
            w, l = ah_fractions(dp, d[col].to_numpy(float), sd)
            out[f"ah_m_{tag}_{sd}"] = _push_free(w, l)
    return pd.DataFrame(out, index=d.index)


def ah_fold(window: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> tuple:
    window = window.reset_index(drop=True)
    test = test.reset_index(drop=True)
    engine = MatchEngine(seed=seed)
    oof = engine.fit_predict_oof(window)
    engine.fit(window)
    engine.fit_rho(oof, window)
    phys_te = engine.predict_physical(test)
    W = pd.concat([window, _ah_model_view(engine, oof, window)], axis=1)
    T = pd.concat([test, _ah_model_view(engine, phys_te, test)], axis=1)
    for d, ph in ((W, oof), (T, phys_te)):
        for sd in AH_SIDES:
            d[f"ah_gap_{sd}"] = d[f"ah_m_open_{sd}"] - d[f"ah_p_open_{sd}"]
        for c in ("s1_goals_home", "s1_goals_away"):
            d[c] = ph[c].to_numpy()

    feats = assert_pre_close(
        [f"ah_p_open_{sd}" for sd in AH_SIDES] + [f"ah_log_open_{sd}" for sd in AH_SIDES]
        + [f"ah_spread_{sd}" for sd in AH_SIDES] + ["ah_overround_open", "ah_line_open"]
        + [c for c in CLV_CONTEXT_FEATURES if c in W.columns]
        + [f"ah_m_open_{sd}" for sd in AH_SIDES] + [f"ah_gap_{sd}" for sd in AH_SIDES]
        + ["s1_goals_home", "s1_goals_away"])
    same = W["ah_same_line"].to_numpy()
    for sd in AH_SIDES:
        y = (W[f"AvgAH{sd}"] * W[f"ah_p_close_{sd}"] - 1.0).to_numpy(float)
        model = make_regressor(seed).fit(W.loc[same, feats].to_numpy(float), y[same])
        T[f"ah_pred_{sd}"] = model.predict(T[feats].to_numpy(float))
        W[f"ah_pred_{sd}"] = np.nan                  # never used for selection
    return W, T


def ah_candidates(d: pd.DataFrame) -> pd.DataFrame:
    frames = []
    diff = d["goal_diff"].to_numpy()
    for sd in AH_SIDES:
        wo, lo = ah_settle(diff, d["AHh"].to_numpy(float), sd)
        wc, lc = ah_settle(diff, d["AHCh"].to_numpy(float), sd)
        same = d["ah_same_line"].to_numpy()
        frames.append(pd.DataFrame({
            "match_key": d["match_key"].to_numpy(), "Date": d["Date"].to_numpy(),
            "Season": d["Season"].astype(str).to_numpy(), "Div": d["Div"].to_numpy(),
            "outcome": sd, "same_line": same,
            "odds_avg": d[f"AvgAH{sd}"].to_numpy(float),
            "odds_best": d[f"MaxAH{sd}"].to_numpy(float),
            "odds_ps": _col(d, f"PAH{sd}"), "odds_bfe": _col(d, f"BFEAH{sd}"),
            "odds_close_avg": d[f"AvgCAH{sd}"].to_numpy(float),
            "odds_close_best": d[f"MaxCAH{sd}"].to_numpy(float),
            "q_close": np.where(same, d[f"ah_p_close_{sd}"].to_numpy(float), np.nan),
            "w_open": wo, "l_open": lo, "w_close": wc, "l_close": lc,
            "edge_close": (d[f"ah_m_close_{sd}"] - d[f"ah_p_close_{sd}"]).to_numpy(float),
            "pred": d[f"ah_pred_{sd}"].to_numpy(float),
        }))
    return pd.concat(frames, ignore_index=True)


def ah_pnl(b: pd.DataFrame, odds_col: str, when: str = "open") -> np.ndarray:
    return b[f"w_{when}"] * (b[odds_col] - 1.0) - b[f"l_{when}"]


def ah_summary(b: pd.DataFrame, odds_col: str, label: str, when: str = "open",
               clv_col: str = None) -> dict:
    if b.empty:
        return {"rule": label, "n_bets": 0}
    pnl = ah_pnl(b, odds_col, when).to_numpy()
    r = bootstrap_roi(b.assign(Stake=1.0, PnL=pnl), n_boot=5000)
    out = {"rule": label, "n_bets": len(b),
           "seasons": ",".join(sorted(b["Season"].unique())),
           "avg_odds": b[odds_col].mean(),
           "units_staked": float(len(b)), "units_won": float(pnl.sum()),
           "roi%": r["roi"], "roi_lo": r["ci_low"], "roi_hi": r["ci_high"],
           "P(roi<=0)": r["p_le_zero"]}
    if clv_col:
        c = b.dropna(subset=[clv_col])
        if len(c):
            lo, hi, p0 = _match_boot(c, clv_col)
            out.update({"clv_n": len(c), "clv%": c[clv_col].mean() * 100,
                        "clv_lo": lo, "clv_hi": hi, "P(clv<=0)": p0})
    return out


def ah_random(pool: pd.DataFrame, bets: pd.DataFrame, odds_col: str,
              when: str = "open", clv_col: str = None, n_draws: int = 300,
              seed: int = 42) -> dict:
    """Random picks from the same pool, matched on odds decile."""
    if bets.empty:
        return {}
    edges = np.unique(np.quantile(pool[odds_col], np.linspace(0, 1, 11)))
    pb = np.clip(np.digitize(pool[odds_col], edges[1:-1]), 0, len(edges) - 2)
    bb = np.clip(np.digitize(bets[odds_col], edges[1:-1]), 0, len(edges) - 2)
    by = {k: np.flatnonzero(pb == k) for k in np.unique(pb)}
    want = pd.Series(bb).value_counts()
    pnl = ah_pnl(pool, odds_col, when).to_numpy()
    clv = pool[clv_col].to_numpy() if clv_col else None
    rng = np.random.default_rng(seed)
    rr, rc = np.empty(n_draws), np.empty(n_draws)
    for i in range(n_draws):
        idx = np.concatenate([rng.choice(by[k], n, replace=len(by[k]) < n)
                              for k, n in want.items() if k in by])
        rr[i] = pnl[idx].mean() * 100
        rc[i] = np.nanmean(clv[idx]) * 100 if clv is not None else np.nan
    m_roi = ah_pnl(bets, odds_col, when).mean() * 100
    out = {"random_roi%": rr.mean(), "beats_random_roi": (rr < m_roi).mean()}
    if clv_col:
        m_clv = bets[clv_col].mean() * 100
        out.update({"random_clv%": np.nanmean(rc), "beats_random_clv": (rc < m_clv).mean()})
    return out


def ah_main(source: str, mode: str = "multiseason", seed: int = 42):
    print("=" * 70)
    print("ASIAN HANDICAP")
    print(f"  learner: {LEARNER}, ensemble of {ENSEMBLE_K}")
    print("=" * 70)
    df = load_and_prepare(source) if mode == "export" else load_multiseason(source)
    full = df
    missing = [c for c in AH_PRICE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Asian handicap columns missing from this file: {missing}")
    df = df[df[STAGE1_REQUIRED + ["FTHG", "FTAG"] + AH_PRICE_COLS].notna().all(axis=1)]
    df = df[(df[AH_PRICE_COLS[2:]] > 1.0).all(axis=1)]
    df = add_ah_market(df).sort_values("Date").reset_index(drop=True)
    df["Season"] = df["Season"].astype(str)
    seasons = sorted(df["Season"].unique())
    print(f"  opening line = closing line in {df['ah_same_line'].mean():.0%} of matches")

    tests = []
    for i in range(1, len(seasons)):
        te = df[df["Season"] == seasons[i]]
        win = df[df["Season"].isin(seasons[:i]) & (df["Date"] < te["Date"].min())]
        if len(win) < 2000 or len(te) < 300:
            continue
        win, te = with_window_elo(full, win, te)
        t0 = time.time()
        W, T = ah_fold(win, te, seed)
        cw, ct = ah_candidates(W), ah_candidates(T)
        ct["chosen_A"] = ct["edge_close"] >= float(np.quantile(cw["edge_close"], 0.90))
        tests.append(ct)
        print(f"  fold {seasons[i]} done [{time.time() - t0:.0f}s]", flush=True)
    pool = pd.concat(tests, ignore_index=True)

    # ---------------- A: the closing price ----------------
    chosen = pool[pool["chosen_A"]]
    show("A. CLOSING PRICE -- top 10% by the model's edge over the closing handicap price:",
         pd.DataFrame([
             ah_summary(pool, "odds_close_best", "every side, no model, best closing", "close"),
             {**ah_summary(chosen, "odds_close_best", "top 10% by edge, best closing", "close"),
              **ah_random(pool, chosen, "odds_close_best", "close")},
             ah_summary(chosen, "odds_close_avg", "same bets, average closing", "close")]),
         "Settled at the CLOSING line with full push / half-win rules.")
    show("A. By test season:", pd.DataFrame(
        [ah_summary(g, "odds_close_best", s, "close") for s, g in chosen.groupby("Season")]))

    # ---------------- B: the opening price ----------------
    rows, seasons_tab = [], []
    for venue, col, comm in VENUES:
        v = pool[pool[col].notna() & (pool[col] > 1.0)].copy()
        if v.empty:
            continue
        v["odds_x"] = 1.0 + (v[col] - 1.0) * (1.0 - comm)
        v["pred_x"] = v["odds_x"] / v["odds_avg"] * (1.0 + v["pred"]) - 1.0
        v["clv_x"] = v["odds_x"] * v["q_close"] - 1.0      # NaN where the line moved
        rows.append({"venue": venue, **ah_summary(v, "odds_x", "every side, no model",
                                                  clv_col="clv_x")})
        for m in VENUE_MARGINS:
            b = v[v["pred_x"] > m]
            rows.append({"venue": venue,
                         **ah_summary(b, "odds_x", f"predicted value > {m:.0%}", clv_col="clv_x"),
                         **ah_random(v, b, "odds_x", clv_col="clv_x")})
        for sname, g in v[v["pred_x"] > 0.0].groupby("Season"):
            seasons_tab.append({"venue": venue,
                                **ah_summary(g, "odds_x", f"season {sname}", clv_col="clv_x")})
    show("B. OPENING PRICE -- by venue:", pd.DataFrame(rows),
         "roi% is settled on EVERY bet at the opening line, pushes and half-wins included.",
         "clv% is only where the line did not move (clv_n of the bets).",
         "Pinnacle covers 2023/24-2024/25; the exchange 2024/25-2025/26.")
    show("B. Rule 'predicted value > 0%', by season and venue:", pd.DataFrame(seasons_tab))

    # Benchmark check: on same-line matches, does the closing price match reality?
    same = pool[pool["same_line"]].copy()
    same["band"] = pd.cut(same["odds_avg"], AH_BANDS).astype(str)
    g = same.groupby("band")
    bench = pd.DataFrame({"n": g.size(), "closing_prob": g["q_close"].mean(),
                          "realised_share": g["w_open"].sum() / (g["w_open"].sum()
                                                                 + g["l_open"].sum())})
    show("BENCHMARK CHECK (no model, line unchanged): closing price vs reality.",
         bench.reset_index(),
         "realised_share is the share of non-refunded stake that won. Where it sits",
         "  below closing_prob, value measured against the close is overstated.")

    print("\n" + "=" * 70)
    print("An edge here needs ALL of: positive profit whose interval clears zero")
    print("at a venue you can use, the same sign in every season, and beating")
    print("odds-matched random picks. Nothing here was tuned on the test seasons.")
    print("=" * 70)
    return pool


# =========================================================================
# PART 22 -- VALUE AGAINST A SHARP REFERENCE  (no model)
#
# Every test before this one asked whether OUR MODEL knows something the
# market does not. This one asks whether the market disagrees WITH ITSELF.
#
# Pinnacle takes the most money from professionals, keeps a low margin and
# does not limit winners, so its prices with the margin removed are the best
# public estimate of the true probability. Other books set their prices
# partly for their own customers and update them more slowly. When a book
# offers MORE than Pinnacle's fair price at the same moment, the bet is worth
# more than it costs -- by the sharpest available opinion.
#
# THE REFERENCE AND THE BET ARE ALWAYS FROM THE SAME SNAPSHOT: Pinnacle's
# opening price against each book's opening price, and closing against
# closing. Mixing snapshots would compare a price with information that did
# not exist when the other was quoted.
#
# TWO REFERENCES, TWO PERIODS. Pinnacle is in the file for 2022/23-2024/25.
# The Betfair Exchange -- an even sharper market, and one that is in the file
# for 2024/25 onward -- is used as a second reference. The two overlap in
# 2024/25, and from 2025/26 the exchange is an independent confirmation
# period for a rule fixed on Pinnacle.
#
# THE RULE IS FIXED HERE, BEFORE ANY RESULT WAS SEEN, and never changed:
VALUE_PRIMARY = {"min_value": 0.02,   # at least 2% above the fair price
                 "max_odds": 5.0,     # longshots: fair prices overrate them
                 "max_value": 0.15}   # above this it is almost always an error
#                                       that the book voids ("palpable error")
# Other margins are reported alongside for context, not for choosing.
VALUE_MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05)
#
# JUDGED BY: profit, settled by the real rules of each market; value against
# the REFERENCE's CLOSING fair price (the closing-line test, with the sharpest
# closing price available); every season separately; and odds-matched random
# bets from the same books.
#
# WHAT NO BACKTEST CAN TELL YOU: whether the book would have taken the bet.
# Books that are beaten this way cut stakes or close accounts, often within
# months. The exchange rows are the exception -- an exchange does not limit
# winners -- which is why they are reported as a book in their own right.
# =========================================================================

VALUE_BOOKS_1X2 = ["B365", "BW", "IW", "WH", "VC", "1XB", "BF", "BFD", "BMGM",
                   "BV", "CL", "LB", "PP", "SKB",
                   # books that appear only in older seasons
                   "BS", "GB", "SB", "SJ", "SO", "SY"]
VALUE_MARKETS = {
    # market: (sides, books, reference prefixes {ref: (open, close)},
    #          execution prefix pattern (open, close))
    "1X2": (("H", "D", "A"), VALUE_BOOKS_1X2,
            {"pinnacle": ("PS", "PSC"), "exchange": ("BFE", "BFEC")}),
    "O/U 2.5": ((">2.5", "<2.5"), ["B365"],
                {"pinnacle": ("P", "PC"), "exchange": ("BFE", "BFEC")}),
    "Asian handicap": (("H", "A"), ["B365"],
                       {"pinnacle": ("PAH", "PCAH"), "exchange": ("BFEAH", "BFECAH")}),
}
EXCHANGE_COMMISSION = 0.02
VALUE_BANDS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 1000.0)


def shin_vec(odds: np.ndarray) -> np.ndarray:
    """
    Shin de-vigging for a whole array of markets at once (rows = markets).

    The same model as shin_de_vig(), solved by bisection on every row together.
    Rows with a missing or impossible price come back as NaN.
    """
    odds = np.asarray(odds, float)
    ok = np.isfinite(odds).all(1) & (odds > 1.0).all(1)
    q = np.where(ok[:, None], 1.0 / np.where(ok[:, None], odds, 2.0), np.nan)
    booksum = q.sum(1, keepdims=True)
    lo = np.zeros((len(q), 1))
    hi = np.full((len(q), 1), 0.99)

    def implied(z):
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / booksum) - z) / (2.0 * (1.0 - z))

    for _ in range(40):
        mid = (lo + hi) / 2.0
        over = implied(mid).sum(1, keepdims=True) > 1.0
        lo, hi = np.where(over, mid, lo), np.where(over, hi, mid)
    p = implied((lo + hi) / 2.0)
    p = p / p.sum(1, keepdims=True)
    under = (booksum <= 1.0).ravel()                 # no margin: just normalise
    p[under] = (q / booksum)[under]
    p[~ok] = np.nan
    return p


def _value_frame(df: pd.DataFrame, snapshots=("open", "close")) -> pd.DataFrame:
    """
    One row per match x market x side x book x snapshot x reference.

    Unplayed fixtures are allowed: their settlement columns come back missing.
    """
    played = df["FTR"].isin(["H", "D", "A"]).to_numpy() if "FTR" in df.columns \
        else np.zeros(len(df), bool)
    hg = pd.to_numeric(df.get("FTHG"), errors="coerce") if "FTHG" in df.columns \
        else pd.Series(np.nan, index=df.index)
    ag = pd.to_numeric(df.get("FTAG"), errors="coerce") if "FTAG" in df.columns \
        else pd.Series(np.nan, index=df.index)
    gd = np.nan_to_num((hg - ag).to_numpy(), nan=999).astype(int)
    tot = (hg + ag).to_numpy()
    ftr = df["FTR"].to_numpy() if "FTR" in df.columns else np.full(len(df), "")
    base = {"match_key": df["match_key"].to_numpy(), "Date": df["Date"].to_numpy(),
            "Season": df["Season"].astype(str).to_numpy(), "Div": df["Div"].to_numpy()}
    frames = []
    for market, (sides, books, refs) in VALUE_MARKETS.items():
        for snap in snapshots:
            k = 0 if snap == "open" else 1
            closing_fair = {}
            for ref, pre in refs.items():
                cols = [f"{pre[1]}{sd}" for sd in sides]
                if all(c in df.columns for c in cols):
                    closing_fair[ref] = shin_vec(df[cols].to_numpy(float))
            for ref, pre in refs.items():
                cols = [f"{pre[k]}{sd}" for sd in sides]
                if not all(c in df.columns for c in cols):
                    continue
                fair = shin_vec(df[cols].to_numpy(float))
                exec_books = list(books) + ["Max"]
                if ref == "pinnacle":
                    exec_books.append("EXCH")         # the exchange as a venue
                for book in exec_books:
                    if market == "1X2":
                        bpre = {"EXCH": ("BFE", "BFEC"), "Max": ("Max", "MaxC")}.get(
                            book, (book, f"{book}C"))[k]
                    elif market == "O/U 2.5":
                        bpre = {"EXCH": ("BFE", "BFEC"), "Max": ("Max", "MaxC")}.get(
                            book, (book, f"{book}C"))[k]
                    else:
                        bpre = {"EXCH": ("BFEAH", "BFECAH"), "Max": ("MaxAH", "MaxCAH")}.get(
                            book, (f"{book}AH", f"{book}CAH"))[k]
                    bcols = [f"{bpre}{sd}" for sd in sides]
                    if not all(c in df.columns for c in bcols):
                        continue
                    line = None
                    if market == "Asian handicap":
                        line = df["AHh" if snap == "open" else "AHCh"].to_numpy(float)
                        same_line = np.isclose(df["AHh"].to_numpy(float), df["AHCh"].to_numpy(float))
                    for j, sd in enumerate(sides):
                        odds = df[bcols[j]].to_numpy(float)
                        if book == "EXCH":
                            odds = 1.0 + (odds - 1.0) * (1.0 - EXCHANGE_COMMISSION)
                        keep = np.isfinite(odds) & (odds > 1.0) & np.isfinite(fair[:, j])
                        if not keep.any():
                            continue
                        if market == "1X2":
                            w = (ftr == sd).astype(float)
                            l = 1.0 - w
                        elif market == "O/U 2.5":
                            w = ((tot > 2.5) == (sd == ">2.5")).astype(float)
                            l = 1.0 - w
                        else:
                            w, l = ah_settle(gd, line, sd)
                        w = np.where(played, w, np.nan)
                        l = np.where(played, l, np.nan)
                        clv = {}
                        for cref, cf in closing_fair.items():
                            c = odds * cf[:, j] - 1.0
                            if market == "Asian handicap" and snap == "open":
                                c = np.where(same_line, c, np.nan)
                            clv[f"clv_{cref}"] = c
                        frames.append(pd.DataFrame({
                            **{kk: v[keep] for kk, v in base.items()},
                            "market": market, "snapshot": snap, "reference": ref,
                            "book": book, "side": sd,
                            "odds": odds[keep], "p_fair": fair[keep, j],
                            "value": (odds * fair[:, j] - 1.0)[keep],
                            "w": w[keep], "l": l[keep],
                            **{kk: v[keep] for kk, v in clv.items()}}))
    out = pd.concat(frames, ignore_index=True)
    out["pnl"] = out["w"] * (out["odds"] - 1.0) - out["l"]
    return out


def _week_boot(b: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 7) -> tuple:
    """
    95% interval of the mean of `col` per bet, resampling whole WEEKS.

    Matches on one weekend share news, weather and market mood, so their
    outcomes are not independent. Resampling single matches treats them as if
    they were and gives intervals that are too narrow; resampling weeks keeps
    each weekend's bets together. Returns (low, high, P(mean <= 0)).
    """
    c = b.dropna(subset=[col])
    if len(c) < 20 or "Date" not in c.columns:
        return (np.nan, np.nan, np.nan)
    week = pd.to_datetime(c["Date"]).dt.to_period("W").astype(str).to_numpy()
    g = pd.DataFrame({"w": week, "v": c[col].to_numpy(float)}).groupby("w")["v"]
    sums, counts = g.sum().to_numpy(), g.count().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(sums), (n_boot, len(sums)))
    m = sums[idx].sum(1) / counts[idx].sum(1) * 100
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m <= 0).mean())


def _vsum(b: pd.DataFrame, label: str, clv_col: str = None, n_boot: int = 3000) -> dict:
    if b.empty:
        return {"set": label, "n_bets": 0}
    r = bootstrap_roi(b.assign(Stake=1.0, PnL=b["pnl"]), n_boot=n_boot)
    wlo, whi, wp = _week_boot(b, "pnl", n_boot=min(n_boot, 2000))
    out = {"set": label, "n_bets": len(b), "avg_odds": b["odds"].mean(),
           "avg_value%": b["value"].mean() * 100,
           "units_staked": float(len(b)), "units_won": float(b["pnl"].sum()),
           "roi%": r["roi"], "roi_lo": r["ci_low"], "roi_hi": r["ci_high"],
           "P(roi<=0)": r["p_le_zero"],
           "roi_wk_lo": wlo, "roi_wk_hi": whi, "P(roi<=0)_wk": wp}
    if clv_col and clv_col in b.columns:
        c = b.dropna(subset=[clv_col])
        if len(c) >= 20:
            lo, hi, p0 = _match_boot(c.assign(_c=c[clv_col]), "_c", n_boot=n_boot)
            clo, chi, _ = _week_boot(c, clv_col, n_boot=min(n_boot, 2000))
            out.update({"clv%": c[clv_col].mean() * 100, "clv_lo": lo, "clv_hi": hi,
                        "clv_wk_lo": clo, "clv_wk_hi": chi})
    return out


def _vrandom(pool: pd.DataFrame, bets: pd.DataFrame, n_draws: int = 300,
             seed: int = 42) -> dict:
    """Random bets from the same books and markets, matched on odds band."""
    if bets.empty:
        return {}
    pool = pool.reset_index(drop=True)
    pb = pd.cut(pool["odds"], VALUE_BANDS, labels=False).fillna(-1).astype(int).to_numpy()
    bb = pd.cut(bets["odds"], VALUE_BANDS, labels=False).fillna(-1).astype(int)
    by = {k: np.flatnonzero(pb == k) for k in np.unique(pb)}
    want = bb.value_counts()
    pnl = pool["pnl"].to_numpy()
    rng = np.random.default_rng(seed)
    rr = np.array([pnl[np.concatenate([rng.choice(by[k], n) for k, n in want.items()
                                       if k in by])].mean() * 100
                   for _ in range(n_draws)])
    m = bets["pnl"].mean() * 100
    return {"random_roi%": rr.mean(), "random_p95": np.percentile(rr, 95),
            "beats_random": (rr < m).mean()}


def primary_bets(v: pd.DataFrame, rule: dict = VALUE_PRIMARY,
                 one_per_selection: bool = True) -> pd.DataFrame:
    """The fixed rule; at most one bet per selection, at its best-value book."""
    b = v[(v["value"] >= rule["min_value"]) & (v["value"] <= rule["max_value"])
          & (v["odds"] <= rule["max_odds"])]
    if one_per_selection and len(b):
        b = (b.sort_values("value", ascending=False)
              .drop_duplicates(["match_key", "market", "snapshot", "reference", "side"]))
    return b


def tier_of(div) -> int:
    """League tier: 1 = top flight, 2 = second division, ... (see TIERS)."""
    return TIERS.get(str(div), ("?", 0))[1]


def tier_label(min_tier: int) -> str:
    return "all divisions" if min_tier <= 1 else f"tier {min_tier} and below only"


def value_main(source: str, min_tier: int = 1, exclude_books=()) -> pd.DataFrame:
    print("=" * 70)
    print("VALUE AGAINST A SHARP REFERENCE  (no model -- the market against itself)")
    print(f"  leagues: {tier_label(min_tier)}")
    print(f"  fixed rule: value >= {VALUE_PRIMARY['min_value']:.0%}, odds <= "
          f"{VALUE_PRIMARY['max_odds']}, value <= {VALUE_PRIMARY['max_value']:.0%} "
          f"(palpable errors), one bet per selection")
    print("=" * 70)
    df = clean_prices(apply_column_renames(read_match_table(source)))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Season"].astype(str)
    df = df.dropna(subset=["Date", "FTHG", "FTAG", "FTR"])
    df = df[df["FTR"].isin(["H", "D", "A"])].reset_index(drop=True)
    df["match_key"] = (df["Div"].astype(str) + "|" + df["Date"].dt.strftime("%Y-%m-%d")
                       + "|" + df["HomeTeam"].astype(str) + "|" + df["AwayTeam"].astype(str))
    if min_tier > 1:
        df = df[df["Div"].map(tier_of) >= min_tier].reset_index(drop=True)
        print(f"  kept divisions: {', '.join(sorted(df['Div'].unique()))}")
    print(f"{len(df):,} played matches, seasons {', '.join(sorted(df['Season'].unique()))}")
    t0 = time.time()
    v = _value_frame(df)
    if exclude_books:
        v = v[~v["book"].isin(exclude_books)].reset_index(drop=True)
        print(f"  bookmakers excluded: {', '.join(exclude_books)}")
    print(f"{len(v):,} priced candidate bets built [{time.time() - t0:.0f}s]")

    # 0. Is the reference itself honest? ------------------------------------
    rows = []
    for ref in ("pinnacle", "exchange"):
        r = v[(v["reference"] == ref) & (v["snapshot"] == "open") & (v["book"] == "Max")]
        r = r.drop_duplicates(["match_key", "market", "side"])
        r = r.assign(band=pd.cut(1.0 / r["p_fair"], VALUE_BANDS).astype(str),
                     won=r["w"] / (r["w"] + r["l"]).replace(0, np.nan))
        g = r.groupby(["market", "band"], sort=False)
        t = pd.DataFrame({"n": g.size(), "fair_prob": g["p_fair"].mean(),
                          "won": g["won"].mean()}).reset_index()
        t["reference"] = ref
        rows.append(t[t["n"] >= 200])
    show("0. IS THE REFERENCE HONEST? Opening fair probability vs how often it happened.",
         pd.concat(rows, ignore_index=True),
         "If 'won' sits clearly below 'fair_prob' in a band, value measured there is fake.")

    # 1. The fixed rule ------------------------------------------------------
    rows = []
    for ref in ("pinnacle", "exchange"):
        for snap in ("open", "close"):
            for market in VALUE_MARKETS:
                sub = v[(v["reference"] == ref) & (v["snapshot"] == snap)
                        & (v["market"] == market)]
                soft = sub[~sub["book"].isin(["Max", "EXCH"])]
                for label, pool in (("single books", soft),
                                    ("exchange as venue", sub[sub["book"] == "EXCH"]),
                                    ("best price (Max)", sub[sub["book"] == "Max"])):
                    if pool.empty:
                        continue
                    b = primary_bets(pool)
                    rows.append({"reference": ref, "snapshot": snap, "market": market,
                                 "venue": label,
                                 **_vsum(b, "", f"clv_{ref}" if snap == "open" else None),
                                 **_vrandom(pool, b)})
    t = pd.DataFrame(rows).drop(columns="set")
    show("1. THE FIXED RULE, everywhere it can be applied.", t,
         "clv% (opening rows): value against the SAME reference's closing fair price.",
         "beats_random: share of 300 odds-matched random draws from the same pool beaten.")

    # 2. By season, for the rows that matter --------------------------------
    rows = []
    for (ref, snap, market), sub in v.groupby(["reference", "snapshot", "market"]):
        soft = sub[~sub["book"].isin(["Max", "EXCH"])]
        b = primary_bets(soft)
        for season, g in b.groupby("Season"):
            rows.append({"reference": ref, "snapshot": snap, "market": market,
                         **_vsum(g, season, f"clv_{ref}" if snap == "open" else None)})
    show("2. THE FIXED RULE AT SINGLE BOOKS, by season:", pd.DataFrame(rows),
         "An edge has the same sign in every season. 2025/26 onward is the exchange-",
         "  reference confirmation period; Pinnacle is not in the file for it.")

    # 3. By book -------------------------------------------------------------
    rows = []
    for (ref, snap, market, book), sub in v.groupby(["reference", "snapshot", "market", "book"]):
        b = primary_bets(sub)
        if len(b) >= 30:
            rows.append({"reference": ref, "snapshot": snap, "market": market, "book": book,
                         **_vsum(b, ",".join(sorted(b["Season"].unique())),
                                 f"clv_{ref}" if snap == "open" else None, n_boot=1500)})
    show("3. THE FIXED RULE, book by book (>= 30 bets):", pd.DataFrame(rows),
         "Small books, single seasons: noise. A soft book shows up in every season.")

    # 4. Margins, for context -----------------------------------------------
    rows = []
    for ref in ("pinnacle", "exchange"):
        for snap in ("open", "close"):
            soft = v[(v["reference"] == ref) & (v["snapshot"] == snap)
                     & ~v["book"].isin(["Max", "EXCH"])]
            for m in VALUE_MARGINS:
                b = primary_bets(soft, {**VALUE_PRIMARY, "min_value": m})
                rows.append({"reference": ref, "snapshot": snap,
                             **_vsum(b, f"value >= {m:.0%}",
                                     f"clv_{ref}" if snap == "open" else None)})
    show("4. OTHER MARGINS (context, NOT for choosing), all markets, single books:",
         pd.DataFrame(rows),
         "If profit rises with the margin, the value is real. If it does not, it is noise.",
         "units_staked / units_won: flat 1 unit per bet; units_won is the net profit.")

    # 5. by league tier -- the lower-league hypothesis, with intervals ------
    soft = v[(v["reference"] == "pinnacle") & (v["snapshot"] == "open")
             & ~v["book"].isin(["Max", "EXCH"])]
    b = primary_bets(soft).assign(tier=lambda x: x["Div"].map(tier_of))
    rows = []
    groups = [(f"tier {t}", b[b["tier"] == t]) for t in sorted(b["tier"].unique())]
    groups += [("tier 2 and below", b[b["tier"] >= 2]), ("tier 3 and below", b[b["tier"] >= 3])]
    for label, g in groups:
        if len(g) >= 30:
            rows.append({**_vsum(g, "", "clv_pinnacle"), "set": label,
                         "divisions": ",".join(sorted(g["Div"].unique()))})
    if rows:
        t = pd.DataFrame(rows)
        show("5. THE FIXED RULE BY LEAGUE TIER (Pinnacle reference, early prices, single books):",
             t[["set"] + [c for c in t.columns if c != "set"]],
             "Tier 1 = top flights. Tiers 3-5 exist only for England and Scotland here.",
             "clv% (value against Pinnacle's close) is the steadier measure; roi% needs many bets.")
        rows = []
        for (season, lower), g in b.assign(lower=b["tier"] >= 2).groupby(["Season", "lower"]):
            r = _vsum(g, "", "clv_pinnacle")
            rows.append({"season": season, "leagues": "tier 2 and below" if lower else "tier 1",
                         "n_bets": r.get("n_bets"), "units_won": r.get("units_won"),
                         "roi%": r.get("roi%"), "clv%": r.get("clv%")})
        ts = pd.DataFrame(rows).pivot(index="season", columns="leagues",
                                      values=["n_bets", "units_won", "roi%", "clv%"])
        ts.columns = [f"{a} ({b2})" for a, b2 in ts.columns]
        show("   ...top flight against the lower leagues, season by season:",
             ts.reset_index().round(3),
             "A real difference between them shows up with the same sign in most seasons.")

    # 6. which bookmaker the profit came from --------------------------------
    soft = v[(v["reference"] == "pinnacle") & (v["snapshot"] == "open")
             & ~v["book"].isin(["Max", "EXCH"])]
    b = primary_bets(soft)
    if len(b):
        total = b["pnl"].sum()
        rows = []
        for book, g in b.groupby("book"):
            r = _vsum(g, "", "clv_pinnacle", n_boot=1500)
            rows.append({"book": book, "seasons": f"{g['Season'].min()}-{g['Season'].max()}",
                         "n_bets": len(g), "units_won": r["units_won"],
                         "share_of_profit%": r["units_won"] / total * 100 if total else np.nan,
                         "roi%": r["roi%"], "roi_wk_lo": r["roi_wk_lo"],
                         "roi_wk_hi": r["roi_wk_hi"], "clv%": r.get("clv%")})
        show("6. WHERE THE PROFIT CAME FROM (the fixed rule; each bet at its best-value bookmaker):",
             pd.DataFrame(rows).sort_values("units_won", ascending=False).round(3),
             "share_of_profit% can exceed 100 when other bookmakers lost money.")
        # 7. leave one bookmaker out: the bet goes to the next-best book instead
        rows = [{"without": "(none: every bookmaker available)",
                 **_vsum(b, "", "clv_pinnacle", n_boot=1500)}]
        for book in sorted(b["book"].unique()):
            bb = primary_bets(soft[soft["book"] != book])
            rows.append({"without": book, **_vsum(bb, "", "clv_pinnacle", n_boot=1500)})
        t7 = pd.DataFrame(rows).drop(columns="set")
        show("7. IF ONE BOOKMAKER WERE UNAVAILABLE (its bets go to the next-best price, if any):",
             t7[["without", "n_bets", "units_won", "roi%", "roi_lo", "roi_hi",
                 "roi_wk_lo", "roi_wk_hi", "clv%", "clv_wk_lo", "clv_wk_hi"]].round(3),
             "A robust edge stays positive whichever single bookmaker is removed.")

    print("\n" + "=" * 70)
    print("An edge needs: profit > 0 with an interval clearing zero, the same sign")
    print("in every season, positive closing-line value, beating random bets, and")
    print("profit that rises with the margin. Then it needs a book that lets you bet.")
    print("=" * 70)
    return v


# =========================================================================
# PART 23 -- THE SCANNER AND THE BET LOG  (for live use and paper trading)
#
# `scan` reads a file of UPCOMING matches in the football-data layout -- the
# fixtures.csv that football-data.co.uk publishes, or any file with the same
# column names -- applies VALUE_PRIMARY unchanged, and lists the bets, with a
# stake. With --log it also appends them to a bet log.
#
# `review` reads that log back against a file that now has the results and the
# closing prices, and reports the two numbers that matter: closing-line value
# (the proof, available after a few hundred bets) and profit (the outcome,
# which needs thousands to settle).
#
# WHICH REFERENCE, WHEN. The backtest found:
#   * Pinnacle's price is a good reference at ANY time -- early prices
#     included, which is when soft books are most often wrong.
#   * The exchange is a good reference only CLOSE TO KICKOFF, when it is
#     liquid. Early, its prices are thin, and "value" against them was mostly
#     the exchange being wrong (negative closing-line value in the backtest).
# So the scanner refuses the exchange as a reference for early prices unless
# told --closing, which means "these prices were taken shortly before kickoff".
#
# STAKES are a quarter of the Kelly fraction, capped at 2% of the bank:
# full Kelly assumes the edge is known exactly, and it is not. At the edge the
# backtest found, a quarter-Kelly bank grows slowly and survives bad runs.
# =========================================================================

SCAN_KELLY_FRACTION = 0.25
SCAN_MAX_STAKE = 0.02
LOG_COLUMNS = ["logged_at", "match_key", "Div", "Date", "HomeTeam", "AwayTeam",
               "market", "side", "line", "book", "odds", "p_fair", "value",
               "reference", "snapshot", "stake", "model_agrees"]


def _prepare_fixtures(path: str) -> pd.DataFrame:
    df = clean_prices(apply_column_renames(read_match_table(path)),
                      verbose=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if "Season" not in df.columns:
        df["Season"] = "live"
    df["Season"] = df["Season"].astype(str)
    df["match_key"] = (df["Div"].astype(str) + "|" + df["Date"].dt.strftime("%Y-%m-%d")
                       + "|" + df["HomeTeam"].astype(str) + "|" + df["AwayTeam"].astype(str))
    return df


def scan(path: str, reference: str = "pinnacle", closing: bool = False,
         bank: float = 1000.0, log: str = None, history: str = None) -> pd.DataFrame:
    if reference == "exchange" and not closing:
        raise SystemExit(
            "The exchange is only a reliable reference close to kickoff. If these "
            "prices were taken shortly before kickoff, rerun with --closing; "
            "otherwise use --reference pinnacle.")
    df = _prepare_fixtures(path)
    snap = "close" if closing else "open"
    v = _value_frame(df, snapshots=(snap,))
    v = v[(v["reference"] == reference) & (v["book"] != "Max")]
    if v.empty:
        need = "PSH/PSD/PSA (Pinnacle)" if reference == "pinnacle" else "BFEH/BFED/BFEA"
        print(f"No {reference} prices in {path} -- the scan needs {need} columns"
              f"{' (closing: PSCH ...)' if closing else ''}.")
        return v
    bets = primary_bets(v)
    if bets.empty:
        print(f"{len(df)} matches scanned: no bet meets the rule.")
        return bets
    kelly = bets["value"] / (bets["odds"] - 1.0)
    bets = bets.assign(
        stake=np.round(bank * np.minimum(SCAN_KELLY_FRACTION * kelly, SCAN_MAX_STAKE), 2))
    info = df.set_index("match_key")[["HomeTeam", "AwayTeam"]]
    bets = bets.join(info, on="match_key")
    bets["line"] = np.nan
    if "AHh" in df.columns:
        lines = df.set_index("match_key")["AHCh" if closing and "AHCh" in df.columns else "AHh"]
        ah = bets["market"] == "Asian handicap"
        bets.loc[ah, "line"] = bets.loc[ah, "match_key"].map(lines)
    bets["model_agrees"] = ""
    if history:
        view = model_view(history, df)
        if len(view):
            bets = bets.merge(view, on=["Div", "Date", "HomeTeam", "AwayTeam"], how="left")
            mp = np.select([bets["market"].eq("1X2") & bets["side"].eq(o) for o in OUTCOMES],
                           [bets[f"model_p_{o}"] for o in OUTCOMES], np.nan)
            bets["model_p"] = mp
            bets["model_agrees"] = np.where(np.isnan(mp), "",
                                            np.where(mp > bets["p_fair"], "AGREE", "disagree"))
    bets = bets.sort_values(["Date", "match_key", "value"], ascending=[True, True, False])
    show(f"{len(bets)} bets from {len(df)} matches  (reference {reference}, "
         f"{'near-kickoff' if closing else 'early'} prices, bank {bank:,.0f})",
         bets[["Date", "Div", "HomeTeam", "AwayTeam", "market", "side", "line",
               "book", "odds", "p_fair", "value", "stake", "model_agrees"]].assign(
             Date=lambda x: x["Date"].dt.strftime("%Y-%m-%d"),
             p_fair=lambda x: x["p_fair"].round(3),
             value=lambda x: (x["value"] * 100).round(1).astype(str) + "%"),
         "book: where the price is. EXCH = Betfair Exchange, after 2% commission.",
         "model_agrees: AGREE = stage 1 also rates this side above the fair price"
         + (" (match result only)." if history else " -- add --history FILE to see it."),
         "Check the price is still there before betting; skip it if it has moved.")
    if log:
        out = bets.assign(logged_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                          Date=bets["Date"].dt.strftime("%Y-%m-%d"),
                          snapshot=snap)[LOG_COLUMNS]
        exists = Path(log).exists()
        if exists:
            old = pd.read_csv(log)
            key = ["match_key", "market", "side", "book"]
            out = out[~out.set_index(key).index.isin(old.set_index(key).index)]
        out.to_csv(log, mode="a", header=not exists, index=False)
        print(f"\nlogged {len(out)} new bets to {log}")
    return bets


def review(log: str, results: str, reference: str = "pinnacle") -> None:
    bets = pd.read_csv(log)
    df = _prepare_fixtures(results)
    df = df[df["FTR"].isin(["H", "D", "A"])]
    v = _value_frame(df, snapshots=("open",))
    v = v[v["reference"] == v["reference"].iloc[0]] if reference == "any" else \
        v[v["reference"] == reference]
    key = ["match_key", "market", "side"]
    # settlement and closing fair price per selection (independent of the book)
    sel = (v.drop_duplicates(key)
             .set_index(key)[["w", "l", f"clv_{reference}", "odds"]]
             .rename(columns={"odds": "_ref_odds"}))
    close_fair = {}
    for market, (sides, _, refs) in VALUE_MARKETS.items():
        cols = [f"{refs[reference][1]}{sd}" for sd in sides]
        if all(c in df.columns for c in cols):
            f = shin_vec(df[cols].to_numpy(float))
            for j, sd in enumerate(sides):
                close_fair.update({(mk, market, sd): f[i, j]
                                   for i, mk in enumerate(df["match_key"])})
    b = bets.join(sel[["w", "l"]], on=key)
    b["p_close"] = [close_fair.get(k, np.nan) for k in zip(b["match_key"], b["market"], b["side"])]
    if (b["market"] == "Asian handicap").any() and "AHCh" in df.columns:
        moved = b["market"].eq("Asian handicap") & ~np.isclose(
            b["line"], b["match_key"].map(df.set_index("match_key")["AHCh"]))
        b.loc[moved, "p_close"] = np.nan          # different bet at the close
    b["clv"] = b["odds"] * b["p_close"] - 1.0
    b["pnl_units"] = b["w"] * (b["odds"] - 1.0) - b["l"]
    b["pnl"] = b["pnl_units"] * b["stake"]
    done = b.dropna(subset=["w"])
    print(f"{len(bets)} bets logged, {len(done)} settled, "
          f"{done['clv'].notna().sum()} with a closing price to judge.")
    if done.empty:
        return
    c = done.dropna(subset=["clv"])
    lo, hi, _ = _match_boot(c, "clv", n_boot=3000) if len(c) >= 20 else (np.nan, np.nan, 0)
    r = bootstrap_roi(done.assign(Stake=done["stake"], PnL=done["pnl"]), n_boot=3000)
    print(f"\n  closing-line value  {c['clv'].mean() * 100:+.2f}%   "
          f"95% interval [{lo:+.2f}, {hi:+.2f}]   (the backtest: about +2.7%)")
    print(f"  profit              {done['pnl'].sum():+,.2f} on {done['stake'].sum():,.2f} staked"
          f"   ROI {r['roi']:+.2f}%  [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]")
    if "model_agrees" in done.columns and done["model_agrees"].fillna("").ne("").any():
        show("  By stage 1's view (the backtest: AGREE ahead in 12 of 12 seasons):",
             done.assign(model_agrees=done["model_agrees"].fillna("").replace("", "no view"))
                 .groupby("model_agrees").agg(
                     bets=("odds", "size"), clv_pct=("clv", lambda x: x.mean() * 100),
                     roi_pct=("pnl_units", lambda x: x.mean() * 100)).round(2).reset_index())
    show("  By book:", done.groupby("book").agg(
        bets=("odds", "size"), clv_pct=("clv", lambda x: x.mean() * 100),
        roi_pct=("pnl_units", lambda x: x.mean() * 100)).round(2).reset_index())
    print("\n  Judge the strategy on closing-line value. Profit needs thousands of")
    print("  bets to mean anything; CLV that stays clearly positive over a few")
    print("  hundred is the evidence. If it drifts to zero, stop.")


# =========================================================================
# PART 24 -- THE TWO-STAGE MODEL, ANCHORED ON PINNACLE
#
# The main pipeline shrinks stage 1 toward the AVERAGE closing price and bets
# at the BEST closing price. Both are the same moment and the same books, and
# with w = 0 -- what it found -- there is nothing left to bet. This keeps the
# method and changes the two ends:
#
#   STAGE 1  unchanged: the market-blind match engine, Elo fitted per fold.
#   STAGE 2  two versions, both pulled toward PINNACLE'S FAIR OPENING PRICE:
#              (a) the correction -- stage 1 shrunk to Pinnacle with a weight
#                  w fitted by log loss on the training window, out of fold;
#              (b) a market-aware learner on Pinnacle's price and stage 1
#                  together, itself shrunk to Pinnacle by a fitted weight w2.
#   BETS     at the SOFT books' opening prices (and the exchange), by the same
#            fixed rule as PART 22 -- VALUE_PRIMARY, untouched.
#
# THE DESIGN NESTS THE NO-MODEL STRATEGY. At w = 0 arm (a) is exactly PART 22.
# So the comparison is clean: if w comes out above zero and arm (a) beats the
# Pinnacle-only arm on the same books, stage 1 knows something Pinnacle's
# opening price does not. If w is zero, it does not, and the edge in PART 22
# belongs to the soft books' lag alone.
#
# One more arm, declared here before running: the Pinnacle value bets split by
# whether stage 1 AGREES (puts the side above Pinnacle's fair price) or not.
# If the model carries information, the agreeing half should do better.
#
# Pinnacle's prices end in 2024/25 in this file, so the walk-forward has two
# test seasons: 2023/24 (trained on 2022/23) and 2024/25 (on both before it).
# =========================================================================

PIN_OPEN = ["PSH", "PSD", "PSA"]
PIN_CLOSE = ["PSCH", "PSCD", "PSCA"]


def _logit3(p: np.ndarray) -> np.ndarray:
    lp = np.log(np.clip(p, 1e-6, 1.0))
    return lp - lp[:, [1]]                       # relative to the draw


def pinnacle_fold(window: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> tuple:
    window = window.reset_index(drop=True)
    test = test.reset_index(drop=True)
    yw = window["target"].astype(int).to_numpy()

    # ---- stage 1, market-blind ------------------------------------------
    engine = MatchEngine(seed=seed)
    oof = engine.fit_predict_oof(window)
    engine.fit(window)
    engine.fit_rho(oof, window)
    p1_w = engine.probs_from_physical(oof, window)          # out of fold
    p1_t = engine.predict_proba(test)

    pin_w = shin_vec(window[PIN_OPEN].to_numpy(float))
    pin_t = shin_vec(test[PIN_OPEN].to_numpy(float))

    # ---- stage 2 (a): the correction, fitted on out-of-fold stage 1 -----
    w = fit_static_shrinkage(pin_w, p1_w, yw)
    corr_t = geometric_blend(pin_t, p1_t, w)

    # ---- stage 2 (b): market-aware learner, then shrunk to Pinnacle ------
    ctx = [c for c in CLV_CONTEXT_FEATURES if c in window.columns]

    def s2x(pin, p1, d):
        return np.column_stack([_logit3(pin), _logit3(p1), p1 - pin,
                                d[ctx].to_numpy(float)])

    Xw, Xt = s2x(pin_w, p1_w, window), s2x(pin_t, p1_t, test)
    p2_w = cross_val_predict(make_classifier(seed), Xw, yw, cv=3, method="predict_proba")
    w2 = fit_static_shrinkage(pin_w, p2_w, yw)
    p2_t = make_classifier(seed).fit(Xw, yw).predict_proba(Xt)
    s2_t = geometric_blend(pin_t, p2_t, w2)

    arms = {"Pinnacle only (w = 0)": pin_t,
            "stage 1 shrunk to Pinnacle (fitted w)": corr_t,
            "stage 2 learner shrunk to Pinnacle (fitted w2)": s2_t,
            "stage 1 raw (w = 1, for contrast)": p1_t}
    yt = test["target"].astype(int).to_numpy()
    diag = {"w": w, "w2": w2,
            **{f"ll {k}": log_loss(yt, v, labels=[0, 1, 2]) for k, v in arms.items()},
            "ll Pinnacle closing": log_loss(
                yt, shin_vec(test[PIN_CLOSE].to_numpy(float)), labels=[0, 1, 2])}
    return arms, p1_t, diag


def _soft_pool(test: pd.DataFrame, arms: dict, p1: np.ndarray) -> pd.DataFrame:
    """Every soft-book opening price in the test season, valued by every arm."""
    y = test["target"].astype(int).to_numpy()
    pin = arms["Pinnacle only (w = 0)"]
    pin_close = shin_vec(test[PIN_CLOSE].to_numpy(float))
    # Context for the learned stage 2 (PART 27). Everything here exists when the
    # opening prices are quoted; nothing is a closing price.
    ctx = {c: test[c].to_numpy(float) for c in S2_MATCH_CONTEXT if c in test.columns}
    ctx["pin_overround"] = (1.0 / test[PIN_OPEN]).sum(axis=1).to_numpy(float) - 1.0
    for k, o in enumerate(OUTCOMES):
        ctx[f"p1_{o}"] = p1[:, k]
        ctx[f"pin_{o}"] = pin[:, k]
    side_ctx = {}
    for o in OUTCOMES:
        mx, av = test.get(f"Max{o}"), test.get(f"Avg{o}")
        side_ctx[o] = ((mx / av - 1.0).to_numpy(float) if mx is not None and av is not None
                       else np.full(len(test), np.nan))
    frames = []
    for book in VALUE_BOOKS_1X2 + ["EXCH"]:
        pre = "BFE" if book == "EXCH" else book
        cols = [f"{pre}{o}" for o in OUTCOMES]
        if not all(c in test.columns for c in cols):
            continue
        for k, o in enumerate(OUTCOMES):
            odds = test[cols[k]].to_numpy(float)
            if book == "EXCH":
                odds = 1.0 + (odds - 1.0) * (1.0 - EXCHANGE_COMMISSION)
            keep = np.isfinite(odds) & (odds > 1.0) & np.isfinite(pin[:, k])
            if not keep.any():
                continue
            win = (y == k).astype(float)
            frames.append(pd.DataFrame({
                "match_key": test["match_key"].to_numpy()[keep],
                "Date": test["Date"].to_numpy()[keep],
                "Season": test["Season"].astype(str).to_numpy()[keep],
                "Div": test["Div"].to_numpy()[keep],
                "market": "1X2", "snapshot": "open", "reference": "pinnacle",
                "book": book, "side": o, "odds": odds[keep],
                "w": win[keep], "l": 1.0 - win[keep],
                "pnl": (win * (odds - 1.0) - (1.0 - win))[keep],
                "clv_pinnacle": (odds * pin_close[:, k] - 1.0)[keep],
                "p_stage1": p1[keep, k], "p_pin": pin[keep, k],
                "spread_side": side_ctx[o][keep],
                **{f"ctx_{c}": v[keep] for c, v in ctx.items()},
                **{f"p::{name}": a[keep, k] for name, a in arms.items()}}))
    return pd.concat(frames, ignore_index=True)


def pinnacle_main(source: str, mode: str = "multiseason", seed: int = 42):
    print("=" * 70)
    print("TWO-STAGE MODEL ANCHORED ON PINNACLE, BET AT THE SOFT BOOKS")
    print(f"  learner: {LEARNER}, ensemble of {ENSEMBLE_K};  fixed rule: value >= "
          f"{VALUE_PRIMARY['min_value']:.0%}, odds <= {VALUE_PRIMARY['max_odds']}, "
          f"value <= {VALUE_PRIMARY['max_value']:.0%}")
    print("=" * 70)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = load_and_prepare(source) if mode == "export" else load_multiseason(source)
    full = df
    missing = [c for c in PIN_OPEN + PIN_CLOSE if c not in df.columns]
    if missing:
        raise KeyError(f"Pinnacle columns missing from this file: {missing}")
    df = df[df[STAGE1_REQUIRED + ["target"] + PIN_OPEN + PIN_CLOSE].notna().all(axis=1)]
    df = df[(df[PIN_OPEN + PIN_CLOSE] > 1.0).all(axis=1)].sort_values("Date")
    df = df.reset_index(drop=True)
    df["Season"] = df["Season"].astype(str)
    seasons = sorted(df["Season"].unique())
    print(f"{len(df):,} matches with Pinnacle prices, seasons {', '.join(seasons)}")

    pools, diags = [], []
    for i in range(1, len(seasons)):
        te = df[df["Season"] == seasons[i]]
        win = df[df["Season"].isin(seasons[:i]) & (df["Date"] < te["Date"].min())]
        if len(win) < 2000 or len(te) < 300:
            continue
        win, te = with_window_elo(full, win, te)
        t0 = time.time()
        arms, p1, diag = pinnacle_fold(win, te, seed)
        pools.append(_soft_pool(te.reset_index(drop=True), arms, p1))
        diags.append({"test": seasons[i], "n_window": len(win), "n_test": len(te),
                      **diag, "seconds": round(time.time() - t0)})
        print(f"  fold {seasons[i]}: w = {diag['w']:.3f}, w2 = {diag['w2']:.3f}  "
              f"[{diags[-1]['seconds']}s]", flush=True)
    pool = pd.concat(pools, ignore_index=True)
    names = list(pools and [c[3:] for c in pool.columns if c.startswith("p::")])

    d = pd.DataFrame(diags)
    show("1. STAGE 2: how much weight does stage 1 earn against Pinnacle?",
         d[["test", "n_window", "n_test", "w", "w2"]],
         "w, w2 = 0 means: trust Pinnacle completely. Fitted on the training window,",
         "  out of fold, never on the test season.")
    ll = d[[c for c in d.columns if c.startswith("ll ")]].T
    ll.columns = d["test"].tolist()
    show("   Log loss on the test season (lower is better):", ll.reset_index(names="arm"),
         "Pinnacle closing is the ceiling no opening-time arm should reach.")

    # 2. every arm, the same fixed rule, the same books ----------------------
    rows = []
    for name in names:
        arm = pool.assign(p_fair=pool[f"p::{name}"],
                          value=pool["odds"] * pool[f"p::{name}"] - 1.0)
        b = primary_bets(arm)
        rows.append({"arm": name, **_vsum(b, "", "clv_pinnacle"), **_vrandom(arm, b)})
    show("2. THE SAME FIXED RULE, EACH ARM's PROBABILITIES, THE SAME SOFT BOOKS:",
         pd.DataFrame(rows).drop(columns="set"),
         "clv%: value against Pinnacle's CLOSING fair price -- the sharpest judge.",
         "Arm (a) at w = 0 is the no-model strategy; a model that helps must beat it.")

    # 3. by season -----------------------------------------------------------
    rows = []
    for name in names[:3]:
        arm = pool.assign(p_fair=pool[f"p::{name}"],
                          value=pool["odds"] * pool[f"p::{name}"] - 1.0)
        for season, g in primary_bets(arm).groupby("Season"):
            rows.append({"arm": name, **_vsum(g, season, "clv_pinnacle")})
    show("3. By test season:", pd.DataFrame(rows))

    # 4. does stage 1 agree with the Pinnacle value bets? -------------------
    base = pool.assign(p_fair=pool["p_pin"], value=pool["odds"] * pool["p_pin"] - 1.0)
    b = primary_bets(base)
    agree = b["p_stage1"] > b["p_pin"]
    show("4. THE PINNACLE VALUE BETS, SPLIT BY WHETHER STAGE 1 AGREES (declared in advance):",
         pd.DataFrame([_vsum(b[agree], "stage 1 agrees", "clv_pinnacle"),
                       _vsum(b[~agree], "stage 1 disagrees", "clv_pinnacle"),
                       _vsum(b, "all (the no-model strategy)", "clv_pinnacle")]),
         "If stage 1 carries information Pinnacle lacks, 'agrees' beats 'disagrees'",
         "  on BOTH closing-line value and profit.")
    rows = []
    for season, g in b.groupby("Season"):
        ag = g["p_stage1"] > g["p_pin"]
        ca, cd = _vsum(g[ag], "", "clv_pinnacle"), _vsum(g[~ag], "", "clv_pinnacle")
        rows.append({"season": season,
                     "n_agree": ca.get("n_bets"), "clv_agree%": ca.get("clv%"),
                     "roi_agree%": ca.get("roi%"), "units_won_agree": ca.get("units_won"),
                     "n_disagree": cd.get("n_bets"), "clv_disagree%": cd.get("clv%"),
                     "roi_disagree%": cd.get("roi%"), "units_won_disagree": cd.get("units_won")})
    t = pd.DataFrame(rows)
    if len(t):
        t["agree_better_clv"] = t["clv_agree%"] > t["clv_disagree%"]
        total = {"season": "ALL", "n_agree": t["n_agree"].sum(),
                 "units_won_agree": t["units_won_agree"].sum(),
                 "n_disagree": t["n_disagree"].sum(),
                 "units_won_disagree": t["units_won_disagree"].sum()}
        total["roi_agree%"] = total["units_won_agree"] / total["n_agree"] * 100
        total["roi_disagree%"] = total["units_won_disagree"] / total["n_disagree"] * 100
        t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)
    show("   ...and season by season:", t,
         f"'agrees' has the higher closing-line value in "
         f"{int(t['agree_better_clv'].fillna(False).astype(bool).sum()) if len(t) else 0} "
         f"of {max(len(t) - 1, 0)} seasons.",
         "units_won: net profit at 1 unit per bet; units staked = the number of bets.",
         "A real signal wins most seasons; a fluke wins about half.")

    print("\n" + "=" * 70)
    print("Read section 1 first. w above zero AND arm (a) beating the Pinnacle-only")
    print("arm in section 2 means the two-stage model adds an edge. w at zero means")
    print("the sharp price already holds what stage 1 knows.")
    print("=" * 70)
    return pool, d


# =========================================================================
# PART 25 -- GETTING MORE SEASONS: download and stack
#
# `download` fetches football-data.co.uk's main-league files, one CSV per
# division per season, into a folder. Run it on your own computer -- it needs
# ordinary internet access and nothing beyond Python's standard library. It
# skips files it already has, so it can be rerun safely after an interruption.
#
# `stack` merges a folder of those files with an existing stacked file (such
# as matches_multiseason.csv) into one file in the layout every command here
# reads: one row per match, a Season column, ISO dates. Where the same match
# appears in both, the existing file's row is kept.
#
# WHICH SEASONS. Pinnacle's opening prices start in 2012/13 and its closing
# prices in 2019/20, so 2012/13-2018/19 can test profit but not closing-line
# value, and 2019/20 onward can test both.
# =========================================================================

MAIN_DIVISIONS = ["E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3",
                  "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2",
                  "N1", "B1", "P1", "T1", "G1"]
FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"


def season_codes(first: int, last: int) -> list:
    """2012, 2021 -> ['1213', '1314', ..., '2122']  (first year of each season)."""
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first, last + 1)]


def _fetch(url: str) -> bytes:
    """
    One file over HTTPS, with certificate checks ON, three ways in turn.

    Python from python.org on a Mac does not use the system's certificates and
    fails every HTTPS request with CERTIFICATE_VERIFY_FAILED. So: Python's own
    store first; then the `certifi` bundle if it is installed; then the
    system's curl, which on a Mac uses the Mac's certificates. Verification is
    never switched off.
    """
    import ssl
    import subprocess
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    errors = []
    contexts = [None]
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    for ctx in contexts:
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                return r.read()
        except Exception as exc:
            errors.append(str(exc))
            if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
                raise                       # a real network or HTTP error
    try:                                    # the system's own curl
        out = subprocess.run(["curl", "-sSfL", "--max-time", "30", "-A", "Mozilla/5.0", url],
                             capture_output=True, check=True)
        return out.stdout
    except FileNotFoundError:
        errors.append("curl not found")
    except subprocess.CalledProcessError as exc:
        errors.append(exc.stderr.decode(errors="replace").strip())
    raise RuntimeError(" | ".join(errors))


def download(folder: str = "raw_history", first: int = 2012, last: int = 2021,
             divisions=None, pause: float = 1.0) -> None:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    divisions = divisions or MAIN_DIVISIONS
    jobs = [(sc, dv) for sc in season_codes(first, last) for dv in divisions]
    got = skipped = failed = 0
    print(f"{len(jobs)} files -> {out.resolve()}")
    for i, (sc, dv) in enumerate(jobs, 1):
        dest = out / f"{sc}_{dv}.csv"
        if dest.exists() and dest.stat().st_size > 1000:
            skipped += 1
            continue
        try:
            data = _fetch(FD_URL.format(season=sc, div=dv))
            if len(data) < 1000 or not data.lstrip(b"\xef\xbb\xbf").startswith(b"Div"):
                raise ValueError("not a match file")
            dest.write_bytes(data)
            got += 1
            print(f"  [{i:3d}/{len(jobs)}] {sc} {dv:4s} ok", flush=True)
        except Exception as exc:                       # keep going; report at end
            failed += 1
            print(f"  [{i:3d}/{len(jobs)}] {sc} {dv:4s} FAILED: {exc}", flush=True)
            if failed == 5 and got == 0:
                raise SystemExit(
                    "\nThe first five files all failed, so this is a setup problem, not "
                    "missing data.\nIf the errors say CERTIFICATE_VERIFY_FAILED, run:\n"
                    "    pip install certifi\nand try again.")
        time.sleep(pause)                             # be polite to the server
    print(f"\ndownloaded {got}, already had {skipped}, failed {failed}")
    if failed:
        print("Rerun the same command to retry the failures; finished files are skipped.")


def _parse_fd_dates(s: pd.Series) -> pd.Series:
    """football-data dates: dd/mm/yy in old files, dd/mm/yyyy later, ISO in stacked ones."""
    s = s.astype(str).str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    long = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return iso.fillna(long).fillna(short)


def stack(folder: str, existing: str = None, out: str = "matches_all_seasons.csv") -> pd.DataFrame:
    frames, report = [], []
    for f in sorted(Path(folder).glob("*.csv")):
        try:
            d = read_match_table(f)
        except Exception as exc:
            report.append((f.name, f"unreadable: {exc}"))
            continue
        d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
        d = d.dropna(subset=[c for c in ("HomeTeam", "AwayTeam") if c in d.columns])
        if d.empty or "Div" not in d.columns:
            report.append((f.name, "no matches"))
            continue
        stem = f.stem.split("_")[0]
        if "Season" not in d.columns:
            if not (len(stem) == 4 and stem.isdigit()):
                report.append((f.name, "skipped: name it like 2122_E0.csv so the season is known"))
                continue
            d["Season"] = stem
        frames.append(d)
    if not frames:
        raise SystemExit(f"no usable files in {folder}")
    new = pd.concat(frames, ignore_index=True, sort=False)
    new["Date"] = _parse_fd_dates(new["Date"])
    parts = [new.copy().assign(_src=1)]
    if existing:
        old = read_match_table(existing)
        old = old.loc[:, ~old.columns.astype(str).str.startswith("Unnamed")]
        old["Date"] = _parse_fd_dates(old["Date"])
        parts.append(old.copy().assign(_src=0))
    allm = pd.concat(parts, ignore_index=True, sort=False)
    allm["Season"] = allm["Season"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    allm = allm.dropna(subset=["Date"])
    key = ["Div", "Date", "HomeTeam", "AwayTeam"]
    n0 = len(allm)
    allm = (allm.sort_values("_src").drop_duplicates(key, keep="first")
                .drop(columns="_src").sort_values(["Date", "Div"]))
    allm["Date"] = allm["Date"].dt.strftime("%Y-%m-%d")
    allm.to_csv(out, index=False)

    cov = allm.groupby("Season").agg(
        matches=("Div", "size"), divisions=("Div", "nunique"),
        pinnacle_open=("PSH", lambda x: x.notna().mean()) if "PSH" in allm else ("Div", "size"),
        pinnacle_close=("PSCH", lambda x: x.notna().mean()) if "PSCH" in allm else ("Div", "size"))
    for c in ("pinnacle_open", "pinnacle_close"):
        if c in cov:
            cov[c] = (cov[c] * 100).round(0).astype(int).astype(str) + "%"
    show(f"Stacked {len(allm):,} matches into {out}  ({n0 - len(allm):,} duplicates dropped)",
         cov.reset_index())
    for name, why in report:
        print(f"  {name}: {why}")
    return allm


# =========================================================================
# PART 26 -- ROBUSTNESS OF THE AGREEMENT RESULT, AND THE MODEL IN THE SCANNER
#
# `pinnacle-sweep` reruns PART 24 under several seeds and learners and asks one
# question of each run: in how many test seasons did the value bets stage 1
# agrees with keep more closing-line value than the ones it disagrees with?
# A real signal wins most seasons under every seed and learner; an accident of
# one seed does not. Elo does not depend on the seed, so it is fitted once per
# season and reused by every run in the sweep.
#
# `scan --history FILE` adds stage 1's view to the live scanner. It trains the
# same market-blind engine the backtest used, on every played match in FILE,
# with Elo fitted on the same matches, and marks each value bet AGREE (stage 1
# also puts that side above the reference's fair price) or DISAGREE. The rule
# for WHICH bets to list is unchanged -- VALUE_PRIMARY -- the flag only ranks
# them. Training takes a few minutes, so the fitted model is cached next to the
# history file and reused until that file changes.
# =========================================================================


def agreement_summary(pool: pd.DataFrame) -> dict:
    """The agreement test on one run's pool of soft-book candidates."""
    b = primary_bets(pool.assign(p_fair=pool["p_pin"],
                                 value=pool["odds"] * pool["p_pin"] - 1.0))
    ag = b["p_stage1"] > b["p_pin"]
    per = b.assign(agree=ag).groupby(["Season", "agree"])["clv_pinnacle"].mean().unstack()
    won = int((per.get(True) > per.get(False)).sum()) if per.shape[1] == 2 else 0
    n_seasons = int(per.dropna().shape[0])
    from math import comb
    p_sign = sum(comb(n_seasons, k) for k in range(won, n_seasons + 1)) / 2 ** n_seasons \
        if n_seasons else np.nan
    return {"seasons_won": won, "seasons": n_seasons, "sign_test_p": p_sign,
            "clv_agree%": b.loc[ag, "clv_pinnacle"].mean() * 100,
            "clv_disagree%": b.loc[~ag, "clv_pinnacle"].mean() * 100,
            "roi_agree%": b.loc[ag, "pnl"].mean() * 100,
            "roi_disagree%": b.loc[~ag, "pnl"].mean() * 100,
            "units_won_agree": float(b.loc[ag, "pnl"].sum()),
            "units_won_disagree": float(b.loc[~ag, "pnl"].sum()),
            "n_agree": int(ag.sum()), "n_disagree": int((~ag).sum())}


def pinnacle_sweep(source: str, seeds, learners, out: str = None) -> pd.DataFrame:
    if "torch" in learners and "lgb" in learners:
        raise SystemExit("torch and lgb cannot run in one process (the OpenMP crash). "
                         "Run them as two separate sweeps.")
    rows = []
    t_all = time.time()
    for learner in learners:
        for seed in seeds:
            configure(learner=learner, ensemble_k=1)
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pool, diag = pinnacle_main(source, mode="multiseason", seed=seed)
            r = {"learner": learner, "seed": seed,
                 "w_median": float(diag["w"].median()), "w_max": float(diag["w"].max()),
                 **agreement_summary(pool), "minutes": round((time.time() - t0) / 60, 1)}
            rows.append(r)
            print(f"  {learner:8s} seed {seed}: agree wins {r['seasons_won']}/{r['seasons']} "
                  f"seasons, CLV {r['clv_agree%']:+.2f}% vs {r['clv_disagree%']:+.2f}%, "
                  f"median w {r['w_median']:.3f}  [{r['minutes']} min]", flush=True)
            if out:
                pd.DataFrame(rows).to_csv(out, index=False)     # survive interruption
    t = pd.DataFrame(rows)
    show("PINNACLE SWEEP -- the agreement result under every seed and learner:", t,
         "seasons_won: test seasons where 'stage 1 agrees' kept more closing-line",
         "  value than 'disagrees'. sign_test_p: chance of at least that many by luck.",
         f"{time.time() - t_all:.0f}s in all.")
    return t


# --- the model inside the live scanner -----------------------------------

def _model_cache_path(history: str) -> Path:
    h = Path(history).resolve()
    st = h.stat()
    return h.parent / f".{h.stem}.stage1v2_{LEARNER}_{int(st.st_mtime)}_{st.st_size}.joblib"


def _build_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Every feature, for played and unplayed matches alike (no season dropped)."""
    built = add_shot_features(add_market_features(calculate_rolling_features(raw)))
    return attach_external_features(add_league_context(built))


def _scan_tab(allm: pd.DataFrame, params: tuple) -> pd.DataFrame:
    """Elo and proxy xG for every row, from (Elo constants, proxy-xG coefficients)."""
    elo_p, coefs = params
    tab = run_elo(allm, elo_p)
    tab["match_key"] = allm["match_key"].to_numpy()
    tab = tab.drop_duplicates("match_key")
    px = pxg_features(allm, coefs).drop_duplicates("match_key")
    return tab.merge(px, on="match_key", how="left")


def model_view(history: str, fixtures: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stage 1's probabilities for each fixture, trained on every played match."""
    import joblib
    hist = clean_prices(apply_column_renames(read_match_table(history)), verbose=False)
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist["Season"] = hist["Season"].astype(str)
    hist = hist.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    fx = fixtures.copy()
    fx["Season"] = hist.sort_values("Date")["Season"].iloc[-1]   # the current season
    for c in ("FTHG", "FTAG", "FTR"):
        fx[c] = np.nan
    fx["_fixture"] = True
    hist["_fixture"] = False
    # a fixture already in the history file (with its result) is not a fixture
    key = ["Div", "Date", "HomeTeam", "AwayTeam"]
    fx = fx[~fx.set_index(key).index.isin(hist.set_index(key).index)]
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        allm = _build_feature_table(pd.concat([hist, fx], ignore_index=True, sort=False))
    played = allm[~allm["_fixture"].astype(bool) & allm["target"].notna()]
    upcoming = allm[allm["_fixture"].astype(bool)]
    if upcoming.empty:
        return pd.DataFrame(columns=key + [f"model_p_{o}" for o in OUTCOMES])

    cache = _model_cache_path(history)
    if cache.exists():
        engine, elo_params = joblib.load(cache)
        print(f"  stage 1: using the model cached from {Path(history).name}")
    else:
        print(f"  stage 1: training on {len(played):,} played matches "
              f"(once per history file; cached afterwards) ...", flush=True)
        t0 = time.time()
        fit_on = played[played["FTR"].isin(["H", "D", "A"])]
        elo_params, _ = fit_elo(fit_on)
        elo_params = (elo_params, fit_pxg(fit_on))
        train = put_elo(played, _scan_tab(allm, elo_params))
        train = train[train[STAGE1_REQUIRED].notna().all(axis=1)].reset_index(drop=True)
        engine = MatchEngine(seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof = engine.fit_predict_oof(train)
            engine.fit(train)
            engine.fit_rho(oof, train)
        joblib.dump((engine, elo_params), cache)
        print(f"  stage 1: trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    if not isinstance(elo_params, tuple):          # a model cached before proxy xG
        elo_params = (elo_params, None)
    up = put_elo(upcoming, _scan_tab(allm, elo_params)).reset_index(drop=True)
    ok = up[STAGE1_REQUIRED].notna().all(axis=1).to_numpy()
    probs = np.full((len(up), 3), np.nan)
    if ok.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs[ok] = engine.predict_proba(up[ok].reset_index(drop=True))
    out = up[key].copy()
    for k, o in enumerate(OUTCOMES):
        out[f"model_p_{o}"] = probs[:, k]
    if (~ok).any():
        print(f"  stage 1: no view for {(~ok).sum()} fixtures (teams without enough history)")
    return out


# =========================================================================
# PART 27 -- A LEARNED STAGE 2: WHICH VALUE BETS WILL THE MARKET CONFIRM?
#
# PART 24 found that stage 1 cannot improve Pinnacle's probability (w = 0),
# and PART 26 that a crude yes/no -- does stage 1 also rate the side above
# Pinnacle's fair price? -- picks value bets that keep more value by kickoff,
# in 10 to 12 of 12 seasons. This replaces the yes/no with a model.
#
#   STAGE 1  the market-blind match engine (unchanged; out of sample).
#   STAGE 2  for every value bet, predict its closing-line value: what the bet
#            is worth by Pinnacle's CLOSING fair price. Inputs are only what is
#            known when the opening price is quoted -- stage 1's view and its gap
#            to Pinnacle, the price, the book, the league, Elo, rest, promotion,
#            and how far the best price sits above the average.
#
# WALK-FORWARD, TWICE. Stage 1's predictions for season s come from a model
# trained on seasons before s (PART 24). Stage 2 for season t is trained on the
# value bets of seasons before t, and its keep/drop threshold is read off its
# own out-of-fold predictions on those seasons. Nothing from season t is used
# to decide anything about season t.
#
# FIXED BEFORE RUNNING: keep a value bet when stage 2's prediction is at or
# above the median of its out-of-fold predictions on the training seasons --
# so it keeps about half. The comparison that matters is KEPT against DROPPED,
# season by season, and the learned stage 2 against the simple agree/disagree.
# =========================================================================

S2_MATCH_CONTEXT = ["elo_diff", "elo_home", "elo_away", "diff_rest_days",
                    "home_rest", "away_rest", "home_is_promoted", "away_is_promoted",
                    "home_is_new", "away_is_new", "home_prev_ppg", "away_prev_ppg",
                    "Tier", "min_played", "league_goals_20", "league_home_rate_20"]
S2_CATEGORICAL = ["book", "side", "Div"]
# Everything stage 1 contributes. The CONTROL stage 2 is trained without these,
# so the difference between the two is what stage 1 adds -- not what the size of
# the value, the book or the league would have said on their own.
S2_STAGE1_INPUTS = ["gap", "gap_rel", "p_stage1", "ctx_p1_H", "ctx_p1_D", "ctx_p1_A"]
# The team-level match information stage 1 is built from. The MARKET-ONLY
# control drops these as well, keeping only prices, margins, the book, the side
# and the league -- what a bettor with no model at all could see.
S2_MATCH_INPUTS = ["ctx_elo_diff", "ctx_elo_home", "ctx_elo_away", "ctx_diff_rest_days",
                   "ctx_home_rest", "ctx_away_rest", "ctx_home_is_promoted",
                   "ctx_away_is_promoted", "ctx_home_is_new", "ctx_away_is_new",
                   "ctx_home_prev_ppg", "ctx_away_prev_ppg"]


def _s2_frame(b: pd.DataFrame, cats: dict) -> pd.DataFrame:
    """Stage 2 inputs for a set of value bets. Categories are coded from TRAINING."""
    X = pd.DataFrame(index=b.index)
    X["gap"] = b["p_stage1"] - b["p_pin"]
    X["gap_rel"] = X["gap"] / b["p_pin"]
    X["p_stage1"] = b["p_stage1"]
    X["p_pin"] = b["p_pin"]
    X["value"] = b["value"]
    X["log_odds"] = np.log(b["odds"])
    X["spread_side"] = b["spread_side"]
    for c in [c for c in b.columns if c.startswith("ctx_")]:
        X[c] = b[c]
    for c in S2_CATEGORICAL:
        X[f"cat_{c}"] = b[c].map(cats[c]).fillna(-1).astype(float)
    bad = [c for c in X.columns if any(t.lower() in c.lower()
                                       for t in CLOSE_TOKENS + ("clv", "pnl", "win"))]
    if bad:
        raise ValueError(f"closing-time or outcome data in stage 2 inputs: {bad}")
    return X


def _s2_model(seed: int):
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, max_depth=4,
        min_samples_leaf=50, l2_regularization=1.0, random_state=seed,
        categorical_features=None)


def stage2_value(pool: pd.DataFrame, seed: int = 42, min_train_seasons: int = 1,
                 drop_inputs: tuple = ()) -> tuple:
    """Walk-forward stage 2 over the value bets in `pool` (one row per bet)."""
    bets = primary_bets(pool.assign(p_fair=pool["p_pin"],
                                    value=pool["odds"] * pool["p_pin"] - 1.0)).copy()
    bets["Season"] = bets["Season"].astype(str)
    bets = bets.dropna(subset=["clv_pinnacle", "p_stage1"]).reset_index(drop=True)
    seasons = sorted(bets["Season"].unique())
    out, importances = [], None
    for i in range(min_train_seasons, len(seasons)):
        tr = bets[bets["Season"].isin(seasons[:i])]
        te = bets[bets["Season"] == seasons[i]]
        if len(tr) < 300 or te.empty:
            continue
        cats = {c: {v: j for j, v in enumerate(sorted(tr[c].astype(str).unique()))}
                for c in S2_CATEGORICAL}
        Xtr = _s2_frame(tr.assign(**{c: tr[c].astype(str) for c in S2_CATEGORICAL}), cats)
        Xte = _s2_frame(te.assign(**{c: te[c].astype(str) for c in S2_CATEGORICAL}), cats)
        if drop_inputs:
            Xtr = Xtr.drop(columns=[c for c in drop_inputs if c in Xtr.columns])
            Xte = Xte.drop(columns=[c for c in drop_inputs if c in Xte.columns])
        y = tr["clv_pinnacle"].clip(-0.5, 0.5).to_numpy()   # tame a few wild prices
        oof = cross_val_predict(_s2_model(seed), Xtr, y, cv=3)
        thr = float(np.median(oof))
        m = _s2_model(seed).fit(Xtr, y)
        pred = m.predict(Xte)
        out.append(te.assign(s2_pred=pred, s2_keep=pred >= thr, s2_thr=thr,
                             agree=te["p_stage1"] > te["p_pin"],
                             Tier=te["ctx_Tier"] if "ctx_Tier" in te.columns else np.nan))
        if i == len(seasons) - 1:                       # explain the latest model
            from sklearn.inspection import permutation_importance
            yt = te["clv_pinnacle"].clip(-0.5, 0.5).to_numpy()
            pi = permutation_importance(m, Xte, yt, n_repeats=5, random_state=seed)
            importances = (pd.DataFrame({"feature": Xte.columns,
                                         "importance": pi.importances_mean})
                           .sort_values("importance", ascending=False))
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), importances


def _grp(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"n": 0}
    return {"n": len(b), "clv%": b["clv_pinnacle"].mean() * 100,
            "roi%": b["pnl"].mean() * 100, "units_won": b["pnl"].sum()}


def stage2_report(res: pd.DataFrame, importances: pd.DataFrame,
                  control: pd.DataFrame = None, market_only: pd.DataFrame = None) -> None:
    from math import comb
    from scipy.stats import spearmanr
    rows = []
    for season, g in res.groupby("Season"):
        k, d = g[g["s2_keep"]], g[~g["s2_keep"]]
        a, na = g[g["agree"]], g[~g["agree"]]
        rho = spearmanr(g["s2_pred"], g["clv_pinnacle"]).statistic
        rows.append({"season": season, "bets": len(g), "rank_corr": rho,
                     **{f"keep_{x}": v for x, v in _grp(k).items()},
                     **{f"drop_{x}": v for x, v in _grp(d).items()},
                     "agree_clv%": a["clv_pinnacle"].mean() * 100,
                     "disagree_clv%": na["clv_pinnacle"].mean() * 100})
    t = pd.DataFrame(rows)
    t["keep_beats_drop"] = t["keep_clv%"] > t["drop_clv%"]
    t["keep_beats_agree"] = t["keep_clv%"] > t["agree_clv%"]
    show("1. STAGE 2 BY TEST SEASON (value vs Pinnacle's closing price, flat 1 unit):",
         t[["season", "bets", "rank_corr", "keep_n", "keep_clv%", "keep_roi%",
            "keep_units_won", "drop_n", "drop_clv%", "drop_roi%", "drop_units_won",
            "agree_clv%", "disagree_clv%", "keep_beats_drop", "keep_beats_agree"]],
         "rank_corr: does stage 2's ranking of value bets match how they closed?",
         "keep / drop: stage 2's prediction above / below its training-season median.")

    def sign_p(won, n):
        return sum(comb(n, j) for j in range(won, n + 1)) / 2 ** n
    n = len(t)
    kd, ka = int(t["keep_beats_drop"].sum()), int(t["keep_beats_agree"].sum())
    print(f"\n  KEPT beat DROPPED in {kd} of {n} seasons   (chance by luck {sign_p(kd, n):.4f})")
    print(f"  KEPT beat the simple AGREE set in {ka} of {n} seasons   "
          f"(chance by luck {sign_p(ka, n):.4f})")
    print(f"  median rank correlation {t['rank_corr'].median():+.3f}; positive in "
          f"{int((t['rank_corr'] > 0).sum())} of {n} seasons")

    pooled = []
    for label, b in (("all value bets (no stage 2)", res),
                     ("stage 2 KEEPS", res[res["s2_keep"]]),
                     ("stage 2 DROPS", res[~res["s2_keep"]]),
                     ("simple filter: stage 1 AGREES", res[res["agree"]]),
                     ("simple filter: stage 1 DISAGREES", res[~res["agree"]])):
        pooled.append({**_vsum(b, "", "clv_pinnacle"), "set": label})
    if control is not None and len(control):
        pooled.append({**_vsum(control[control["s2_keep"]], "", "clv_pinnacle"),
                       "set": "CONTROL stage 2 (no stage 1) KEEPS"})
        pooled.append({**_vsum(control[~control["s2_keep"]], "", "clv_pinnacle"),
                       "set": "CONTROL stage 2 (no stage 1) DROPS"})
    if market_only is not None and len(market_only):
        pooled.append({**_vsum(market_only[market_only["s2_keep"]], "", "clv_pinnacle"),
                       "set": "MARKET-ONLY stage 2 KEEPS"})
        pooled.append({**_vsum(market_only[~market_only["s2_keep"]], "", "clv_pinnacle"),
                       "set": "MARKET-ONLY stage 2 DROPS"})
    tp = pd.DataFrame(pooled)
    show("2. POOLED OVER ALL STAGE 2 TEST SEASONS:",
         tp[["set"] + [c for c in tp.columns if c != "set"]],
         "units_staked = bets (1 unit each); units_won = net profit.")

    if control is not None and len(control):
        # Compared at the SAME number of kept bets -- the top half of each
        # season by each model's own prediction -- because keeping fewer bets
        # raises closing-line value by itself and would favour the pickier one.
        # Plus the rank correlation, which needs no threshold at all.
        rows = []
        for season, g in res.groupby("Season"):
            c = control[control["Season"] == season]
            n = len(g) // 2
            gf = g.nlargest(n, "s2_pred")
            gc = c.nlargest(n, "s2_pred")
            rows.append({"season": season, "kept_each": n,
                         "full_top_half_clv%": gf["clv_pinnacle"].mean() * 100,
                         "control_top_half_clv%": gc["clv_pinnacle"].mean() * 100,
                         "full_units": gf["pnl"].sum(), "control_units": gc["pnl"].sum(),
                         "full_rank_corr": spearmanr(g["s2_pred"], g["clv_pinnacle"]).statistic,
                         "control_rank_corr": spearmanr(c["s2_pred"], c["clv_pinnacle"]).statistic})
        tc = pd.DataFrame(rows)
        tc["full_better_half"] = tc["full_top_half_clv%"] > tc["control_top_half_clv%"]
        tc["full_better_rank"] = tc["full_rank_corr"] > tc["control_rank_corr"]
        wh, wr = int(tc["full_better_half"].sum()), int(tc["full_better_rank"].sum())
        show("2b. DOES STAGE 1 ADD ANYTHING? Full stage 2 against the same stage 2 "
             "trained WITHOUT stage 1's inputs, keeping the same number of bets:", tc,
             f"Full beats control on its top half in {wh} of {len(tc)} seasons "
             f"(luck: {sign_p(wh, len(tc)):.4f}); on ranking in {wr} of {len(tc)} "
             f"(luck: {sign_p(wr, len(tc)):.4f}).",
             "This, not section 1, is the test of the two-stage claim.")

    if market_only is not None and len(market_only):
        rows = []
        for season, g in res.groupby("Season"):
            n = len(g) // 2
            r = {"season": season, "kept_each": n}
            for label, d in (("full", res), ("no_stage1", control), ("market_only", market_only)):
                gg = d[d["Season"] == season].nlargest(n, "s2_pred")
                r[f"{label}_clv%"] = gg["clv_pinnacle"].mean() * 100
                r[f"{label}_units"] = gg["pnl"].sum()
            rows.append(r)
        tm = pd.DataFrame(rows)
        tm["match_info_helps"] = tm["no_stage1_clv%"] > tm["market_only_clv%"]
        tm["full_beats_market_only"] = tm["full_clv%"] > tm["market_only_clv%"]
        a, b2 = int(tm["match_info_helps"].sum()), int(tm["full_beats_market_only"].sum())
        tot = tm[[c for c in tm.columns if c.endswith("_units")]].sum()
        show("2c. MATCH INFORMATION vs MARKET INFORMATION (top half of each season, "
             "each version ranked by its own prediction):", tm,
             f"Raw match information (Elo, rest, table) beats market-only in {a} of "
             f"{len(tm)} seasons (luck: {sign_p(a, len(tm)):.4f}).",
             f"The full two-stage version beats market-only in {b2} of {len(tm)} "
             f"(luck: {sign_p(b2, len(tm)):.4f}).",
             f"Units won over all seasons -- full {tot['full_units']:+.1f}, "
             f"no stage 1 {tot['no_stage1_units']:+.1f}, market only "
             f"{tot['market_only_units']:+.1f}.")

    # deciles of the stage 2 prediction: does value rise with it?
    res = res.assign(decile=res.groupby("Season")["s2_pred"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1))
    q = res.groupby("decile").agg(bets=("pnl", "size"),
                                  clv_pct=("clv_pinnacle", lambda x: x.mean() * 100),
                                  roi_pct=("pnl", lambda x: x.mean() * 100),
                                  units_won=("pnl", "sum")).reset_index()
    q = q.rename(columns={"decile": "quintile (1 = lowest prediction)"})
    show("3. BY QUINTILE OF STAGE 2's PREDICTION (within each season):", q.round(3),
         "A model that works shows closing-line value rising from quintile 1 to 5.")
    def by_bins(col, label, q=5):
        x = res[col]
        if x.notna().sum() < 50:
            return None
        bins = pd.qcut(x.rank(method="first"), q, labels=False) + 1
        g = res.assign(_b=bins).groupby("_b")
        return pd.DataFrame({"input": label, "level (1 = low)": g.size().index,
                             "range": [f"{x[bins == k].min():.3f} .. {x[bins == k].max():.3f}"
                                       for k in g.size().index],
                             "bets": g.size().values,
                             "clv%": (g["clv_pinnacle"].mean() * 100).values,
                             "roi%": (g["pnl"].mean() * 100).values,
                             "units_won": g["pnl"].sum().values})
    parts = []
    work = res.assign(gap=res["p_stage1"] - res["p_pin"])
    res_backup, res = res, work
    for col, label in (("value", "value vs Pinnacle fair"),
                       ("ctx_pin_overround", "Pinnacle's margin"),
                       ("gap", "stage 1 minus Pinnacle"),
                       ("odds", "odds")):
        t5 = by_bins(col, label) if col in res.columns else None
        if t5 is not None:
            parts.append(t5)
    res = res_backup
    if parts:
        show("5. WHICH WAY DOES EACH INPUT POINT? Value bets by level of each input "
             "(all test seasons):", pd.concat(parts, ignore_index=True).round(3),
             "Rising clv% across levels = more of that input means the value is more real.")
    cat_rows = []
    for col in ("side", "book", "Tier"):
        if col in res.columns:
            g = res.groupby(res[col].astype(str))
            cat_rows.append(pd.DataFrame({
                "input": col, "level": g.size().index, "bets": g.size().values,
                "clv%": (g["clv_pinnacle"].mean() * 100).values,
                "roi%": (g["pnl"].mean() * 100).values,
                "units_won": g["pnl"].sum().values}))
    if cat_rows:
        ct = pd.concat(cat_rows, ignore_index=True)
        show("   ...and by side, bookmaker and league tier:",
             ct[ct["bets"] >= 50].round(3),
             "Side: H home, D draw, A away. Small groups (< 50 bets) are left out.")

    if importances is not None:
        show("4. WHAT STAGE 2 RELIES ON (latest season, permutation importance):",
             importances.head(12).round(5),
             "Higher = predictions get worse when that input is scrambled.",
             "This is the starting point for the 'what do bookmakers get wrong' chapter.")


def stage2_main(source: str, seed: int = 42, pool_file: str = None, reuse: bool = False,
                min_tier: int = 1, exclude_books=()):
    print("=" * 70)
    print("LEARNED STAGE 2 -- predicting which value bets the market will confirm")
    print(f"  leagues: {tier_label(min_tier)}")
    print("=" * 70)
    if reuse and pool_file and Path(pool_file).exists():
        pool = pd.read_csv(pool_file, low_memory=False)
        print(f"  reusing the stage 1 pool in {pool_file} ({len(pool):,} rows)")
    else:
        print("  building the stage 1 pool (the PART 24 walk-forward) ...", flush=True)
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pool, _ = pinnacle_main(source, mode="multiseason", seed=seed)
        print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)
        if pool_file:
            pool.to_csv(pool_file, index=False)
            print(f"  saved to {pool_file}; rerun with --reuse to skip this step")
    if min_tier > 1:
        # Stage 1 was trained on every league; stage 2 is trained AND tested on
        # the chosen tiers only -- a lower-league strategy, judged on its own.
        tiers = pool["Div"].map(tier_of)
        pool = pool[tiers >= min_tier].reset_index(drop=True)
        print(f"  stage 2 on {', '.join(sorted(pool['Div'].unique()))} only "
              f"({len(pool):,} candidate rows)")
    if exclude_books:
        pool = pool[~pool["book"].isin(exclude_books)].reset_index(drop=True)
        print(f"  bookmakers excluded: {', '.join(exclude_books)}")
    t0 = time.time()
    res, imp = stage2_value(pool, seed=seed)
    if res.empty:
        raise SystemExit("not enough seasons for a stage 2 walk-forward")
    control, _ = stage2_value(pool, seed=seed, drop_inputs=tuple(S2_STAGE1_INPUTS))
    market_only, _ = stage2_value(pool, seed=seed,
                                  drop_inputs=tuple(S2_STAGE1_INPUTS + S2_MATCH_INPUTS))
    print(f"  stage 2 walk-forward over {res['Season'].nunique()} test seasons, "
          f"{len(res):,} value bets, plus two controls [{time.time() - t0:.0f}s]")
    stage2_report(res, imp, control, market_only)
    print("\n" + "=" * 70)
    print("Stage 2 earns its place if KEPT beats DROPPED in most seasons and the")
    print("quintiles rise. STAGE 1 earns its place only if the full stage 2 also")
    print("beats the control trained without it (section 2b).")
    print("=" * 70)
    return res


# =========================================================================
# PART 28 -- PLACEBO TESTS  (does the method find "value" where none can exist?)
#
# The value strategy uses Pinnacle as the truth and bets at the soft books. If
# the method had a flaw -- a leak, a settlement bug, a quirk of de-vigging --
# it would also find profit where there is none. So the roles are swapped:
#
#   REAL       truth = Pinnacle,                     bet at the soft books
#   PLACEBO 1  truth = Bet365 (a soft book),         bet at Pinnacle
#   PLACEBO 2  truth = the market average,           bet at Pinnacle
#   PLACEBO 3  truth = Pinnacle, SHUFFLED between    bet at the soft books
#              matches of the same season
#
# Every arm uses the same fixed rule (VALUE_PRIMARY), match-result market,
# opening prices, one bet per selection at its best-value venue, judged by
# profit and by value against Pinnacle's CLOSING fair price. A sound method
# makes money only in the REAL arm.
# =========================================================================

def placebo_main(source: str, seed: int = 0) -> pd.DataFrame:
    print("=" * 70)
    print("PLACEBO TESTS -- the same rule, with the roles swapped")
    print("=" * 70)
    df = clean_prices(apply_column_renames(read_match_table(source)))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Season"].astype(str)
    df = df[df["FTR"].isin(["H", "D", "A"])].dropna(subset=["Date"])
    need = PIN_OPEN + PIN_CLOSE
    df = df[df[need].notna().all(axis=1)].reset_index(drop=True)
    df["match_key"] = (df["Div"].astype(str) + "|" + df["Date"].dt.strftime("%Y-%m-%d")
                       + "|" + df["HomeTeam"].astype(str) + "|" + df["AwayTeam"].astype(str))
    print(f"{len(df):,} matches with Pinnacle opening and closing prices, "
          f"seasons {', '.join(sorted(df['Season'].unique()))}")
    y = df["FTR"].to_numpy()
    pin_close = shin_vec(df[PIN_CLOSE].to_numpy(float))
    pin_open = shin_vec(df[PIN_OPEN].to_numpy(float))
    rng = np.random.default_rng(seed)
    shuffled = pin_open.copy()
    for _, idx in df.groupby("Season").indices.items():        # within each season
        shuffled[idx] = pin_open[rng.permutation(idx)]

    def ref(cols):
        return shin_vec(df[cols].to_numpy(float)) if all(c in df.columns for c in cols) \
            else np.full((len(df), 3), np.nan)

    soft_books = [b for b in VALUE_BOOKS_1X2 if all(f"{b}{o}" in df.columns for o in OUTCOMES)]
    arms = {
        "REAL: truth = Pinnacle, bet at soft books": (pin_open, soft_books),
        "PLACEBO 1: truth = Bet365, bet at Pinnacle": (ref(["B365H", "B365D", "B365A"]), ["PS"]),
        "PLACEBO 2: truth = market average, bet at Pinnacle": (ref(["AvgH", "AvgD", "AvgA"]), ["PS"]),
        "PLACEBO 3: truth = Pinnacle shuffled, bet at soft books": (shuffled, soft_books),
    }
    pooled, seasons = [], []
    for name, (fair, venues) in arms.items():
        frames = []
        for book in venues:
            for k, o in enumerate(OUTCOMES):
                odds = df[f"{book}{o}"].to_numpy(float)
                val = odds * fair[:, k] - 1.0
                win = (y == o)
                keep = np.isfinite(val) & np.isfinite(odds) & (odds > 1.0)
                frames.append(pd.DataFrame({
                    "match_key": df["match_key"].to_numpy()[keep],
                    "Date": df["Date"].to_numpy()[keep],
                    "Season": df["Season"].to_numpy()[keep],
                    "market": "1X2", "snapshot": "open", "reference": "x",
                    "book": book, "side": o, "odds": odds[keep], "value": val[keep],
                    "pnl": np.where(win, odds - 1.0, -1.0)[keep],
                    "clv_pinnacle": (odds * pin_close[:, k] - 1.0)[keep]}))
        b = primary_bets(pd.concat(frames, ignore_index=True))
        pooled.append({"arm": name, **_vsum(b, "", "clv_pinnacle")})
        for season, g in b.groupby("Season"):
            seasons.append({"arm": name.split(":")[0], "season": season, "n_bets": len(g),
                            "units_won": g["pnl"].sum(), "roi%": g["pnl"].mean() * 100,
                            "clv%": g["clv_pinnacle"].mean() * 100})
    t = pd.DataFrame(pooled).drop(columns="set")
    show("POOLED OVER ALL SEASONS:",
         t[["arm", "n_bets", "units_won", "roi%", "roi_lo", "roi_hi", "roi_wk_lo",
            "roi_wk_hi", "clv%", "clv_wk_lo", "clv_wk_hi"]].round(3),
         "Only the REAL arm should make money. roi_wk / clv_wk: intervals that resample",
         "  whole weeks, so a weekend's matches stay together.")
    ts = pd.DataFrame(seasons)
    wins = ts.assign(pos=ts["units_won"] > 0).groupby("arm")["pos"].agg(["sum", "count"])
    show("SEASON BY SEASON (units won):",
         ts.pivot(index="season", columns="arm", values="units_won").round(1).reset_index(),
         "Seasons with a profit: " + ", ".join(f"{a} {int(r['sum'])} of {int(r['count'])}"
                                                for a, r in wins.iterrows()))
    return t

# =========================================================================
# PART 16 -- COMMAND LINE
# =========================================================================

DEFAULT_SOURCES = {"export": "raw/all-euro-data-2025-2026.csv",
                   "multiseason": "matches_all.csv"}
COMMANDS = ("run", "sweep", "elo", "clv", "ou", "traps", "ah", "value", "scan", "review", "pinnacle", "pinnacle-sweep", "stage2-value", "placebo", "download", "stack", "paper")


# =========================================================================
# PART 30 -- PAPER TESTING (the backtest run forward, with prices typed in)
#
#   1. paper template  -> a sheet: one row per match per bookmaker
#   2. you type Pinnacle's and each book's 1X2 prices AT THE SAME MOMENT
#   3. paper flag      -> value bets by the fixed rule (VALUE_PRIMARY), with
#                         German betting tax taken off the book's price, are
#                         appended to the paper log
#   4. you open the book's site, type the price you really see (live_odds)
#   5. near kickoff: Pinnacle's prices again (pin_close_*); after: result
#   6. paper report    -> availability, CLV, profit, and the targets fixed
#                         before the test started (PAPER_TARGET)
# =========================================================================
PAPER_BOOKS = ("bwin", "tipico", "bet365")
PAPER_TAX = 0.053                     # German betting tax, if the book passes it on
PAPER_TARGET = {"available_bets": 200, "min_availability": 0.50, "min_clv": 0.01}
PAPER_SHEET_COLS = ["Date", "Time", "Div", "HomeTeam", "AwayTeam",
                    "pin_h", "pin_d", "pin_a", "book", "book_h", "book_d", "book_a"]
PAPER_LOG_COLS = ["flagged_at", "Date", "Time", "Div", "HomeTeam", "AwayTeam", "side",
                  "book", "odds", "odds_after_tax", "p_fair", "value",
                  "checked_at", "live_odds",
                  "pin_close_h", "pin_close_d", "pin_close_a", "result"]


def _paper_read(path: str) -> pd.DataFrame:
    """Read a sheet saved by Excel/Numbers: comma or semicolon, BOM, decimal commas."""
    if not Path(path).exists():
        raise SystemExit(f"{path} not found -- check the name and the folder you are in.")
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip() if s.dtype == object else s)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False)
                          .replace({"": np.nan, "nan": np.nan, "None": np.nan}),
                         errors="coerce")


def paper_template(fixtures: str, out: str, books=PAPER_BOOKS, divs=None) -> None:
    fx = _paper_read(fixtures)
    if fx.empty:
        raise SystemExit(f"no fixtures in {fixtures}")
    if divs:
        fx = fx[fx["Div"].astype(str).isin(divs)]
    fx = fx.assign(Date=pd.to_datetime(fx["Date"], dayfirst=True).dt.strftime("%Y-%m-%d"),
                   Time=fx["Time"].fillna("") if "Time" in fx.columns else "")
    rows = [dict(Date=r.Date, Time=r.Time, Div=r.Div, HomeTeam=r.HomeTeam,
                 AwayTeam=r.AwayTeam, book=b)
            for r in fx.sort_values(["Date", "Time", "Div"]).itertuples() for b in books]
    pd.DataFrame(rows).reindex(columns=PAPER_SHEET_COLS).to_csv(out, index=False)
    print(f"wrote {out}: {len(fx)} matches x {len(books)} books = {len(rows)} rows")
    print("Type Pinnacle's prices on the first row of each match (the rest may stay")
    print("blank) and each book's prices on its own row -- all within a few minutes.")


def paper_flag(sheet: str, log: str, tax: float = PAPER_TAX) -> pd.DataFrame:
    s = _paper_read(sheet)
    miss = [c for c in PAPER_SHEET_COLS if c not in s.columns]
    if miss:
        raise SystemExit(f"{sheet} is missing columns: {', '.join(miss)}")
    key = ["Date", "Div", "HomeTeam", "AwayTeam"]
    for c in ("pin_h", "pin_d", "pin_a", "book_h", "book_d", "book_a"):
        s[c] = _num(s[c])
    for c in ("pin_h", "pin_d", "pin_a"):        # Pinnacle typed once per match
        s[c] = s.groupby(key)[c].transform(lambda x: x.ffill().bfill())
    s = s.dropna(subset=["pin_h", "pin_d", "pin_a"])
    s = s[s[["book_h", "book_d", "book_a"]].notna().any(axis=1)]
    if s.empty:
        print(f"no rows in {sheet} have both Pinnacle's and a book's prices yet.")
        return s
    fair = shin_vec(s[["pin_h", "pin_d", "pin_a"]].to_numpy(float))
    rows = []
    for j, side in enumerate(("H", "D", "A")):
        o = s[f"book_{side.lower()}"].to_numpy(float)
        eff = o * (1.0 - tax) if tax else o                 # 5.3% of the stake goes to tax
        rows.append(s[key + ["Time", "book"]].assign(
            side=side, odds=o, odds_after_tax=eff, p_fair=fair[:, j],
            value=eff * fair[:, j] - 1.0))
    v = pd.concat(rows, ignore_index=True).dropna(subset=["odds"])
    r = VALUE_PRIMARY
    b = v[(v["value"] >= r["min_value"]) & (v["value"] <= r["max_value"])
          & (v["odds"] <= r["max_odds"])]
    b = b.sort_values("value", ascending=False).drop_duplicates(key + ["side"])
    n_m = s.drop_duplicates(key).shape[0]
    if b.empty:
        print(f"{n_m} matches, {len(s)} book rows checked: no bet meets the rule"
              f"{f' after {tax:.1%} tax' if tax else ''}.")
        return b
    b = b.assign(flagged_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    show(f"{len(b)} paper bets from {n_m} matches"
         f"{f'  (prices after {tax:.1%} German tax)' if tax else ''}",
         b[["Date", "Div", "HomeTeam", "AwayTeam", "side", "book", "odds",
            "odds_after_tax", "p_fair", "value"]].assign(
             p_fair=lambda x: x["p_fair"].round(3),
             odds_after_tax=lambda x: x["odds_after_tax"].round(3),
             value=lambda x: (x["value"] * 100).round(1).astype(str) + "%"),
         "NOW: open each book's site, find the bet, and type the price you see in",
         "the log's live_odds column (blank = not offered), with checked_at = time.",
         "Later: pin_close_h/d/a just before kickoff, result (H/D/A) after.")
    out = b.reindex(columns=PAPER_LOG_COLS)
    exists = Path(log).exists()
    if exists:
        old = _paper_read(log)
        k = key + ["side", "book"]
        out = out[~out.set_index(k).index.isin(old.set_index(k).index)]
    out.to_csv(log, mode="a", header=not exists, index=False)
    print(f"\nlogged {len(out)} new paper bets to {log}")
    return b


def paper_report(log: str, tax: float = PAPER_TAX) -> dict:
    b = _paper_read(log)
    for c in ("odds", "live_odds", "p_fair", "pin_close_h", "pin_close_d", "pin_close_a"):
        b[c] = _num(b[c])
    b["result"] = b["result"].fillna("").str.upper().str[:1]
    checked = b["checked_at"].fillna("").ne("") | b["live_odds"].notna()
    b["available"] = np.where(checked, b["live_odds"] >= b["odds"] - 1e-9, np.nan)
    live = b["live_odds"] * (1.0 - tax) if tax else b["live_odds"]
    b["live_value"] = live * b["p_fair"] - 1.0
    cl = b[["pin_close_h", "pin_close_d", "pin_close_a"]].to_numpy(float)
    pc = shin_vec(np.where(np.isfinite(cl), cl, np.nan))
    idx = b["side"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    b["p_close"] = pc[np.arange(len(b)), np.nan_to_num(idx).astype(int)] if len(b) else []
    b["clv"] = live * b["p_close"] - 1.0           # at the price really there, after tax
    won = (b["result"] == b["side"]).astype(float).where(b["result"].isin(["H", "D", "A"]))
    b["pnl"] = won * live - 1.0                    # 1 unit flat, tax paid

    n, n_chk = len(b), int(checked.sum())
    av = b[b["available"] == 1]
    n_av = len(av)
    print(f"{n} paper bets flagged, {n_chk} checked on the site, {n_av} really available.")
    if n_chk:
        print(f"\n  availability        {n_av / n_chk:.0%}  of checked bets"
              f"   (target at least {PAPER_TARGET['min_availability']:.0%})")
        moved = b[checked & (b["available"] == 0)]
        if len(moved):
            print(f"  when not available, the price had fallen by "
                  f"{((moved['odds'] - moved['live_odds']) / moved['odds']).median():.0%} "
                  f"(median); not offered at all: {moved['live_odds'].isna().sum()}")
    c = av.dropna(subset=["clv"])
    lo = hi = np.nan
    if len(c) >= 20:
        rng = np.random.default_rng(42)
        x = c["clv"].to_numpy()
        bs = rng.choice(x, (5000, len(x))).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
    if len(c):
        print(f"  closing-line value  {c['clv'].mean() * 100:+.2f}% on {len(c)} available bets"
              + (f"   95% interval [{lo:+.2f}, {hi:+.2f}]" if len(c) >= 20 else
                 "   (interval once 20+)") + "   (backtest about +2.7% before tax)")
    s = av.dropna(subset=["pnl"])
    if len(s):
        print(f"  profit              {s['pnl'].sum():+.2f} units on {len(s)} settled bets"
              f"   ROI {s['pnl'].mean() * 100:+.1f}%   (noise until thousands of bets)")
    if len(av):
        show("  By book (available bets):", av.groupby("book").agg(
            bets=("odds", "size"), clv_pct=("clv", lambda x: x.mean() * 100),
            units=("pnl", "sum")).round(2).reset_index())
    if n_chk:
        show("  By book (all checked):", b[checked].groupby("book").agg(
            checked=("odds", "size"), available=("available", "mean")).assign(
                available=lambda x: (x["available"] * 100).round(0).astype(str) + "%")
            .reset_index())
    t = PAPER_TARGET
    ok_n = n_av >= t["available_bets"]
    ok_a = n_chk > 0 and n_av / n_chk >= t["min_availability"]
    ok_c = len(c) >= 20 and c["clv"].mean() > t["min_clv"] and lo > 0
    print(f"\n  TARGETS (fixed before the test)")
    print(f"    {'done' if ok_n else '....'}  {t['available_bets']} available bets"
          f"      now {n_av}")
    print(f"    {'pass' if ok_a else 'FAIL' if ok_n else '....'}  availability >= "
          f"{t['min_availability']:.0%}")
    print(f"    {'pass' if ok_c else 'FAIL' if ok_n else '....'}  CLV > "
          f"{t['min_clv']:.0%} with the interval clear of zero")
    if not ok_n:
        print("    Judge only once the first line says done.")
    return {"flagged": n, "checked": n_chk, "available": n_av,
            "clv": c["clv"].mean() if len(c) else np.nan, "clv_lo": lo, "clv_hi": hi}


def cli(argv=None) -> None:
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS + ("-h", "--help"):
        argv.insert(0, "run")              # the old single-command usage

    ap = argparse.ArgumentParser(description="Two-stage selective betting pipeline.")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--mode", default="export", choices=["export", "multiseason"])
        p.add_argument("--source", default=None)
        p.add_argument("--blocks", type=int, default=10)
        p.add_argument("--learner", default="torch", choices=list(LEARNERS),
                       help="torch = the MLP pipeline, lgb = the old tree pipeline")
        p.add_argument("--ensemble-k", type=int, default=1,
                       help="bagged copies per stage, averaged within the stage")

    r = sub.add_parser("run", help="one walk-forward run with the full report")
    common(r)
    r.add_argument("--stage2", default="probability",
                   choices=["probability", "deviation"],
                   help="fit an outcome probability, or the mispricing directly")
    r.add_argument("--no-dixon-coles", action="store_true",
                   help="plain independent Poisson, no low-score correction")
    r.add_argument("--no-learned-w", action="store_true",
                   help="skip the fitted per-match shrinkage weight w(x)")
    r.add_argument("--no-learned-stakes", action="store_true")
    r.add_argument("--export", default=None,
                   help="write per-match predictions to this CSV")

    s = sub.add_parser("sweep", help="re-run under many seeds; report distributions")
    common(s)
    s.add_argument("--seeds", type=int, default=100)
    s.add_argument("--seed-start", type=int, default=0)
    s.add_argument("--learned-w", action="store_true",
                   help="also fit w(x); slower, and not needed for w or w_band")
    s.add_argument("--jobs", type=int, default=8)
    s.add_argument("--out", default=None, help="output folder")

    e = sub.add_parser("elo", help="fit Elo, test it against base rate and market")
    e.add_argument("--source", default="matches_multiseason.csv")
    e.add_argument("-v", "--verbose", action="store_true")

    c = sub.add_parser("clv", help="bet opening prices, judged by closing-line value")
    common(c)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--export", default=None, help="write every test candidate here")

    u = sub.add_parser("ou", help="over/under 2.5 goals, at closing and opening prices")
    common(u)
    u.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("traps", help="measure what test-set tuning and closing-odds leaks do")
    common(t)
    t.add_argument("--seed", type=int, default=42)

    h = sub.add_parser("ah", help="Asian handicap, at closing and opening prices")
    common(h)
    h.add_argument("--seed", type=int, default=42)

    vv = sub.add_parser("value", help="bet books that beat Pinnacle's fair price (no model)")
    vv.add_argument("--source", default="matches_multiseason.csv")
    vv.add_argument("--export", default=None, help="write every candidate bet here")
    vv.add_argument("--min-tier", type=int, default=1,
                    help="only leagues at this tier or lower (2 = below the top flight)")
    vv.add_argument("--exclude-books", default="",
                    help="comma-separated bookmakers to leave out, e.g. IW")

    sc = sub.add_parser("scan", help="list today's value bets from an odds file")
    sc.add_argument("fixtures", help="upcoming matches with odds, football-data layout")
    sc.add_argument("--reference", default="pinnacle", choices=["pinnacle", "exchange"])
    sc.add_argument("--closing", action="store_true",
                    help="the prices were taken shortly before kickoff")
    sc.add_argument("--bank", type=float, default=1000.0)
    sc.add_argument("--log", default="bets_log.csv", help="append bets here ('' = don't)")
    sc.add_argument("--history", default=None,
                    help="stacked results file: adds stage 1's AGREE/disagree to each bet")
    sc.add_argument("--learner", default="hgb", choices=list(LEARNERS))

    rv = sub.add_parser("review", help="judge the logged bets once results are in")
    rv.add_argument("--log", default="bets_log.csv")
    rv.add_argument("--results", required=True,
                    help="file with results AND closing prices for the logged matches")
    rv.add_argument("--reference", default="pinnacle", choices=["pinnacle", "exchange"])

    pn = sub.add_parser("pinnacle", help="two-stage model anchored on Pinnacle, bet at soft books")
    common(pn)
    pn.add_argument("--seed", type=int, default=42)

    dl = sub.add_parser("download", help="fetch football-data main-league seasons (run on your own computer)")
    dl.add_argument("--folder", default="raw_history")
    dl.add_argument("--first", type=int, default=2012, help="first season's starting year")
    dl.add_argument("--last", type=int, default=2021, help="last season's starting year")

    st = sub.add_parser("stack", help="merge a folder of season files into one stacked file")
    st.add_argument("folder")
    st.add_argument("--existing", default=None, help="stacked file to merge with, e.g. matches_multiseason.csv")
    st.add_argument("--out", default="matches_all_seasons.csv")

    ps = sub.add_parser("pinnacle-sweep", help="the agreement result under many seeds and learners")
    ps.add_argument("--source", default="matches_all_seasons.csv")
    ps.add_argument("--seeds", default="1,2,3,4,5", help="comma-separated")
    ps.add_argument("--learners", default="hgb", help="comma-separated, e.g. hgb,torch")
    ps.add_argument("--out", default="pinnacle_sweep.csv")

    s2 = sub.add_parser("stage2-value", help="learned stage 2: which value bets will the market confirm?")
    s2.add_argument("--source", default="matches_all_seasons.csv")
    s2.add_argument("--learner", default="hgb", choices=list(LEARNERS), help="stage 1 learner")
    s2.add_argument("--seed", type=int, default=42)
    s2.add_argument("--pool", default="stage1_pool.csv",
                    help="where to save the stage 1 pool (slow to build)")
    s2.add_argument("--reuse", action="store_true", help="reuse a saved pool")
    s2.add_argument("--min-tier", type=int, default=1,
                    help="stage 2 on leagues at this tier or lower only (2 = below the top flight)")
    s2.add_argument("--exclude-books", default="",
                    help="comma-separated bookmakers to leave out, e.g. IW")

    pl = sub.add_parser("placebo", help="the value rule with the roles swapped: must lose")
    pl.add_argument("--source", default="matches_all_seasons.csv")

    pp = sub.add_parser("paper", help="paper testing: template, flag, report")
    pp.add_argument("step", choices=["template", "flag", "report"])
    pp.add_argument("--fixtures", default="paper_fixtures.csv",
                    help="template: football-data fixtures file to list the matches")
    pp.add_argument("--sheet", default="paper_round.csv",
                    help="template writes it, flag reads it (your typed prices)")
    pp.add_argument("--log", default="paper_log.csv")
    pp.add_argument("--books", default=",".join(PAPER_BOOKS), help="comma-separated")
    pp.add_argument("--divs", default="", help="template: only these, e.g. E1,E2,E3")
    pp.add_argument("--tax", type=float, default=PAPER_TAX,
                    help="German betting tax on the stake; 0 if the book pays it")

    args = ap.parse_args(argv)

    if args.command == "paper":
        if args.step == "template":
            paper_template(args.fixtures, args.sheet,
                           [b for b in args.books.split(",") if b],
                           [d for d in args.divs.split(",") if d] or None)
        elif args.step == "flag":
            paper_flag(args.sheet, args.log, args.tax)
        else:
            paper_report(args.log, args.tax)
        return

    if args.command == "download":
        download(args.folder, args.first, args.last)
        return
    if args.command == "stack":
        stack(args.folder, args.existing, args.out)
        return
    if args.command == "scan":
        configure(learner=args.learner, ensemble_k=1)
        scan(args.fixtures, args.reference, args.closing, args.bank, args.log or None,
             history=args.history)
        return
    if args.command == "stage2-value":
        configure(learner=args.learner, ensemble_k=1)
        stage2_main(args.source, seed=args.seed, pool_file=args.pool, reuse=args.reuse,
                    min_tier=args.min_tier,
                    exclude_books=[b for b in args.exclude_books.split(",") if b])
        return
    if args.command == "placebo":
        placebo_main(args.source)
        return
    if args.command == "pinnacle-sweep":
        pinnacle_sweep(args.source, [int(x) for x in args.seeds.split(",")],
                       [x.strip() for x in args.learners.split(",")], args.out)
        return
    if args.command == "review":
        review(args.log, args.results, args.reference)
        return
    if args.command == "value":
        v = value_main(args.source, min_tier=args.min_tier,
                       exclude_books=[b for b in args.exclude_books.split(",") if b])
        if args.export:
            v.to_csv(args.export, index=False)
            print(f"wrote {args.export}")
        return
    if args.command == "elo":
        elo_report(args.source, verbose=args.verbose)
        return

    configure(learner=args.learner, ensemble_k=args.ensemble_k)
    if args.command == "sweep":
        sweep_main(args)
        return
    if args.command == "pinnacle":
        pinnacle_main(args.source or DEFAULT_SOURCES[args.mode], mode=args.mode, seed=args.seed)
        return
    if args.command == "ah":
        ah_main(args.source or DEFAULT_SOURCES[args.mode], mode=args.mode, seed=args.seed)
        return
    if args.command == "traps":
        traps_main(args.source or DEFAULT_SOURCES[args.mode], mode=args.mode, seed=args.seed)
        return
    if args.command == "ou":
        ou_main(args.source or DEFAULT_SOURCES[args.mode], mode=args.mode, seed=args.seed)
        return
    if args.command == "clv":
        clv_main(args.source or DEFAULT_SOURCES[args.mode], mode=args.mode,
                 seed=args.seed, export=args.export)
        return

    main(source=args.source or DEFAULT_SOURCES[args.mode], mode=args.mode,
         n_blocks=args.blocks, stage2_target=args.stage2,
         use_dixon_coles=not args.no_dixon_coles,
         learn_w=not args.no_learned_w,
         learn_stakes=not args.no_learned_stakes,
         export=args.export)


if __name__ == "__main__":
    cli()
