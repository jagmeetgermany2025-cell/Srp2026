"""
two_stage.py -- selective betting with a two-stage model. Standalone.

WHAT THIS FILE IS

Value betting treated as a SELECTIVE PREDICTION problem. A selective classifier
abstains on most inputs and acts only where it is confident, and is judged on a
risk-coverage curve: as it acts on fewer cases, does its error on those cases
actually fall? That is this project's headline figure with different axis
labels, and the failure mode is the same -- confidence-based selection breaks
down in the tail, because the score is least reliable exactly where it is most
extreme. Betting calls this the winner's curse.

The claim is not that we beat the market. It is that selecting on an uncertain
score is biased, and correcting the score BEFORE selecting repairs the curve.

THE ARCHITECTURE

    Stage 1  MATCH ENGINE.  Predicts physical match quantities -- goals, shots,
             shots on target -- from form alone. Sees no price, ever. Its goal
             predictions become outcome probabilities through a Dixon-Coles
             scoring model, which is what makes it a match engine rather than a
             classifier with extra steps.

    Stage 2  MARKET & EDGE MODEL.  Takes stage 1's physical predictions plus
             odds dynamics. Fits either a market-aware probability or the
             mispricing directly. The better forecaster, and the WRONG thing to
             shrink -- it has seen the market, so its disagreement with the
             market is not independent of it.

    HEADS    MarketReactionModel predicts how the price moves. StakeModel sizes
             bets that have already been selected.

ORDER OF OPERATIONS -- do not vary:

    de-vig -> stage 1 -> scoreline -> deviate -> SHRINK -> SELECT -> stake

The correction must come before selection. A correction applied afterwards
cannot repair a bias the threshold introduced. Staking comes last and after
selection: a model that learned which bets to place would move selection inside
the model, leaving no uncorrected baseline to compare against.

NO NUMBERS APPEAR IN THESE COMMENTS. A figure quoted in a comment is a figure
somebody has to trust without checking, and this project has been damaged once
already by a number that was believed rather than tested. Run it and read the
output.

LAYOUT

    1  constants
    2  loading -- the export parser, and the regression checks on it
    3  features -- de-vigging, the long table, everything known at kickoff
    4  estimators
    5  stage 1 -- the match engine and the scoreline layer
    6  stage 2 -- the market and edge model
    7  auxiliary heads -- market reaction, stake sizing
    8  shrinkage -- the correction, static and learned
    9  walk-forward folds
    10 selection
    11 evaluation -- bootstrap, scoring, accuracy, ROI
    12 orchestration
    13 report

HOW TO RUN

    python3 two_stage.py                      # 2025/26 export, blocks
    python3 two_stage.py --stage2 deviation   # fit mispricing instead
    python3 two_stage.py --no-dixon-coles     # plain independent Poisson
    python3 two_stage.py --mode multiseason
"""
import csv
import warnings
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize, minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:                                          # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_LGB = False
    warnings.warn("lightgbm not installed; using sklearn HistGradientBoosting.")


# =========================================================================
# PART 1 -- CONSTANTS
# =========================================================================

OUTCOMES = ("H", "D", "A")
TARGET_MAP = {"H": 0, "D": 1, "A": 2}

COLUMN_RENAMES = {
    "PinnacleH": "PSH", "PinnacleD": "PSD", "PinnacleA": "PSA",
    "PH": "PSH", "PD": "PSD", "PA": "PSA",
    "BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA",
    "BbMxH": "MaxH", "BbMxD": "MaxD", "BbMxA": "MaxA",
}

# --- price sources -------------------------------------------------------
# The benchmark and the execution price must be DIFFERENT columns. Using one
# book for both is circular: the edge is measured against the de-vigged price
# of the same book that pays the bet, so a real edge and a modelling error look
# identical. Pinnacle is also the least complete, so requiring it discards most
# of the data before modelling starts.
BENCHMARK_COLS = ["AvgCH", "AvgCD", "AvgCA"]   # what we measure against
EXECUTION_COLS = ["MaxH", "MaxD", "MaxA"]      # what we are paid at
REFERENCE_COLS = ["AvgH", "AvgD", "AvgA"]      # reported alongside Max
PINNACLE_COLS = ["PSH", "PSD", "PSA"]          # fallback only

# --- load-time regression checks -----------------------------------------
# Left as None deliberately. Fill these in from a run you have inspected
# yourself, and they become a regression test: a later change to the loader
# that alters the row count or the coverage then fails immediately instead of
# quietly producing different results. A number nobody has checked is worse
# than no number at all, because it looks verified.
EXPECTED_ROWS = None
EXPECTED_DIVISIONS = None
EXPECTED_COVERAGE = {}        # e.g. {"MaxH": <coverage you measured>}
COVERAGE_TOLERANCE = 0.5      # percentage points

# Warn when the fitted correction weight goes above this. A high weight means
# the model is being trusted more than the market, which is worth stopping to
# explain rather than discovering later in a plot.
W_WARNING_LEVEL = 0.5

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

STAGE1_FEATURES = BASE_FEATURE_COLS + SHOT_FEATURE_COLS

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
                    "abs_deviation"]

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

    blocks, header, body = [], None, []
    for r in rows:
        if r[0].strip() == "Div":                 # a header row
            if header is not None and body:
                blocks.append((header, body))
            header, body = [c.strip() for c in r], []
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
    df = load_export(export_csv, season=season).rename(columns=COLUMN_RENAMES)

    checks = {
        "rows": (len(df), EXPECTED_ROWS),
        "divisions": (df["Div"].nunique(), EXPECTED_DIVISIONS),
        "unparsed dates": (int(df["Date"].isna().sum()), 0),
    }
    problems = [f"{k}: got {got}, expected {want}"
                for k, (got, want) in checks.items()
                if want is not None and got != want]

    for col, want in EXPECTED_COVERAGE.items():
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

    return add_shot_features(add_market_features(calculate_rolling_features(df)))


def load_multiseason(path: str = "matches_multiseason.csv",
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
    df = pd.read_csv(path, low_memory=False).rename(columns=COLUMN_RENAMES)
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

    return add_shot_features(add_market_features(calculate_rolling_features(df)))


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


def pick_benchmark_columns(df: pd.DataFrame) -> list:
    """
    Choose which odds columns become the benchmark probabilities.

    Order of preference: closing average, then pre-match average, then Pinnacle.
    Closing average first because it is every book's final opinion and is
    independent of the price we bet at. Pinnacle last because using one book as
    both benchmark and execution price is circular, and it covers less of the
    data. The fallback exists because closing odds are only available in the
    later part of the historical record.
    """
    for cols in (BENCHMARK_COLS, REFERENCE_COLS, PINNACLE_COLS):
        if all(c in df.columns for c in cols) and df[cols[0]].notna().any():
            return cols
    raise KeyError(f"No usable benchmark columns. Tried {BENCHMARK_COLS}, "
                   f"{REFERENCE_COLS}, {PINNACLE_COLS}.")


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
                    min_periods: int = 3) -> pd.DataFrame:
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
    bench = pick_benchmark_columns(df)
    devigged = [shin_de_vig(h, d, a)
                for h, d, a in zip(df[bench[0]], df[bench[1]], df[bench[2]])]
    for k, col in enumerate(MARKET_COLS):
        df[col] = [r[k] for r in devigged]

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
    if HAVE_LGB:
        return lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.01, num_leaves=15, max_depth=3,
            subsample=0.7, colsample_bytree=0.7, random_state=seed, verbosity=-1)
    return HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.01, max_leaf_nodes=15,
        max_depth=3, random_state=seed)


def make_regressor(seed: int = 42):
    """
    The regressor used for physical targets and the uncertainty head.

    Given more capacity than the classifier: predicting a continuous quantity
    from structured inputs is an easier, more structured problem than predicting
    a football result.
    """
    if HAVE_LGB:
        return lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
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

    def _cols(self, df: pd.DataFrame) -> list:
        feats = self.features or STAGE1_FEATURES
        return assert_market_blind([c for c in feats if c in df.columns])

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
        cols = self._cols(train_df)
        for name, target_col in self._usable_targets(train_df).items():
            ok = train_df[target_col].notna()
            model = make_regressor(self.seed)
            model.fit(train_df.loc[ok, cols], train_df.loc[ok, target_col])
            self.models[name] = model

        if self.use_dixon_coles and val_df is not None:
            lam = self.predict_physical(val_df)
            self.rho = fit_dixon_coles_rho(
                lam["s1_goals_home"].to_numpy(), lam["s1_goals_away"].to_numpy(),
                val_df["target"].astype(int).to_numpy())
        return self

    def predict_physical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicted goals, shots and shots on target, one column each."""
        cols = self._cols(df)
        return pd.DataFrame(
            {f"s1_{name}": model.predict(df[cols])
             for name, model in self.models.items()},
            index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Market-blind P(H, D, A). This is the estimate that gets corrected."""
        phys = self.predict_physical(df)
        return outcome_probs(phys["s1_goals_home"].to_numpy(),
                             phys["s1_goals_away"].to_numpy(), self.rho)

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
        cols = self._cols(train_df)
        out = {}
        for name, target_col in self._usable_targets(train_df).items():
            ok = train_df[target_col].notna()
            preds = pd.Series(np.nan, index=train_df.index)
            preds.loc[ok] = cross_val_predict(
                make_regressor(self.seed),
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


def fit_static_shrinkage(p_ref_val: np.ndarray, p_model_val: np.ndarray,
                         y_val: np.ndarray) -> float:
    """
    Find the weight that predicts validation results best.

    WHY VALIDATION DATA AND NEVER TRAINING DATA. On training data the model
    looks better than it is, so the weight would come out too high and the
    correction too weak.

    Refitted once per fold rather than once globally, so the weight is allowed
    to differ over time rather than being assumed constant. Whether it moves is
    itself one of the questions worth answering.
    """
    def objective(w):
        return log_loss(y_val, apply_static_shrinkage(p_ref_val, p_model_val, w),
                        labels=[0, 1, 2])

    return float(minimize_scalar(objective, bounds=(0.0, 1.0),
                                 method="bounded").x)


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
    and the honest move is to keep the simpler version and say so.
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
               n_blocks: int = 10, verbose: bool = True) -> tuple:
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
    needed = features + ["target"] + [c for c in EXECUTION_COLS if c in df.columns]
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
        return p * preds_df[f"Max{outcome}"] - 1.0
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
            for o in allowed if f"Max{o}" in preds_df]
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
        odds = preds_df.get(f"Max{o}")
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
        odds = preds_df.get(f"Max{o}")
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
        odds_ref = sub.get(f"Avg{o}", pd.Series(np.nan, index=sub.index))
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
                          n_boot: int = 1000) -> pd.DataFrame:
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
    if taus is None:
        taus = np.quantile(score_pool(preds_df, corrected_prefix),
                           np.linspace(0.60, 0.99, 12))

    rows = []
    for tau in taus:
        bets_c = select_bets(preds_df, tau, corrected_prefix)
        n_target = len(bets_c)
        if n_target < 20:
            continue

        # Bisect so the uncorrected rule picks the same number of bets. Bet
        # count falls as the threshold rises, so bisection is safe here.
        lo, hi = -1.0, 5.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if len(select_bets(preds_df, mid, raw_prefix)) > n_target:
                lo = mid
            else:
                hi = mid
        tau_prime = (lo + hi) / 2
        bets_r = select_bets(preds_df, tau_prime, raw_prefix)

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
                  prob_prefix: str = "p_corr", tau: float = 0.02,
                  unit_col: str = None) -> pd.DataFrame:
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

    rows = []
    for unit, part in preds_df.groupby(preds_df[unit_col]):
        bets = select_bets(part, tau, prob_prefix)
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
             stake_tau: float = 0.02) -> dict:
    """
    Fit every stage on one walk-forward fold and apply the correction.

    ORDER OF OPERATIONS, which nothing may rearrange:
        stage 1 -> scoreline -> deviate from market -> shrink -> SELECT -> stake
    """
    tr, va, te = fold.train, fold.val, fold.test.copy()

    # --- stage 1, market-blind -------------------------------------------
    engine = MatchEngine(seed=seed, use_dixon_coles=use_dixon_coles)
    s1_oof = engine.fit_predict_oof(tr)
    engine.fit(tr, va)                        # rho is fitted on va, not tr

    phys_te = engine.predict_physical(te)
    p1_val, p1_test = engine.predict_proba(va), engine.predict_proba(te)

    # --- stage 2, market-aware -------------------------------------------
    stage2 = MarketModel(seed=seed, target=stage2_target).fit(tr, s1_oof)
    p2_test = stage2.predict_proba(te, phys_te)

    # --- market reaction, a diagnostic and nothing more -------------------
    reaction = {}
    try:
        rx = MarketReactionModel(seed=seed).fit(tr, s1_oof)
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
    p_ref_val, p_ref_test = probs(va, "p_ref"), probs(te, "p_ref")
    y_val = va["target"].astype(int).to_numpy()

    w = fit_static_shrinkage(p_ref_val, p1_val, y_val)
    te[outcome_cols("p_indep")] = p1_test
    te[outcome_cols("p_model")] = p2_test
    te[outcome_cols("p_corr")] = apply_static_shrinkage(p_ref_test, p1_test, w)
    te[outcome_cols("p_mispricing")] = stage2.predict_mispricing(te, phys_te)

    # --- per-match shrinkage, two ways -----------------------------------
    # Both produce a weight that varies by match. shrink_by_uncertainty() gets
    # it from a hand-written rule applied to predicted error; LearnedShrinkage
    # fits the mapping against validation log loss directly. Carrying both means
    # the hand-written shape can be judged rather than assumed.
    corr = np.nan
    if use_learned_uncertainty:
        try:
            head, check = fit_uncertainty_head(va, p1_val, y_val, seed)
            err = head.predict(uncertainty_inputs(te, p1_test))
            te[outcome_cols("p_unc")] = shrink_by_uncertainty(
                p_ref_test, p1_test, err)
            corr = check["in_sample_corr"]
        except Exception as exc:                            # pragma: no cover
            warnings.warn(f"uncertainty head failed on {fold.test_unit}: {exc}")

    w_stats = {}
    if learn_w:
        try:
            X_val = uncertainty_inputs(va, p1_val)
            X_test = uncertainty_inputs(te, p1_test)
            wx = LearnedShrinkage().fit(p_ref_val, p1_val, X_val, y_val, w)
            weights = wx.weights(X_test)
            te[outcome_cols("p_wfit")] = wx.apply(p_ref_test, p1_test, X_test)
            w_stats = {"wx_mean": float(weights.mean()),
                       "wx_sd": float(weights.std()),
                       "wx_min": float(weights.min()),
                       "wx_max": float(weights.max())}
        except Exception as exc:                            # pragma: no cover
            warnings.warn(f"learned w(x) failed on {fold.test_unit}: {exc}")

    # --- stake sizing, fitted on VALIDATION bets -------------------------
    # The validation fold needs corrected probabilities of its own before it can
    # produce bets to learn from. They are never pooled into preds.
    stake_model = None
    if learn_stakes:
        try:
            va_scored = va.copy()
            va_scored[outcome_cols("p_corr")] = apply_static_shrinkage(
                p_ref_val, p1_val, w)
            stake_model = StakeModel(seed=seed).fit(
                select_bets(va_scored, stake_tau, "p_corr"))
        except Exception as exc:                            # pragma: no cover
            warnings.warn(f"stake model failed on {fold.test_unit}: {exc}")
            stake_model = None

    for col, values in phys_te.items():
        te[col] = values

    y_test = te["target"].astype(int)
    diagnostics = {
        "test_unit": fold.test_unit, "val_unit": fold.val_unit,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "w": w, "rho": engine.rho, "uncertainty_corr": corr,
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
        df, features, min_test_matches=min_test_matches, n_blocks=n_blocks)

    preds, diagnostics, stake_models = [], [], {}
    for fold in fold_list:
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


def main(source: str = "all-euro-data-2025-2026.csv", mode: str = "export",
         n_blocks: int = 10, stage2_target: str = "probability",
         use_dixon_coles: bool = True, learn_w: bool = True,
         learn_stakes: bool = True):
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

    roi_arms = ["p_indep", "p_corr"] + (["p_wfit"] if has_arm(preds, "p_wfit") else [])
    show("ROI at matched coverage:",
         betting_report(preds, prefixes=tuple(roi_arms)),
         "shop_gain is mechanical. Subtract it before calling any roi an edge.")

    op_tau = float(np.quantile(score_pool(preds, "p_corr"), 0.90))
    report_roi(select_bets(preds, op_tau, "p_corr"),
               f"Operating point: corrected, top 10% by EV (tau = {op_tau:+.4f})")

    unit_col = "Block" if "Block" in preds.columns else "Season"
    if stake_models:
        show("Staking rules on identical bets:",
             staking_comparison(preds, stake_models, unit_col),
             "total_staked shows whether a rule won by allocating or leveraging.")

    curve_arms = [("p_indep", "uncorrected"), ("p_corr", "corrected, static w")]
    if has_arm(preds, "p_wfit"):
        curve_arms.append(("p_wfit", "corrected, fitted w(x)"))
    for prefix, label in curve_arms:
        curve = risk_coverage_curve(preds, prefix)
        show(f"Risk-coverage, {label}:  AURC = {aurc(curve):+.3f}", curve)

    nv = show("Non-vacuousness check:",
              check_non_vacuousness(preds, "p_corr", "p_indep"))
    if len(nv) and nv["overlap"].mean() > 0.95:
        print("\n  WARNING: the corrected and uncorrected rules are selecting "
              "almost the same bets.\n  The correction may be a relabelled "
              "threshold rather than a correction.")

    show(f"By {unit_col.lower()}:", era_breakdown(preds, folds))

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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Two-stage selective betting pipeline.")
    ap.add_argument("--mode", default="export", choices=["export", "multiseason"])
    ap.add_argument("--source", default=None)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--stage2", default="probability",
                    choices=["probability", "deviation"],
                    help="fit an outcome probability, or the mispricing directly")
    ap.add_argument("--no-dixon-coles", action="store_true",
                    help="plain independent Poisson, no low-score correction")
    ap.add_argument("--no-learned-w", action="store_true",
                    help="skip the fitted per-match shrinkage weight w(x)")
    ap.add_argument("--no-learned-stakes", action="store_true")
    args = ap.parse_args()

    defaults = {"export": "all-euro-data-2025-2026.csv",
                "multiseason": "matches_multiseason.csv"}
    main(source=args.source or defaults[args.mode], mode=args.mode,
         n_blocks=args.blocks, stage2_target=args.stage2,
         use_dixon_coles=not args.no_dixon_coles,
         learn_w=not args.no_learned_w,
         learn_stakes=not args.no_learned_stakes)
