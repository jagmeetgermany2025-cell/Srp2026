"""
selective.py -- selective prediction, applied to football value betting.

WHAT THIS FILE IS

A restructure of newFrame.py around one idea: value betting is a SELECTIVE
PREDICTION problem, not a forecasting problem.

A selective classifier does not act on every input. It abstains on most and
acts only where it is confident. It is judged on a risk-coverage curve: as it
acts on fewer cases, does its error on those cases actually fall?

That is this project's headline figure with different axis labels. Coverage is
how many bets clear the threshold; risk is what they return. And the failure
mode is the same one: confidence-based selection breaks down in the tail,
because the score is least reliable exactly where it is most extreme. Betting
calls this the winner's curse.

So the claim is not that we beat the market. It is that selecting on an
uncertain score is biased, and correcting the score before selecting repairs
the curve.

HOW TO READ THE DOCSTRINGS

Every function says four things:
    What it does.
    Why it does it.
    How it differs from newFrame.py, and why.
    Whether it matches what n.pdf asked for.

No numbers appear in these comments. Any figure quoted in a comment is a
figure somebody has to trust without checking, and this project has already
been damaged once by a number that was believed rather than tested. Run the
code and read the output.

ON n.pdf

n.pdf proposes five extensions. This script uses extra market and on-pitch
features, a two-stage model, and fixed staking rules. The separate run_tfm.py
experiment compares TabPFN, TabNet and FT-Transformer with LightGBM while
preserving calibration, shrinkage and selection. Changing the probability
estimator does not require removing the separable selection step. See
README_TFM.md for the chronological comparison protocol.

ORDER OF OPERATIONS -- do not vary:

    de-vig -> model -> calibrate -> deviation -> correct -> SELECT

The correction must come before selection. A correction applied afterwards
cannot repair a bias the threshold introduced.
"""
import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict

# LightGBM is the intended model. If it is missing we fall back to
# scikit-learn's histogram gradient booster, which is the same algorithm, so
# the file stays runnable. The fallback is a convenience, not a modelling
# decision -- install lightgbm before quoting any result.
try:
    import lightgbm as lgb
    HAVE_LGB = True
except (ImportError, OSError):                               # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_LGB = False
    warnings.warn("lightgbm unavailable; legacy factories use sklearn HistGradientBoosting. "
                  "The TFM runner requires the explicitly requested model and never substitutes it.")


# =========================================================================
# CONSTANTS
# =========================================================================
COLUMN_RENAMES = {
    "PinnacleH": "PSH", "PinnacleD": "PSD", "PinnacleA": "PSA",
    "PH": "PSH", "PD": "PSD", "PA": "PSA",
    "BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA",
    "BbMxH": "MaxH", "BbMxD": "MaxD", "BbMxA": "MaxA",
}

# --- Price sources -------------------------------------------------------
# The benchmark and the execution price must be DIFFERENT columns.
#
# newFrame.py used Pinnacle for both. That is circular: we measured our edge
# against the de-vigged price of the same book that paid the bet, so a real
# edge and a modelling error looked identical. Pinnacle also covers only part
# of the data, so requiring it discarded most matches before modelling began.
BENCHMARK_COLS = ["AvgCH", "AvgCD", "AvgCA"]   # what we measure against
EXECUTION_COLS = ["MaxH", "MaxD", "MaxA"]      # what we are paid at
REFERENCE_COLS = ["AvgH", "AvgD", "AvgA"]      # reported alongside Max
PINNACLE_COLS = ["PSH", "PSD", "PSA"]          # fallback only

# --- Load-time regression checks -----------------------------------------
# Left as None deliberately. Fill these in from a run you have inspected
# yourself, then they become a regression test: if a later change to the
# loader alters the row count or the coverage, load_and_prepare() fails
# immediately instead of quietly producing different results.
#
# They are None rather than pre-filled because a number written here that
# nobody has checked is worse than no number at all -- it looks verified.
EXPECTED_ROWS = None
EXPECTED_DIVISIONS = None
EXPECTED_COVERAGE = {}        # e.g. {"MaxH": <coverage you measured>}
COVERAGE_TOLERANCE = 0.5      # percentage points

# Stop the download after this many failures in a row. A run of failures means
# the connection is broken, not that files are missing, and grinding through
# hundreds of doomed requests before saying so wastes minutes.
MAX_CONSECUTIVE_ERRORS = 5

# Warn when the fitted correction weight goes above this. A high weight means
# the model is being trusted far more than the market, which is worth stopping
# to explain rather than discovering later in a plot.
W_WARNING_LEVEL = 0.5

OUTCOMES = ("H", "D", "A")
TARGET_MAP = {"H": 0, "D": 1, "A": 2}

# --- Features ------------------------------------------------------------
# Stage 1 of the two-stage model trains on BASE_FEATURE_COLS only. It must
# never see the market, so that its disagreement with the market is real
# rather than an echo of it.
BASE_FEATURE_COLS = [
    "home_rest", "away_rest", "diff_rest_days",
    "home_GF_5", "home_GA_5", "home_GD_5",
    "away_GF_5", "away_GA_5", "away_GD_5",
    "home_GF_10", "home_GA_10", "away_GF_10", "away_GA_10",
    "diff_roll_GF_5", "diff_roll_GA_5",
]

MARKET_COLS = ["p_ref_H", "p_ref_D", "p_ref_A"]

# Built by add_market_features() and add_shot_features(). Kept separate from
# the base list so a run can include or exclude them and the difference can be
# attributed to them rather than guessed at.
MARKET_FEATURE_COLS = ["drift_H", "drift_D", "drift_A", "abs_drift"]
SHOT_FEATURE_COLS = [
    "home_SF_5", "home_SA_5", "home_STF_5", "home_STA_5",
    "away_SF_5", "away_SA_5", "away_STF_5", "away_STA_5",
    "home_shot_quality_5", "away_shot_quality_5",
]

# Single-stage feature set, kept so the two-stage model has something to beat.
FEATURE_COLS = MARKET_COLS + BASE_FEATURE_COLS

# --- Uncertainty ---------------------------------------------------------
# Inputs to the per-match shrinkage weight. These must answer "how much should
# we trust our own estimate here", not "what are these teams like".
#
# newFrame.py used rest days, which answer the second question. Bookmaker
# disagreement answers the first: where the books differ, the true price is
# less settled and our deviation is more likely to be noise.
UNCERTAINTY_COLS = ["spread_mean", "overround_close", "p_ref_entropy",
                    "abs_deviation"]


# =========================================================================
# SMALL HELPERS
# =========================================================================
def outcome_cols(prefix: str) -> list:
    """
    Turn a prefix into its three outcome column names.

    Why: the H/D/A triple is written out constantly; naming it once stops the
    three from drifting apart.
    """
    return [f"{prefix}_{o}" for o in OUTCOMES]


def make_classifier(seed: int = 42):
    """
    Build the base classifier.

    Why small and heavily regularised: the training sets are short and the
    signal is weak, so a bigger model would fit noise.

    Same settings as newFrame.py. Only the fallback branch is new, so the file
    still runs where lightgbm is absent.
    """
    if HAVE_LGB:
        return lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.01, num_leaves=15,
            max_depth=3, subsample=0.7, colsample_bytree=0.7,
            random_state=seed, verbosity=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.01, max_leaf_nodes=15,
        max_depth=3, random_state=seed,
    )


def make_regressor(seed: int = 42):
    """
    Build the regressor used by the uncertainty head.

    Why: the head predicts a continuous quantity (how badly we expect to do on
    a match), so it needs a regressor rather than a classifier.

    New. newFrame.py had no learned uncertainty.
    """
    if HAVE_LGB:
        return lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        max_depth=4, random_state=seed,
    )


def match_key(df: pd.DataFrame) -> pd.Series:
    """
    Build a stable unique id for each match.

    Why: bootstrap_roi() resamples matches rather than bets, so it needs to
    know which bets belong to the same match. A key that collided across
    divisions would merge unrelated matches and make the intervals too narrow.

    New. newFrame.py never grouped bets by match, because it never bootstrapped.
    """
    return (pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            + "|" + df["HomeTeam"].astype(str)
            + "|" + df["AwayTeam"].astype(str))


# =========================================================================
# PART 1 -- DE-VIGGING, SHRINKAGE MATHS, FEATURES
#
# Mostly carried over from newFrame.py. This was the sound part of that file.
# =========================================================================


def shin_de_vig(odd_h: float, odd_d: float, odd_a: float) -> tuple:
    """
    Turn three bookmaker odds into three real probabilities.

    Why: odds do not imply probabilities directly. One-over-odds for the three
    outcomes sums to more than one, because the book takes a margin. That
    excess has to be removed before the numbers mean anything.

    Why Shin's method rather than simple normalising: the margin is not spread
    evenly across outcomes. Longshots carry more of it than favourites, so
    dividing everything by the same constant would leave that bias in place.
    Shin removes it by assuming the margin exists because some bettors are
    better informed than the book.

    Carried over from newFrame.py unchanged.

    Returns three probabilities summing to one, or three NaNs if the odds are
    missing or invalid.
    """
    if pd.isna(odd_h) or pd.isna(odd_d) or pd.isna(odd_a):
        return np.nan, np.nan, np.nan
    if odd_h <= 1.0 or odd_d <= 1.0 or odd_a <= 1.0:
        return np.nan, np.nan, np.nan

    p_raw = np.array([1.0 / odd_h, 1.0 / odd_d, 1.0 / odd_a])
    beta = p_raw.sum()
    if beta <= 1.0:
        # No margin at all, which usually means a data error. Just normalise.
        return p_raw[0] / beta, p_raw[1] / beta, p_raw[2] / beta

    # Solve for Shin's z, the share of informed money, by bisection.
    z_low, z_high = 0.0, 0.4
    for _ in range(30):
        z = (z_low + z_high) / 2.0
        val = np.sum(np.sqrt(z ** 2 + 4 * (1 - z) * (p_raw ** 2) / beta))
        if val > (2.0 - z):
            z_low = z
        else:
            z_high = z

    p = (np.sqrt(z ** 2 + 4 * (1 - z) * (p_raw ** 2) / beta) - z) / (2 * (1 - z))
    p = p / p.sum()
    return float(p[0]), float(p[1]), float(p[2])


def apply_static_shrinkage(p_ref: np.ndarray, p_model: np.ndarray,
                           w: float) -> np.ndarray:
    """
    Pull the model's probabilities toward the market's by a single weight.

    What w means: zero ignores the model and uses the market, one ignores the
    market and uses the model, in between blends them.

    Why the blend is geometric (a weighted average of the logs rather than of
    the probabilities): it keeps the result positive, behaves sensibly near
    zero and one, and corresponds to averaging odds rather than probabilities.

    Carried over from newFrame.py unchanged. This is the project's core
    contribution and it was not the broken part.

    Sanity check worth applying: a fitted w above one half suggests something
    structural is wrong. Investigate rather than carry on.
    """
    w = np.clip(w, 0.0, 1.0)
    eps = 1e-12
    p_ref_c = np.clip(p_ref, eps, 1.0 - eps)
    p_model_c = np.clip(p_model, eps, 1.0 - eps)

    log_p = (1.0 - w) * np.log(p_ref_c) + w * np.log(p_model_c)
    # Subtract the row maximum before exponentiating so nothing overflows.
    p_unnorm = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
    p_shrunk = p_unnorm / np.sum(p_unnorm, axis=1, keepdims=True)
    return (p_shrunk / p_shrunk.sum(axis=1, keepdims=True)).astype(np.float64)


def apply_heteroscedastic_shrinkage(p_ref: np.ndarray, p_model: np.ndarray,
                                    X_proxy: np.ndarray,
                                    theta: np.ndarray) -> np.ndarray:
    """
    The same blend, but with a different weight for every match.

    Why: some matches deserve more trust than others, so a single global
    weight throws information away.

    Carried over from newFrame.py unchanged. The idea is right; its inputs
    were not, because newFrame.py fed it rest days, which say nothing about
    how confident we should be. fit_uncertainty_head() learns the weight
    instead, and this function remains as the thing to compare against.
    """
    eps = 1e-12
    p_ref_c = np.clip(p_ref, eps, 1.0 - eps)
    p_model_c = np.clip(p_model, eps, 1.0 - eps)

    wx = 1.0 / (1.0 + np.exp(-np.clip(X_proxy @ theta, -30, 30)))
    wx = wx[:, np.newaxis]

    log_p = (1.0 - wx) * np.log(p_ref_c) + wx * np.log(p_model_c)
    p_unnorm = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
    p_shrunk = p_unnorm / np.sum(p_unnorm, axis=1, keepdims=True)
    return (p_shrunk / p_shrunk.sum(axis=1, keepdims=True)).astype(np.float64)


def fit_static_shrinkage(p_ref_val: np.ndarray, p_model_val: np.ndarray,
                         y_val: np.ndarray) -> float:
    """
    Find the weight that predicts validation results best.

    Why validation data and never training data: on training data the model
    looks better than it is, so the weight would come out too high and the
    correction would be too weak.

    Carried over from newFrame.py. The only change is that walk-forward now
    calls it once per fold instead of once globally, so the weight is allowed
    to differ between eras rather than being assumed constant.
    """
    def objective(w):
        return log_loss(y_val, apply_static_shrinkage(p_ref_val, p_model_val, w),
                        labels=[0, 1, 2])

    res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)


def fit_heteroscedastic_shrinkage(p_ref_val: np.ndarray, p_model_val: np.ndarray,
                                  X_proxy_val: np.ndarray,
                                  y_val: np.ndarray) -> np.ndarray:
    """
    Fit the per-match weight's parameters on validation data.

    Why: same reasoning as fit_static_shrinkage, applied to the version that
    varies by match.

    Carried over from newFrame.py unchanged.
    """
    init_theta = np.zeros(X_proxy_val.shape[1])

    def objective(theta):
        p = apply_heteroscedastic_shrinkage(p_ref_val, p_model_val,
                                            X_proxy_val, theta)
        return log_loss(y_val, p, labels=[0, 1, 2])

    return minimize(objective, init_theta, method="L-BFGS-B").x


def pick_benchmark_columns(df: pd.DataFrame) -> list:
    """
    Choose which odds columns become the benchmark probabilities.

    Order of preference: closing average, then pre-match average, then
    Pinnacle.

    Why closing average first: it is every book's final opinion, and it is
    independent of the price we bet at. Why Pinnacle last: using one book as
    both benchmark and execution price is circular, and it covers less of the
    data than the averages do.

    Why the fallback exists at all: closing odds are only available in the
    later part of the historical record, so early seasons must fall back to
    the pre-match average.

    New. newFrame.py hard-coded Pinnacle and had no fallback.
    """
    for cols in (BENCHMARK_COLS, REFERENCE_COLS, PINNACLE_COLS):
        if all(c in df.columns for c in cols) and df[cols[0]].notna().any():
            return cols
    raise KeyError(f"No usable benchmark columns. Tried {BENCHMARK_COLS}, "
                   f"{REFERENCE_COLS}, {PINNACLE_COLS}.")


def calculate_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build every feature that is known before kickoff.

    Produces the market's de-vigged probabilities, rest days for both teams,
    and rolling goals scored and conceded.

    HOW THIS DIFFERS FROM newFrame.py, and why it matters.

    newFrame.py computed rest days by grouping on HomeTeam, which gives days
    since that team's last HOME match rather than its last match of any kind.
    A team that played away midweek was recorded as fully rested at the
    weekend. The feature was meant to capture fatigue and was systematically
    reporting tired teams as fresh.

    It is fixed by computing rest on the long table below, where each team
    appears once per match whatever the venue. That is the same structure the
    goal features already used, so the fix reuses existing machinery rather
    than adding any.

    LEAKAGE RULE: every rolling number is shifted by one match before use, so
    a match never sees its own result. Without the shift these are not
    features, they are the answer.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # --- market probabilities from the benchmark price --------------------
    bench = pick_benchmark_columns(df)
    devigged = [shin_de_vig(h, d, a)
                for h, d, a in zip(df[bench[0]], df[bench[1]], df[bench[2]])]
    df["p_ref_H"] = [r[0] for r in devigged]
    df["p_ref_D"] = [r[1] for r in devigged]
    df["p_ref_A"] = [r[2] for r in devigged]

    # --- long table: one row per team per match ---------------------------
    home_df = df[["Date", "HomeTeam", "FTHG", "FTAG"]].rename(
        columns={"HomeTeam": "Team", "FTHG": "GF", "FTAG": "GA"})
    home_df["is_home"] = True
    home_df["match_id"] = home_df.index

    away_df = df[["Date", "AwayTeam", "FTAG", "FTHG"]].rename(
        columns={"AwayTeam": "Team", "FTAG": "GF", "FTHG": "GA"})
    away_df["is_home"] = False
    away_df["match_id"] = away_df.index

    matches = (pd.concat([home_df, away_df])
                 .sort_values(["Date", "match_id"]).reset_index(drop=True))

    # --- rest days, computed on the long table (see the docstring) --------
    # The cap and the default for a team's first match match newFrame.py's
    # scale, so this reads as a correction rather than a rescaling.
    matches["rest"] = matches.groupby("Team")["Date"].diff().dt.days
    matches["rest"] = matches["rest"].fillna(14).clip(upper=14)

    # --- rolling goals ----------------------------------------------------
    for span in (5, 10):
        for col in ("GF", "GA"):
            matches[f"{col}_{span}"] = matches.groupby("Team")[col].transform(
                lambda x: x.shift(1).ewm(span=span, min_periods=3).mean())
        matches[f"GD_{span}"] = matches[f"GF_{span}"] - matches[f"GA_{span}"]

    home_stats = matches[matches["is_home"]].set_index("match_id")
    away_stats = matches[~matches["is_home"]].set_index("match_id")

    df["home_rest"] = home_stats["rest"]
    df["away_rest"] = away_stats["rest"]
    df["diff_rest_days"] = df["home_rest"] - df["away_rest"]

    for span in (5, 10):
        for col in ("GF", "GA", "GD"):
            df[f"home_{col}_{span}"] = home_stats[f"{col}_{span}"]
            df[f"away_{col}_{span}"] = away_stats[f"{col}_{span}"]

    df["diff_roll_GF_5"] = df["home_GF_5"] - df["away_GF_5"]
    df["diff_roll_GA_5"] = df["home_GA_5"] - df["away_GA_5"]

    if "FTR" in df.columns:
        df["target"] = df["FTR"].map(TARGET_MAP)
    df["match_key"] = match_key(df)
    return df


# =========================================================================
# PART 2 -- DATA
# =========================================================================


def load_and_prepare(export_csv: str = "all-euro-data-2025-2026.csv",
                     season: str = "2526", strict: bool = True) -> pd.DataFrame:
    """
    Read the supplied export and build every feature.

    Why it exists: the export is broken in four ways -- semicolons instead of
    commas, commas instead of decimal points, header rows buried inside the
    file, and American month/day dates. load_export.py already handles all
    four, but newFrame.py never called it and read the file with pandas
    defaults instead.

    Why the checks are the important part: all four faults fail SILENTLY.
    Nothing raises, rows are dropped, and odds columns turn blank. The only
    visible symptom is a coverage figure far below what it should be, which
    nobody spots unless they look. So we assert on load and stop, rather than
    hope somebody notices later.

    Set the expected values below from a run you have inspected. They then act
    as a regression test: if a later change to the loader alters them, this
    fails immediately instead of quietly producing different results.

    New. newFrame.py started from a bare pd.read_csv and had no checks at all.
    """
    from load_export import load_export

    df = load_export(export_csv, season=season)
    df = df.rename(columns=COLUMN_RENAMES)

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

    df = calculate_rolling_features(df)
    df = add_market_features(df)
    df = add_shot_features(df)
    return df


def fetch_seasons(start_year: int = 2005, end_year: int = 2025,
                  cache_dir: str = "data/raw", divisions: list = None,
                  pause: float = 0.4) -> pd.DataFrame:
    """
    Download many seasons from football-data.co.uk and cache them.

    WHY THIS GATES EVERYTHING ELSE.

    The limit on this project is sample size, not model quality. A confidence
    interval narrows with the number of bets tested, and one season does not
    supply enough of them to tell a real edge from zero.

    That distinction matters more than it sounds. A study that finds nothing
    is only informative if it could have found something had there been
    something to find. Without more test seasons, a flat result means "we
    could not tell", which is not a finding. With them, it means "it is not
    there", which is.

    No amount of modelling substitutes for this. A better model does not buy
    one extra bet; more test seasons do.

    Why it starts where it does: the download begins at the first season for
    which best-price and average-price columns exist, because the best price
    is what we bet at.

    IMPORTANT: these files are clean -- ordinary commas, ordinary date order.
    Do not put them through load_export(), which is written for the broken
    export and would misread their dates without complaining.

    New. newFrame.py referred to a download script that was never written.
    """
    import time
    from urllib.error import HTTPError, URLError

    if divisions is None:
        from load_export import TIERS
        divisions = list(TIERS)

    os.makedirs(cache_dir, exist_ok=True)
    seasons = [f"{y % 100:02d}{(y + 1) % 100:02d}"
               for y in range(start_year, end_year + 1)]

    frames, missing, errors = [], [], []
    consecutive_errors = 0
    total = len(seasons) * len(divisions)
    done = downloaded = 0
    print(f"Fetching {len(seasons)} seasons x {len(divisions)} divisions "
          f"({total} files). Cached files are skipped.")

    for season in seasons:
        season_rows = season_new = 0
        for div in divisions:
            done += 1
            cache = os.path.join(cache_dir, f"{season}_{div}.parquet")
            if os.path.exists(cache):
                part = pd.read_parquet(cache)
                frames.append(part)
                season_rows += len(part)
                continue

            url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
            try:
                part = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
                consecutive_errors = 0
            except (HTTPError, pd.errors.EmptyDataError) as exc:
                # A genuinely absent file is normal: not every league ran every
                # season, so a 404 here is expected rather than a failure.
                if isinstance(exc, HTTPError) and exc.code != 404:
                    consecutive_errors += 1
                    errors.append(f"{season}/{div}: HTTP {exc.code}")
                else:
                    missing.append(f"{season}/{div}")
                    continue
            except URLError as exc:
                # This is a broken connection, not an absent file, and the two
                # must not be confused. Swallowing it would mean waiting out
                # every remaining request before reporting a problem that was
                # obvious on the first one.
                consecutive_errors += 1
                errors.append(f"{season}/{div}: {exc.reason}")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(
                    f"Aborted after {consecutive_errors} consecutive network "
                    f"failures. First was: {errors[0]}\n"
                    "Nothing is wrong with the code -- check your connection, "
                    "or open the URL in a browser to confirm the site is up:\n"
                    f"  {url}")
            if errors and errors[-1].startswith(f"{season}/{div}"):
                continue

            part = part.rename(columns=COLUMN_RENAMES)
            part = part[part.get("HomeTeam").notna()] if "HomeTeam" in part else part
            part["Season"] = season
            part["Div"] = div
            part.to_parquet(cache, index=False)
            frames.append(part)
            season_rows += len(part)
            season_new += 1
            downloaded += 1
            time.sleep(pause)          # be polite; this is many requests

        # One line per season so a long download visibly progresses rather
        # than looking frozen.
        print(f"  [{done:>4}/{total}] {season}  "
              f"{season_rows:>6,} matches  ({season_new} downloaded)", flush=True)

    if not frames:
        detail = ("\n  " + "\n  ".join(errors[:5])) if errors else ""
        raise RuntimeError(
            "Downloaded nothing and found nothing cached." + detail +
            "\nIf the errors above mention SSL or connection, it is a network "
            "problem rather than a code one.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    # These files use day-first dates, unlike the supplied export.
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    print(f"Fetched {len(df):,} matches across {df['Season'].nunique()} seasons "
          f"({len(missing)} files absent, {len(errors)} failed)")
    print(coverage_table(df))
    return df.sort_values("Date").reset_index(drop=True)


def load_multiseason(path: str = "matches_multiseason.csv",
                     drop_partial_seasons: bool = True,
                     min_season_matches: int = 1000) -> pd.DataFrame:
    """
    Load the multi-season file and build every feature.

    This is the main entry point now. It supersedes load_and_prepare(), which
    read the single-season export.

    Why this file is easier than the export: it is already clean. Ordinary
    commas, ISO dates, one Season column, several seasons stacked. None of the
    four faults load_export.py exists to repair apply here, so it must NOT be
    routed through that loader -- doing so would misread the dates.

    What it still needs doing to it:

      Junk columns. The file carries a few unnamed empty columns, an artefact
      of stacking sources with different widths. They are dropped so they
      cannot be mistaken for data.

      Partial seasons. The newest season in the file is still being played and
      holds far too few matches to test on. Left in, it would become a
      walk-forward fold whose result is noise, and that noise would sit in the
      pooled figures looking like evidence. It is dropped by default, and the
      drop is announced rather than silent.

    What this file gives us that the single export did not: several seasons,
    which is what walk-forward needs and therefore the only route to an
    interval narrow enough to conclude anything.

    What it still does not give us: enough seasons to study how the market
    changed over decades. Treat era comparisons here as indicative only.
    """
    df = pd.read_csv(path, low_memory=False)

    junk = [c for c in df.columns if str(c).startswith("Unnamed")]
    if junk:
        df = df.drop(columns=junk)

    df = df.rename(columns=COLUMN_RENAMES)

    # ISO dates here, so no day-first hint -- adding one would be wrong.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Season"].astype(str)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])

    if drop_partial_seasons:
        counts = df["Season"].value_counts()
        partial = sorted(counts[counts < min_season_matches].index)
        if partial:
            print(f"Dropping partial season(s) {partial}: fewer than "
                  f"{min_season_matches:,} matches, too few to test on.")
            df = df[~df["Season"].isin(partial)]

    print(f"Loaded {len(df):,} matches, {df['Div'].nunique()} divisions, "
          f"{df['Season'].nunique()} seasons "
          f"({df['Date'].min().date()} to {df['Date'].max().date()})")
    print(coverage_table(df))

    df = calculate_rolling_features(df)
    df = add_market_features(df)
    df = add_shot_features(df)
    return df


def load_local_seasons(root: str = "data/raw") -> pd.DataFrame:
    """
    Build the same table as fetch_seasons(), but from files already on disk.

    Why it exists: the download makes hundreds of requests, which is slow and
    fails entirely if the connection is unreliable. football-data.co.uk also
    publishes one zip per season, so downloading twenty-odd files by hand and
    unzipping them here is often quicker and always more reliable.

    It also covers the case where the data was copied from another machine.

    Recognised layouts, tried in order:
        root/<season>/<div>.csv     e.g. data/raw/0506/E0.csv   (zip default)
        root/<season>_<div>.csv     e.g. data/raw/0506_E0.csv   (our cache)
        root/<season>_<div>.parquet                             (our cache)
        root/<anything>.csv         season and division read from the file

    The season code is taken from the folder or filename where possible. If it
    cannot be read there, and the file has no Season column, the file is
    skipped with a warning rather than guessed at -- a wrong season label would
    put matches in the wrong walk-forward fold, which is worse than a gap.

    These files are the clean ones from the site: ordinary commas and day-first
    dates. Do not route them through load_export(), which is written for the
    broken single-file export and would misread their dates.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"No such directory: {root}\n"
            "Either run fetch_seasons() to download, or unzip the per-season "
            "files from football-data.co.uk into this folder.")

    paths = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
                   + glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"{root} exists but contains no csv or parquet files.")

    frames, skipped = [], []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        parent = os.path.basename(os.path.dirname(path))

        # Season/division from the path, in either supported shape.
        season = div = None
        if re.fullmatch(r"\d{4}", parent):            # root/<season>/<div>.csv
            season, div = parent, stem
        elif re.fullmatch(r"(\d{4})_(.+)", stem):      # root/<season>_<div>.csv
            season, div = re.fullmatch(r"(\d{4})_(.+)", stem).groups()

        try:
            part = (pd.read_parquet(path) if path.endswith(".parquet")
                    else pd.read_csv(path, encoding="latin-1", on_bad_lines="skip"))
        except Exception as exc:
            skipped.append(f"{path}: {exc}")
            continue

        part = part.rename(columns=COLUMN_RENAMES)
        if season is not None:
            part["Season"] = season
        if div is not None and "Div" not in part.columns:
            part["Div"] = div

        if "Season" not in part.columns or "Div" not in part.columns:
            skipped.append(f"{path}: no season/division in the path or the file")
            continue
        frames.append(part)

    if not frames:
        raise FileNotFoundError(
            f"Found files in {root} but could not read a season from any.\n  "
            + "\n  ".join(skipped[:5]))

    df = pd.concat(frames, ignore_index=True, sort=False)
    if "Date" in df.columns:
        # Site files are day-first, unlike the supplied export.
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    print(f"Loaded {len(df):,} matches from {root} across "
          f"{df['Season'].nunique()} seasons ({len(skipped)} files skipped)")
    if skipped:
        for line in skipped[:5]:
            print("  skipped:", line)
    print(coverage_table(df))
    return df.sort_values("Date").reset_index(drop=True)


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show how complete the key columns are, season by season.

    Why: it decides the modelling window. Closing odds and best/average prices
    each start in different seasons, so anything depending on them can only be
    used from the season it becomes available. Reading that off a table beats
    assuming it.

    It also exposes provider changes, where a column keeps its name but
    changes meaning. Pooling across such a break silently would mix two
    different quantities under one heading, so check here before pooling.

    New. newFrame.py assumed the seasons it needed were present.
    """
    cols = [c for c in ["B365H", "B365CH", "PSH", "MaxH", "AvgH", "AvgCH", "HST"]
            if c in df.columns]
    out = df.groupby("Season").agg(rows=("Date", "size"))
    for c in cols:
        out[c] = df.groupby("Season")[c].apply(lambda s: s.notna().mean() * 100).round(1)
    return out


# =========================================================================
# PART 3 -- FEATURES FROM n.pdf
#
# The two n.pdf proposals that survive contact with our data and with the
# research claim.
# =========================================================================


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build market movement and bookmaker disagreement features.

    From n.pdf, which calls these "Dynamic Market Signals". This is the part
    of n.pdf's feature proposal that our data can actually support.

    Two quantities are computed here and they do OPPOSITE jobs. Keeping them
    apart is the whole point of the function.

    DRIFT -- how much the price moved between opening and closing. This is a
    candidate predictive feature and goes in MARKET_FEATURE_COLS.

        Why we expect little from it: our benchmark is the closing price, and
        the closing price already reflects whatever caused the move. Drift
        will look strongly related to the error in the OPENING price, but that
        is circular -- the price moved because the opening price was wrong.
        Test it against the CLOSING price, which is the comparison that
        matters, and be ready to report a negative result.

        Why include it anyway: "we tested market movement and it was already
        priced" is worth being able to say, and it is a check an examiner
        expects to have been attempted.

    DISPERSION -- how much the bookmakers disagree with each other. This is
    NOT predictive and must not go in MARKET_FEATURE_COLS. It is an
    uncertainty measure and goes in UNCERTAINTY_COLS.

        Why this is the real gain from n.pdf: the correction step needs a
        per-match answer to "how much should we trust our own deviation here".
        Where the books disagree, the true price is less settled, our
        deviation is more likely to be noise, and it should be shrunk harder.
        That is the winner's curse expressed as something measurable, so it
        feeds the central claim rather than decorating it.

    How this differs from newFrame.py: newFrame.py used rest days for the
    uncertainty inputs. Rest days describe the teams and say nothing about our
    confidence in our own estimate.

    Everything here is known at kickoff, since closing odds are the kickoff
    price, so no shifting is needed -- unlike the shot features below.

    Caveat: closing odds are absent from the earlier part of the historical
    record, so these columns cannot enter the feature list unconditionally.
    Note also that a missing value is not a zero: zero drift claims the price
    did not move, which is a different statement from not knowing whether it
    moved.
    """
    df = df.copy()

    # --- drift: positive means the price shortened into kickoff -----------
    for o in OUTCOMES:
        open_col, close_col = f"Avg{o}", f"AvgC{o}"
        if open_col in df.columns and close_col in df.columns:
            df[f"drift_{o}"] = np.log(df[open_col]) - np.log(df[close_col])
        else:
            df[f"drift_{o}"] = np.nan

    df["abs_drift"] = df[[f"drift_{o}" for o in OUTCOMES]].abs().sum(axis=1)

    # --- dispersion: how far the best price beats the average -------------
    for o in OUTCOMES:
        max_col, avg_col = f"Max{o}", f"Avg{o}"
        if max_col in df.columns and avg_col in df.columns:
            df[f"spread_{o}"] = df[max_col] / df[avg_col] - 1.0
        else:
            df[f"spread_{o}"] = np.nan

    df["spread_mean"] = df[[f"spread_{o}" for o in OUTCOMES]].mean(axis=1)

    # How much more than a whole the closing prices add up to: the book's
    # margin on this match.
    bench = pick_benchmark_columns(df)
    df["overround_close"] = sum(1.0 / df[c] for c in bench) - 1.0

    # How uncertain the market itself is. An evenly priced match is genuinely
    # open; a heavily one-sided one is not.
    p_ref = df[MARKET_COLS].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["p_ref_entropy"] = -np.nansum(p_ref * np.log(np.clip(p_ref, 1e-12, 1)),
                                         axis=1)
    return df


def add_shot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling shot and shots-on-target features.

    From n.pdf, reduced to what our data actually contains.

    What n.pdf asked for and why we cannot give it: it proposes expected
    goals, expected assists, deep completions and press intensity. None of
    those are in football-data.co.uk, and getting them means a second data
    source with its own join keys and its own gaps.

    What we substitute and why it is defensible: shots and shots on target.
    Shots on target is the standard cheap stand-in for expected goals, and it
    beats goals for the same reason expected goals does -- goals are rare, so
    a team's recent goal count is a noisy measure of how well it played, while
    its recent shot count is less noisy.

    How this differs from newFrame.py: newFrame.py built every rolling feature
    from goals and used no shot information at all, so this is the cheapest
    available improvement to the model's inputs.

    What to expect: shot-based form is still form, and the market prices form.
    The honest guess is that this improves the model's standalone accuracy
    while adding little to its DIFFERENCE from the market, which is what we
    actually trade on. If that happens, report it -- it illustrates the gap
    between predicting well and predicting differently.

    Coverage trap: at least one division in this data carries no match
    statistics at all. This function leaves those rows missing rather than
    filling them, so the decision to drop or keep that division is made
    explicitly downstream instead of silently here.
    """
    df = df.copy()
    needed = ["HS", "AS", "HST", "AST"]
    if not all(c in df.columns for c in needed):
        for c in SHOT_FEATURE_COLS:
            df[c] = np.nan
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # Same long-table approach as the goal features: one row per team per
    # match, so a team's history includes its away games. Do not group by
    # HomeTeam -- that is the venue bug fixed in calculate_rolling_features.
    home = df[["Date", "HomeTeam", "HS", "AS", "HST", "AST"]].rename(
        columns={"HomeTeam": "Team", "HS": "SF", "AS": "SA",
                 "HST": "STF", "AST": "STA"})
    home["is_home"] = True
    home["match_id"] = home.index

    away = df[["Date", "AwayTeam", "AS", "HS", "AST", "HST"]].rename(
        columns={"AwayTeam": "Team", "AS": "SF", "HS": "SA",
                 "AST": "STF", "HST": "STA"})
    away["is_home"] = False
    away["match_id"] = away.index

    long = (pd.concat([home, away])
              .sort_values(["Date", "match_id"]).reset_index(drop=True))

    # Shift first: a match must never see its own shot counts. These are
    # post-match numbers, so without the shift this is leakage, not a feature.
    for col in ("SF", "SA", "STF", "STA"):
        long[f"{col}_5"] = long.groupby("Team")[col].transform(
            lambda x: x.shift(1).ewm(span=5, min_periods=3).mean())

    # What fraction of shots were on target: a crude shot-quality measure.
    long["shot_quality_5"] = long["STF_5"] / long["SF_5"].replace(0, np.nan)

    h = long[long["is_home"]].set_index("match_id")
    a = long[~long["is_home"]].set_index("match_id")

    for col in ("SF_5", "SA_5", "STF_5", "STA_5", "shot_quality_5"):
        df[f"home_{col}"] = h[col]
        df[f"away_{col}"] = a[col]
    return df


# =========================================================================
# PART 4 -- SELECTION
#
# The core of the reframing. Selection is one function, called everywhere.
#
# newFrame.py wrote the selection rule out three separate times -- in the
# backtest, the threshold sweep and the live slate -- with slightly different
# filters in each. That is how a sweep quietly stops agreeing with the
# backtest it is meant to summarise, and the mismatch is very hard to trace
# once it appears.
# =========================================================================


def count_candidates(preds_df: pd.DataFrame, allowed=("H", "A"),
                     max_odds: float = None) -> int:
    """
    Count how many bets COULD have been placed.

    Why: this is the denominator of coverage. Without it the curve can only
    show bet counts, which say nothing about how selective the rule is being.

    New. newFrame.py reported bet counts and never defined coverage.
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
                kelly_fraction: float = 0.25,
                max_stake: float = 3.0,
                criterion: str = "ev") -> pd.DataFrame:
    """
    THE SELECTION RULE. Everything else calls this.

    What it does: given probability estimates and a threshold, return the bets
    that clear it. In selective-prediction terms, we act on these and abstain
    on everything else.

    Bets are priced at the best available price and also settled at the
    average price, so the gain from simply shopping around can be reported
    separately from any model edge.

    HOW THIS DIFFERS FROM newFrame.py.

    One copy instead of three. See the note at the top of this section.

    Execution price is the best available price, not Pinnacle. newFrame.py
    both measured its edge against Pinnacle and paid at Pinnacle, which made a
    real edge and a modelling error indistinguishable.

    The odds cap is off by default. newFrame.py hard-coded one. Long odds are
    exactly where estimation error is largest and where the winner's curse
    should bite hardest, so capping the price cuts off the thing we are
    studying. If you reintroduce a cap, sweep it and report the sensitivity
    rather than fixing it silently.

    One threshold instead of two. newFrame.py applied a threshold to expected
    value AND a second one to the probability edge, moving them together. Two
    gates make the coverage axis meaningless, because you can no longer say
    what a given threshold selected.

    Flat stakes by default. newFrame.py always sized bets by estimated edge,
    which lets a bad estimate do damage twice -- once by getting the bet
    selected and again by sizing it large. Our claim is about selection, so
    the default isolates it. See compare_staking_rules().
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

        # WHAT THE THRESHOLD IS APPLIED TO.
        #
        # "ev" multiplies the probability by the odds, so the odds sit inside
        # the score. Raising the threshold then selects longer and longer
        # prices whatever the model thinks, and a model with no opinion at all
        # shows the same falling return -- which is the favourite-longshot
        # bias, not selection acting on estimation error.
        #
        # "edge" is how far our probability sits above the market's. No odds
        # in it, so a tighter threshold means more disagreement rather than
        # longer prices. That is what the claim is about.
        if criterion == "ev":
            score = ev
        elif criterion == "edge":
            score = p_est - preds_df[f"p_ref_{o}"]
        else:
            raise ValueError(f"unknown criterion {criterion!r}")

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
        return pd.DataFrame(columns=["match_key", "Date", "Season", "Div",
                                     "Selection", "Odds", "Odds_Avg", "p_est",
                                     "p_ref", "EV", "Score", "IsWin", "Stake",
                                     "PnL", "PnL_Avg"])

    bets = pd.concat(frames, ignore_index=True)

    # --- staking ----------------------------------------------------------
    if stake_rule == "flat":
        bets["Stake"] = 1.0
    elif stake_rule == "quarter_kelly":
        full_kelly = bets["EV"] / (bets["Odds"] - 1.0)
        bets["Stake"] = np.minimum(full_kelly * kelly_fraction * 100, max_stake)
        bets["Stake"] = bets["Stake"].clip(lower=0.0)
    else:
        raise ValueError(f"unknown stake_rule {stake_rule!r}")

    # --- settle, at the best price and again at the average price ---------
    bets["PnL"] = np.where(bets["IsWin"] == 1,
                           bets["Stake"] * (bets["Odds"] - 1.0),
                           -bets["Stake"])
    ref_ok = bets["Odds_Avg"].notna() & (bets["Odds_Avg"] > 1.0)
    bets["PnL_Avg"] = np.where(
        ref_ok,
        np.where(bets["IsWin"] == 1,
                 bets["Stake"] * (bets["Odds_Avg"] - 1.0), -bets["Stake"]),
        np.nan)
    return bets


def risk_coverage_curve(preds_df: pd.DataFrame, prob_prefix: str = "p_corr",
                        taus: np.ndarray = None, allowed=("H", "A"),
                        stake_rule: str = "flat", n_boot: int = 2000,
                        min_bets: int = 30, max_odds: float = None,
                        criterion: str = "ev") -> pd.DataFrame:
    """
    THE HEADLINE FIGURE.

    What it does: sweeps the threshold and records, at each value, how many
    bets were selected (coverage) and what they returned (risk), with a
    bootstrapped interval.

    HOW THIS DIFFERS FROM newFrame.py's threshold sweep.

    Coverage is an explicit axis rather than implicit in the bet count. That
    is what makes this a risk-coverage curve and connects it to the selective
    classification literature, where the same curve evaluates classifiers that
    are allowed to abstain.

    Every point carries an interval. A curve of bare point estimates cannot
    support a claim that two curves differ in shape, and shape is the entire
    claim.

    Thresholds are spread over the range that actually occurs in the data,
    instead of a short hard-coded list that cannot reveal a shape.

    WHAT TO LOOK FOR. Plot two curves, uncorrected and corrected. The
    uncorrected one is predicted to get WORSE as coverage falls, because
    tightening the threshold selects harder for estimation error, so the
    surviving bets are increasingly those whose edge was overestimated. The
    corrected one is predicted to stay flat.

    They must differ in SHAPE, not merely in position along the threshold
    axis. A corrected curve that is the uncorrected one shifted sideways is a
    relabelled axis, not a correction. check_non_vacuousness() tests that
    formally; this is where it becomes visible.

    All three outcomes are reportable: curves differing in shape supports the
    claim; curves coinciding means the correction is vacuous, which is a
    finding about the method rather than a failed project; and two flat curves
    with wide intervals means there is not enough test data yet.
    """
    if taus is None:
        # Spread thresholds over the expected values that actually occur, so
        # the curve spans real coverage rather than an arbitrary range.
        vals = []
        for o in allowed:
            if f"Max{o}" in preds_df:
                if criterion == "ev":
                    vals.append((preds_df[f"{prob_prefix}_{o}"] * preds_df[f"Max{o}"] - 1).dropna())
                else:
                    vals.append((preds_df[f"{prob_prefix}_{o}"] - preds_df[f"p_ref_{o}"]).dropna())
        pooled = pd.concat(vals) if vals else pd.Series([0.0])
        taus = np.quantile(pooled, np.linspace(0.50, 0.995, 25))

    denom = count_candidates(preds_df, allowed, max_odds)
    rows = []
    for tau in taus:
        bets = select_bets(preds_df, tau, prob_prefix, allowed,
                           max_odds=max_odds, stake_rule=stake_rule,
                           criterion=criterion)
        if len(bets) < min_bets:
            # Below this the interval is meaningless. Record the point and
            # move on rather than drawing a curve into noise.
            rows.append({"tau": tau, "coverage": len(bets) / denom if denom else 0,
                         "n_bets": len(bets), "roi": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan})
            continue
        b = bootstrap_roi(bets, n_boot=n_boot)
        rows.append({"tau": tau, "coverage": len(bets) / denom if denom else 0,
                     "n_bets": len(bets), "avg_odds": bets["Odds"].mean(),
                     "win_rate": bets["IsWin"].mean(), "roi": b["roi"],
                     "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
    return pd.DataFrame(rows)


def aurc(curve_df: pd.DataFrame) -> float:
    """
    Summarise the whole risk-coverage curve as one number.

    Why: it is the standard metric in selective prediction, and it removes the
    temptation to quote whichever threshold happened to look best. newFrame.py
    tuned its threshold on test data, which turned a loss into a reported
    gain. A metric defined over the entire curve makes that impossible.

    New. newFrame.py had no summary metric.
    """
    d = curve_df.dropna(subset=["roi"]).sort_values("coverage")
    if len(d) < 2:
        return np.nan
    span = d["coverage"].iloc[-1] - d["coverage"].iloc[0]
    if span <= 0:
        return np.nan
    return float(np.trapezoid(d["roi"].values, d["coverage"].values) / span)


# =========================================================================
# PART 5 -- UNCERTAINTY
#
# The winner's curse happens because our estimation error is not the same size
# for every match. Selection picks the extreme tail of our estimate, and that
# tail fills up with the matches where our error was biggest, not where our
# edge was biggest. Correcting it needs a per-match estimate of our own error.
# =========================================================================


def uncertainty_inputs(df: pd.DataFrame, p_model: np.ndarray = None) -> np.ndarray:
    """
    Assemble the inputs to the uncertainty model.

    Why these inputs: every one is about OUR confidence, not about the teams.
    How much the books disagree, how much margin the book is taking, how open
    the match is on the market's own numbers, and how far we have strayed from
    the market.

    How this differs from newFrame.py: newFrame.py used rest days and the home
    probability. Rest days are a fact about the fixture, not about how wrong
    we are likely to be, so they cannot do this job.
    """
    d = df.copy()
    if p_model is not None:
        p_ref = d[MARKET_COLS].to_numpy(dtype=float)
        d["abs_deviation"] = np.abs(p_model - p_ref).sum(axis=1)
    elif "abs_deviation" not in d:
        d["abs_deviation"] = 0.0

    cols = [c for c in UNCERTAINTY_COLS if c in d.columns]
    X = d[cols].to_numpy(dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def fit_uncertainty_head(df_val: pd.DataFrame, p_model_val: np.ndarray,
                         y_val: np.ndarray, seed: int = 42):
    """
    Train a model to predict our own error, match by match.

    How it works: for each validation match, measure how badly the model did
    on that single match, then fit a regressor predicting that from the
    uncertainty inputs. Matches predicted to go badly get shrunk harder toward
    the market.

    Why it matters: newFrame.py had the right idea -- a per-match shrinkage
    weight -- but chose its inputs by hand, and chose ones that carry no
    information about our confidence. This learns the answer instead of
    guessing at it, which is the strongest machine-learning claim available to
    the project and sits exactly where the research claim lives.

    Why validation data only: on training data the model's error is
    artificially small, so the head would learn a confidence that does not
    survive contact with new data.

    Returns the head and a check. The check reports whether predicted error
    actually tracks realised error -- if it does not, the head learned
    nothing, and the honest move is to keep the simpler version and say so.

    n.pdf did not propose this. It is the part of the design that its
    "more features" item points toward without naming.
    """
    X = uncertainty_inputs(df_val, p_model_val)
    eps = 1e-12
    realised = -np.log(np.clip(p_model_val[np.arange(len(y_val)), y_val], eps, 1.0))

    head = make_regressor(seed)
    head.fit(X, realised)

    pred = head.predict(X)
    r = float(np.corrcoef(pred, realised)[0, 1]) if len(pred) > 2 else np.nan
    check = {"in_sample_corr": r, "n": len(realised),
             "mean_realised_loss": float(realised.mean())}
    return head, check


def shrink_by_uncertainty(p_ref: np.ndarray, p_model: np.ndarray,
                          predicted_error: np.ndarray,
                          w_max: float = 1.0) -> np.ndarray:
    """
    Shrink each match by how badly we expect to do on it.

    What it does: the same geometric blend as Part 1, but the weight comes
    from predicted error rather than a hand-picked formula. The weight is high
    (trust the model) where predicted error is low, and low (trust the market)
    where predicted error is high.

    Why the direction is asserted below: getting it backwards inverts the
    whole correction and would still produce a plausible-looking curve, so the
    mistake would not announce itself.

    Why the weight is clipped at percentiles rather than the extremes: a few
    unusual matches would otherwise squash every other match into a narrow
    band and the weight would stop discriminating.

    How this differs from newFrame.py: same blend, learned weight.
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

    eps = 1e-12
    p_ref_c = np.clip(p_ref, eps, 1 - eps)
    p_model_c = np.clip(p_model, eps, 1 - eps)
    wc = w[:, None]
    log_p = (1 - wc) * np.log(p_ref_c) + wc * np.log(p_model_c)
    p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float64)


# =========================================================================
# PART 6 -- CORRECTIONS
#
# The shrinkage in Part 1 is one way to correct for selection bias, not the
# only one. Implementing a competitor turns "we applied shrinkage" into "we
# compared corrections", which is a much stronger claim for one function.
# =========================================================================


def tweedie_correction(p_model: np.ndarray, p_ref: np.ndarray,
                       sigma2: float = None, bw: float = 0.3) -> np.ndarray:
    """
    Correct the deviations using Tweedie's formula.

    What it does: adjusts each deviation using the spread of ALL observed
    deviations, rather than pulling everything toward a common point by a
    fixed factor.

    Why it belongs here: this is the standard treatment of the winner's curse
    elsewhere in statistics. Selecting on a noisy quantity guarantees the
    selected values are overstated, and this corrects for that using only the
    observed distribution -- it needs no assumption about the true edges.

    Why it is not just a variant of the shrinkage in Part 1: that shrinks
    everything by one global factor, while this shrinks unevenly and more in
    the tails. Selection happens in the tail, so a correction that treats the
    tail differently from the body is a different proposal.

    Why that difference is the point: a uniform factor is exactly the kind of
    correction that can turn out to be identical to just moving the threshold.
    An uneven one reorders the candidates, and no change of threshold can
    reproduce a reordering. So this is the natural answer if
    check_non_vacuousness() finds the simple version vacuous.

    On sigma2: it is how much of the observed spread is noise rather than real
    disagreement. It must be FITTED, not guessed -- a wrong value makes the
    correction actively harmful. Use fit_tweedie_sigma2() and pass the result
    in. The default here exists only so the function is callable.

    New. Not in newFrame.py, and not proposed by n.pdf.
    """
    eps = 1e-12
    pm = np.clip(np.asarray(p_model, float), eps, 1 - eps)
    pr = np.clip(np.asarray(p_ref, float), eps, 1 - eps)

    z = np.log(pm / pr)          # our deviation from the market, in log space
    flat = z.ravel()
    flat = flat[np.isfinite(flat)]

    if sigma2 is None:
        # Deliberately small placeholder. Fit it with fit_tweedie_sigma2().
        sigma2 = 0.05 * float(np.var(flat))

    # Estimate the density of deviations, then its log-derivative.
    kde = stats.gaussian_kde(flat, bw_method=bw)
    grid = np.linspace(flat.min(), flat.max(), 512)
    dens = np.clip(kde(grid), 1e-12, None)
    dlog = np.gradient(np.log(dens), grid)

    z_corrected = z + sigma2 * np.interp(z, grid, dlog)

    p = pr * np.exp(z_corrected)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


def fit_tweedie_sigma2(p_model_val: np.ndarray, p_ref_val: np.ndarray,
                       y_val: np.ndarray, bw: float = 0.3) -> float:
    """
    Fit Tweedie's noise term on validation data.

    Why: guessing this value wrong makes the correction worse than doing
    nothing, so it has to be fitted like any other parameter.

    Why validation data only: the same reason as fit_static_shrinkage. On
    training data the model looks better than it is, so the noise term would
    come out too small and the correction too weak.

    New, and it exists because an early version used a guessed value.
    """
    var_z = float(np.var(np.log(np.clip(p_model_val, 1e-12, 1) /
                                np.clip(p_ref_val, 1e-12, 1))))

    def objective(frac):
        p = tweedie_correction(p_model_val, p_ref_val,
                               sigma2=max(frac, 1e-6) * var_z, bw=bw)
        return log_loss(y_val, p, labels=[0, 1, 2])

    res = minimize_scalar(objective, bounds=(1e-4, 1.0), method="bounded")
    return float(res.x) * var_z


def conformal_lower_bound(p_model_cal: np.ndarray, y_cal: np.ndarray,
                          p_model_test: np.ndarray,
                          alpha: float = 0.1) -> np.ndarray:
    """
    Put a conservative floor under each probability.

    What it does: on calibration data, measure how far the model's
    probabilities typically miss by, take a high quantile of that, and
    subtract it from the test estimates.

    Why: thresholding a point estimate has no error control at all, which is
    exactly why the winner's curse arises. Selecting on a pessimistic lower
    bound means we only bet when the edge survives a dim view of our own
    accuracy.

    Honest scope: this is a simplified split-conformal bound, not full
    conformal risk control. It is worth having as a third arm alongside raw
    and corrected selection -- if it flattens the curve too, that supports the
    diagnosis from an independent direction.

    Cost: it needs a calibration split kept apart from both training and test,
    which uses up data. Not worth attempting until more seasons are available.

    New. Not in newFrame.py, not proposed by n.pdf.
    """
    onehot = np.zeros_like(p_model_cal)
    onehot[np.arange(len(y_cal)), y_cal] = 1.0
    residuals = np.abs(onehot - p_model_cal).ravel()
    q = float(np.quantile(residuals, 1.0 - alpha))
    return np.clip(p_model_test - q, 1e-12, 1.0)


# =========================================================================
# PART 7 -- EVALUATION
# =========================================================================


def bootstrap_roi(bets_df: pd.DataFrame, n_boot: int = 10000,
                  alpha: float = 0.05, seed: int = 42,
                  pnl_col: str = "PnL") -> dict:
    """
    Return on investment with a confidence interval.

    Why every reported return goes through here: a return without an interval
    is not a result. It cannot be compared with another return, and it cannot
    be distinguished from zero.

    WHY IT RESAMPLES MATCHES, NOT BETS. Two bets on the same match share an
    outcome, so they are correlated. Resampling individual bets pretends they
    are independent and produces an interval that is too narrow. That error is
    dangerous precisely because a too-narrow interval looks confident rather
    than obviously broken.

    New. newFrame.py reported bare point estimates everywhere, which is why
    its headline result could not be assessed.
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
    dropped = int((~ok).sum())     # draws that happened to stake nothing

    return {
        "roi": roi,
        "ci_low": float(np.percentile(boot_roi, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot_roi, 100 * (1 - alpha / 2))),
        "p_le_zero": float(np.mean(boot_roi <= 0)),
        "n_bets": int(len(bets_df)),
        "n_matches": n_matches,
        "dropped": dropped,
    }


def report_roi(bets_df: pd.DataFrame, label: str = "", n_boot: int = 10000) -> dict:
    """
    Print a return with its interval and the line-shopping gain.

    Why the second number: the gap between settling at the best price and at
    the average price is mechanical. It is what shopping around is worth to
    somebody with no model at all, so it must never be presented as model
    edge. Printing both makes that impossible to confuse.

    New. newFrame.py printed one return, at a price it should not have used.
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


def compare_staking_rules(preds_df: pd.DataFrame, prob_prefix: str = "p_corr",
                          taus: np.ndarray = None, n_boot: int = 1000) -> pd.DataFrame:
    """
    Compare flat stakes against fractional Kelly.

    From n.pdf's staking item, with its learning agent removed.

    Why the agent is excluded: an agent that learns stake sizes mixes up WHICH
    bets to place with HOW MUCH to place on them, and this project's claim is
    about the first of those. It would also remove the separable selection
    step the whole project depends on.

    Why the question is still worth asking, and how this differs from
    newFrame.py: newFrame.py always sized bets by estimated edge. That lets a
    bad estimate do damage twice -- once by getting the bet selected, again by
    sizing it large. Measuring a selection effect through a rule that
    amplifies the same error confuses the two. Under flat stakes, any decline
    as coverage falls is down to selection alone.

    Recommendation: report flat as the main result and Kelly as a robustness
    check. If the corrected and uncorrected curves separate under Kelly but
    not under flat, the effect is in the sizing rather than the selection, and
    the claim does not hold as written.
    """
    out = []
    for rule in ("flat", "quarter_kelly"):
        curve = risk_coverage_curve(preds_df, prob_prefix, taus,
                                    stake_rule=rule, n_boot=n_boot)
        curve["stake_rule"] = rule
        curve["aurc"] = aurc(curve)
        out.append(curve)
    return pd.concat(out, ignore_index=True)


def check_non_vacuousness(preds_df: pd.DataFrame,
                          corrected_prefix: str = "p_corr",
                          raw_prefix: str = "p_model",
                          taus: np.ndarray = None,
                          n_boot: int = 1000) -> pd.DataFrame:
    """
    Test whether the correction is real or just a relabelled threshold.

    THE RISK IT GUARDS AGAINST: shrinking and then thresholding might be
    exactly the same as thresholding at a different level and not shrinking at
    all. If so we renamed an axis, and the headline figure is an artefact.

    Why it must be run: this is the first thing a sceptical examiner will
    test. Much better to have tested it ourselves.

    How it works: for each threshold used with correction, find the threshold
    that picks the SAME NUMBER of bets without correction, then compare the
    two sets directly. Equal sizes make the comparison fair.

    How to read it: a high overlap with intervals that sit on top of each
    other means the correction is vacuous, and that should be said plainly. An
    overlap well below one means the correction REORDERS which matches look
    attractive, and no change of threshold can do that.

    What to expect: a single global shrinkage factor is close to a monotone
    rescaling, so it may well score badly here. That is the argument for
    tweedie_correction(), which shrinks unevenly and therefore can reorder. If
    one is vacuous and the other is not, that contrast is itself a result.

    New. newFrame.py had no such check.
    """
    if taus is None:
        evs = pd.concat([
            (preds_df[f"{corrected_prefix}_{o}"] * preds_df[f"Max{o}"] - 1).dropna()
            for o in ("H", "A") if f"Max{o}" in preds_df])
        taus = np.quantile(evs, np.linspace(0.60, 0.99, 12))

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
            n_mid = len(select_bets(preds_df, mid, raw_prefix))
            if n_mid > n_target:
                lo = mid
            else:
                hi = mid
        tau_prime = (lo + hi) / 2
        bets_r = select_bets(preds_df, tau_prime, raw_prefix)

        id_c = set(zip(bets_c["match_key"], bets_c["Selection"]))
        id_r = set(zip(bets_r["match_key"], bets_r["Selection"]))
        overlap = len(id_c & id_r) / len(id_c) if id_c else np.nan

        bc = bootstrap_roi(bets_c, n_boot=n_boot)
        br = bootstrap_roi(bets_r, n_boot=n_boot)
        rows.append({
            "tau": tau, "tau_prime": tau_prime,
            "n_corrected": len(bets_c), "n_raw": len(bets_r),
            "overlap": overlap,
            "roi_corrected": bc["roi"], "ci_corrected": (bc["ci_low"], bc["ci_high"]),
            "roi_raw": br["roi"], "ci_raw": (br["ci_low"], br["ci_high"]),
        })
    return pd.DataFrame(rows)


def scoring_report(y_true: np.ndarray, p: np.ndarray, label: str = "") -> dict:
    """
    Score a set of probabilities properly.

    Reports log loss, ranked probability score, Brier score and calibration
    error. Accuracy is computed too, but named so it cannot be quoted by
    accident.

    Why accuracy is excluded from ranking: a model can be more accurate and
    worse calibrated at the same time. We bet on the probabilities themselves,
    not on which outcome is most likely, so calibration is what matters and
    accuracy can actively mislead.

    Why the ranked probability score is included: home, draw and away are
    ordered, so being wrong by a lot should cost more than being wrong by a
    little, and plain log loss does not know that.

    Use the de-vigged market as the baseline. A dummy model that simply
    returns the market price should reproduce the market's own score exactly,
    which is the sanity check for this whole harness.

    New. newFrame.py reported log loss only.
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
                ece += abs(p[m, k].mean() - onehot[m, k].mean()) * m.sum() / (len(p) * p.shape[1])

    return {
        "label": label,
        "log_loss": float(log_loss(y_true, p, labels=[0, 1, 2])),
        "rps": rps,
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "ece": float(ece),
        "accuracy_do_not_rank_on_this": float(np.mean(p.argmax(axis=1) == y_true)),
    }


# =========================================================================
# PART 8 -- ORCHESTRATION
# =========================================================================


def fit_predict_fold(train_df: pd.DataFrame, val_df: pd.DataFrame,
                     test_df: pd.DataFrame, features: list,
                     seed: int = 42) -> dict:
    """
    Train, calibrate and predict for one fold.

    Why calibration is not optional: raw gradient-boosting probabilities are
    unreliable near zero and one, and we bet on the numbers themselves rather
    than on which outcome is largest. Never correct an uncalibrated
    probability -- that is the order-of-operations rule at the top of the file.

    Same model and calibration as newFrame.py. The difference is that this is
    wrapped so walk-forward can call it once per fold, instead of the whole
    training run happening once on a fixed split.
    """
    Xtr, ytr = train_df[features], train_df["target"].astype(int)
    model = CalibratedClassifierCV(estimator=make_classifier(seed),
                                   method="isotonic", cv=3)
    model.fit(Xtr, ytr)
    return {
        "model": model,
        "p_val": model.predict_proba(val_df[features]),
        "p_test": model.predict_proba(test_df[features]),
    }


def train_two_stage(train_df: pd.DataFrame, val_df: pd.DataFrame,
                    test_df: pd.DataFrame, seed: int = 42) -> dict:
    """
    Two-stage model, where stage one never sees the market.

    From n.pdf's two-stage proposal, built to this project's specification
    rather than n.pdf's -- see the differences below.

    WHAT PROBLEM IT SOLVES. In newFrame.py the market's own probabilities were
    model inputs. So the model partly copied the market, and the pipeline then
    corrected its output back toward that same market. The deviation being
    corrected was partly an echo of the thing it was measured against, so the
    same information was used twice.

    Stage one trains without any market data and forms an independent view.
    Stage two combines that view with the market. The disagreement between
    them is then real rather than circular.

    What to expect: the fitted correction weight should come out LOWER for the
    two-stage version, because stage one's disagreement is genuine. If it
    comes out higher, that is a finding to investigate, not a bug to patch.

    WHERE THIS DIFFERS FROM n.pdf, deliberately:

      n.pdf has stage one predict physical quantities such as expected goals
      and shot counts. Here it predicts outcome probabilities directly, which
      keeps the two stages comparable and avoids needing data we do not have.

      n.pdf feeds stage two raw odds. That would put the execution price
      inside the model and bring back exactly the circularity this design
      removes, so stage two receives de-vigged probabilities instead.

      n.pdf does not mention out-of-fold prediction, which is the detail that
      decides whether this works at all. See the comment below.
    """
    base = [c for c in BASE_FEATURE_COLS if c in train_df.columns]

    # --- stage one: no market information whatsoever ----------------------
    s1 = CalibratedClassifierCV(estimator=make_classifier(seed),
                                method="isotonic", cv=3)
    Xtr, ytr = train_df[base], train_df["target"].astype(int)

    # Stage two must see stage one's predictions for data stage one did NOT
    # train on. Given in-sample predictions it would learn to over-trust them,
    # because they look far better than stage one will manage on new data.
    oof = cross_val_predict(s1, Xtr, ytr, cv=3, method="predict_proba")
    s1.fit(Xtr, ytr)

    def stage2_inputs(frame, p1):
        parts = [p1, frame[MARKET_COLS].to_numpy(float)]
        extra = [c for c in UNCERTAINTY_COLS if c in frame.columns]
        if extra:
            parts.append(np.nan_to_num(frame[extra].to_numpy(float)))
        return np.hstack(parts)

    s2 = CalibratedClassifierCV(estimator=make_classifier(seed),
                                method="isotonic", cv=3)
    s2.fit(stage2_inputs(train_df, oof), ytr)

    p1_val, p1_test = s1.predict_proba(val_df[base]), s1.predict_proba(test_df[base])
    return {
        "stage1": s1, "stage2": s2,
        "p1_val": p1_val, "p1_test": p1_test,
        "p_val": s2.predict_proba(stage2_inputs(val_df, p1_val)),
        "p_test": s2.predict_proba(stage2_inputs(test_df, p1_test)),
    }


def run_walk_forward(df: pd.DataFrame, features: list = None,
                     min_train_seasons: int = 1, two_stage: bool = True,
                     use_learned_uncertainty: bool = True,
                     min_test_matches: int = 500,
                     seed: int = 42) -> tuple:
    """
    Walk forward through the seasons instead of using one fixed split.

    What it does: for each test season, train on everything up to two seasons
    before it, validate on the season before it, test on it, then move forward
    and repeat. Out-of-sample predictions from every fold are pooled.

    WHY THIS IS THE MOST IMPORTANT FUNCTION HERE. More TRAINING seasons do not
    narrow the confidence interval; they may improve the model slightly, but
    they add no test data. Only more TEST seasons do. Walking forward turns
    one test period into many, which is the only change in this file capable
    of making the result statistically significant.

    Why the correction weight is refitted every fold: fitting it once assumes
    the right amount of correction has been the same for decades, which is
    unlikely. Whether it moves is itself one of the questions worth answering,
    and its trajectory over time is the era analysis.

    How this differs from newFrame.py: newFrame.py hard-coded one set of
    training, validation and test seasons, fitted the weight once, and named
    seasons the repository did not contain -- which would have trained the
    model on an empty table while still printing results.

    Returns the pooled out-of-sample predictions and the per-fold diagnostics.
    """
    if features is None:
        features = [c for c in FEATURE_COLS if c in df.columns]

    seasons = sorted(df["Season"].astype(str).unique())
    if len(seasons) < min_train_seasons + 2:
        raise ValueError(
            f"Walk-forward needs at least {min_train_seasons + 2} seasons; the "
            f"data has {len(seasons)}: {seasons}. Run fetch_seasons() first -- "
            "with one season there is no walk-forward, and every interval stays "
            "too wide to conclude anything.")

    needed = features + ["target"] + [c for c in EXECUTION_COLS if c in df.columns]
    clean = df[df[needed].notna().all(axis=1)].copy()
    clean["Season"] = clean["Season"].astype(str)

    all_preds, folds = [], []
    for i in range(min_train_seasons + 1, len(seasons)):
        test_s, val_s = seasons[i], seasons[i - 1]
        train_s = seasons[:i - 1]

        tr = clean[clean["Season"].isin(train_s)]
        va = clean[clean["Season"] == val_s]
        te = clean[clean["Season"] == test_s].copy()
        # Skip folds too small to mean anything. A test season with very few
        # matches produces a return with an interval so wide it cannot be
        # distinguished from any other number, and pooling it in would add
        # noise while looking like extra evidence.
        if len(tr) < 500 or len(va) < 100 or len(te) < min_test_matches:
            print(f"  skipping fold {test_s}: train {len(tr)}, val {len(va)}, "
                  f"test {len(te)} -- too small")
            continue

        if two_stage:
            fit = train_two_stage(tr, va, te, seed)
        else:
            fit = fit_predict_fold(tr, va, te, features, seed)

        p_ref_val = va[MARKET_COLS].to_numpy(float)
        p_ref_test = te[MARKET_COLS].to_numpy(float)
        y_val = va["target"].astype(int).to_numpy()

        # WHICH ESTIMATE GETS CORRECTED, and why it matters more than it looks.
        #
        # The correction pulls our estimate toward the market. That only means
        # anything if our estimate was formed independently of the market. If
        # the model already had the market price as an input, its output is
        # mostly a copy of the market, there is almost nothing left to pull,
        # and the fitted weight goes to one -- the correction switches itself
        # off while still appearing to run.
        #
        # So with the two-stage model we correct STAGE ONE, which never saw
        # the market. Stage two is kept as the comparison: it is the better
        # forecaster, but its disagreement with the market is not independent
        # of it, so it is not the thing to shrink.
        if two_stage:
            p_indep_val, p_indep_test = fit["p1_val"], fit["p1_test"]
        else:
            p_indep_val, p_indep_test = fit["p_val"], fit["p_test"]

        p_val, p_test = fit["p_val"], fit["p_test"]

        # Refit the correction weight on THIS fold's validation season.
        w = fit_static_shrinkage(p_ref_val, p_indep_val, y_val)
        p_corr = apply_static_shrinkage(p_ref_test, p_indep_test, w)
        te[outcome_cols("p_indep")] = p_indep_test

        # Optional: the learned per-match version from Part 5.
        if use_learned_uncertainty:
            try:
                head, check = fit_uncertainty_head(va, p_indep_val, y_val, seed)
                err_test = head.predict(uncertainty_inputs(te, p_indep_test))
                p_unc = shrink_by_uncertainty(p_ref_test, p_indep_test, err_test)
                te[outcome_cols("p_unc")] = p_unc
            except Exception as exc:                       # pragma: no cover
                warnings.warn(f"uncertainty head failed on {test_s}: {exc}")
                check = {"in_sample_corr": np.nan}
        else:
            check = {"in_sample_corr": np.nan}

        te[outcome_cols("p_model")] = p_test
        te[outcome_cols("p_corr")] = p_corr
        all_preds.append(te)
        folds.append({
            "test_season": test_s, "val_season": val_s,
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "w": w,
            "uncertainty_corr": check.get("in_sample_corr", np.nan),
            "ll_market": log_loss(te["target"].astype(int), p_ref_test, labels=[0, 1, 2]),
            "ll_indep": log_loss(te["target"].astype(int), p_indep_test, labels=[0, 1, 2]),
            "ll_model": log_loss(te["target"].astype(int), p_test, labels=[0, 1, 2]),
            "ll_corrected": log_loss(te["target"].astype(int), p_corr, labels=[0, 1, 2]),
        })

        # A correction weight this high means the model is being trusted more
        # than the market, which is worth stopping to explain rather than
        # discovering later in a plot.
        if w > W_WARNING_LEVEL:
            warnings.warn(f"fold {test_s}: w = {w:.3f} exceeds {W_WARNING_LEVEL} "
                          "-- investigate before trusting this fold")

    if not all_preds:
        raise ValueError("No fold had enough data. Check the season coverage.")

    return pd.concat(all_preds, ignore_index=True), pd.DataFrame(folds)


def era_breakdown(preds_df: pd.DataFrame, folds_df: pd.DataFrame,
                  prob_prefix: str = "p_corr", tau: float = 0.02,
                  era_split: str = "1920") -> pd.DataFrame:
    """
    Break the results down by season and era, and track the correction weight.

    Why never report one pooled figure across many seasons: the market changes
    over time, so a single number can hide the effect being alive early and
    dead later. Splitting it is the difference between describing a trend and
    averaging one away.

    Why the era marker: the price columns changed provider partway through the
    historical record, so a jump at that point may be an artefact of the data
    rather than a change in the market. Marking it lets the two be told apart.

    New. newFrame.py pooled everything.
    """
    rows = []
    for season, part in preds_df.groupby(preds_df["Season"].astype(str)):
        bets = select_bets(part, tau, prob_prefix)
        b = bootstrap_roi(bets, n_boot=2000)
        w = folds_df.loc[folds_df["test_season"] == season, "w"]
        rows.append({
            "season": season,
            "era": "early" if season < era_split else "late",
            "w": float(w.iloc[0]) if len(w) else np.nan,
            "n_bets": b["n_bets"], "roi": b["roi"],
            "ci_low": b["ci_low"], "ci_high": b["ci_high"],
        })
    return pd.DataFrame(rows)


# =========================================================================
# PART 9 -- MODEL POOL
#
# Why this exists, and it is not to find a better model.
#
# The claim is that thresholding on an uncertain estimate selects for
# estimation error. If that is true it is a property of the SELECTION RULE and
# should appear whatever produced the estimate. If the declining curve shows up
# only under one library, it is a quirk of that library.
#
# So this runs the same pipeline over several different model families and
# compares them. Agreement across families is the evidence.
#
# It also produces two rankings, one by accuracy and one by calibration, to
# show they disagree -- which is the argument for not ranking on accuracy.
# =========================================================================


def available_models(seed: int = 42) -> dict:
    """
    Return every model family installed on this machine.

    Missing libraries are skipped rather than raising, so the comparison runs
    with whatever is present and reports what it used.

    Each entry is a plain scikit-learn style classifier. Calibration,
    correction and backtesting are unchanged for all of them -- that is the
    point: only the estimator varies.
    """
    models = {}

    if HAVE_LGB:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.01, num_leaves=15, max_depth=3,
            subsample=0.7, colsample_bytree=0.7, random_state=seed, verbosity=-1)

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=100, learning_rate=0.01, max_depth=3,
            subsample=0.7, colsample_bytree=0.7, random_state=seed,
            objective="multi:softprob", num_class=3, verbosity=0)
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(
            iterations=100, learning_rate=0.01, depth=3,
            random_seed=seed, verbose=0, loss_function="MultiClass")
    except ImportError:
        pass

    # A linear model, included because it cannot fit the same shapes the trees
    # can. If the selection effect appears here too, it is not a tree artefact.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models["logistic"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, random_state=seed))

    # A trivial model that always predicts the base rates. If anything scores
    # worse than this, something is broken rather than merely weak.
    from sklearn.dummy import DummyClassifier
    models["base rates"] = DummyClassifier(strategy="prior")

    return models


def run_model_pool(df: pd.DataFrame, features: list = None,
                   taus: np.ndarray = None, seed: int = 42) -> tuple:
    """
    Run the whole pipeline once per model family and compare.

    Returns (scores, curves, rankings).

      scores    one row per family: log loss, RPS, Brier, ECE, accuracy, the
                fitted correction weight, and the area under the corrected and
                uncorrected risk-coverage curves.
      curves    the risk-coverage curve for every family, corrected and not.
      rankings  the same families ordered by accuracy and by calibration, side
                by side, so the disagreement between them is visible.

    Read the AURC columns first. If the uncorrected number is negative for
    every family and the corrected one is higher for every family, the effect
    is structural rather than a property of any one estimator. That is the
    result this function exists to produce.
    """
    if features is None:
        features = [c for c in BASE_FEATURE_COLS if c in df.columns]

    seasons = sorted(df["Season"].astype(str).unique())
    needed = features + ["target"] + [c for c in EXECUTION_COLS if c in df.columns]
    clean = df[df[needed].notna().all(axis=1)].copy()
    clean["Season"] = clean["Season"].astype(str)

    models = available_models(seed)
    print(f"Model pool: {', '.join(models)}")

    score_rows, curve_frames = [], []
    for name, estimator in models.items():
        fold_preds = []
        w_values = []

        for i in range(2, len(seasons)):
            test_s, val_s = seasons[i], seasons[i - 1]
            tr = clean[clean["Season"].isin(seasons[:i - 1])]
            va = clean[clean["Season"] == val_s]
            te = clean[clean["Season"] == test_s].copy()
            if len(tr) < 500 or len(va) < 100 or len(te) < 500:
                continue

            # Identical treatment for every family: same features, same
            # calibration, same correction, same backtest.
            model = CalibratedClassifierCV(estimator=estimator,
                                           method="isotonic", cv=3)
            model.fit(tr[features], tr["target"].astype(int))

            p_val = model.predict_proba(va[features])
            p_test = model.predict_proba(te[features])
            p_ref_val = va[MARKET_COLS].to_numpy(float)
            p_ref_test = te[MARKET_COLS].to_numpy(float)

            w = fit_static_shrinkage(p_ref_val, p_val,
                                     va["target"].astype(int).to_numpy())
            w_values.append(w)

            te[outcome_cols("p_indep")] = p_test
            te[outcome_cols("p_corr")] = apply_static_shrinkage(p_ref_test, p_test, w)
            fold_preds.append(te)

        if not fold_preds:
            continue
        preds = pd.concat(fold_preds, ignore_index=True)
        y = preds["target"].astype(int).to_numpy()

        c_raw = risk_coverage_curve(preds, "p_indep", taus, n_boot=400)
        c_cor = risk_coverage_curve(preds, "p_corr", taus, n_boot=400)
        for c, lab in ((c_raw, "uncorrected"), (c_cor, "corrected")):
            c = c.copy(); c["model"] = name; c["arm"] = lab
            curve_frames.append(c)

        s = scoring_report(y, preds[outcome_cols("p_indep")].to_numpy(float), name)
        s["w"] = float(np.mean(w_values))
        s["aurc_uncorrected"] = aurc(c_raw)
        s["aurc_corrected"] = aurc(c_cor)
        s["n_oos"] = len(preds)
        score_rows.append(s)
        print(f"  {name:<12} done")

    scores = pd.DataFrame(score_rows)

    # The Walsh & Joshi point: ranking by accuracy and by calibration gives
    # different answers, which is why accuracy is not a selection criterion.
    rankings = pd.DataFrame({
        "by accuracy": scores.sort_values("accuracy_do_not_rank_on_this",
                                          ascending=False)["label"].values,
        "by calibration (ECE)": scores.sort_values("ece")["label"].values,
        "by log loss": scores.sort_values("log_loss")["label"].values,
    })
    return scores, pd.concat(curve_frames, ignore_index=True), rankings


# =========================================================================
# MAIN
# =========================================================================
def main(source: str = "matches_multiseason.csv", mode: str = "multiseason",
         start_year: int = 2005, end_year: int = 2025, pool: bool = False):
    """
    Run the pipeline end to end.

    Modes:
      multiseason  the stacked multi-season file. The default, and the one
                   with enough seasons for walk-forward.
      export       the original single-season export, kept so the earlier
                   result can still be reproduced.
      local        a folder of per-season files already on disk.
      download     fetch the full history from the site, then run.

    What it prints, in order: what was loaded, how the market scores (the
    number any model has to beat), one line per walk-forward fold, the
    risk-coverage curve with and without the correction, the non-vacuousness
    check, and a breakdown by season.

    Read them in that order. A model that does not beat the market baseline
    has nothing to correct, and a curve whose intervals all overlap says
    nothing regardless of where its middle sits.
    """
    print("=" * 70)
    print("SELECTIVE BETTING PIPELINE")
    print("=" * 70)

    if mode == "multiseason":
        df = load_multiseason(source)
    elif mode == "export":
        df = load_and_prepare(source)
    elif mode == "local":
        df = load_local_seasons(source)
        df = add_shot_features(add_market_features(calculate_rolling_features(df)))
    elif mode == "download":
        df = fetch_seasons(start_year, end_year)
        df = add_shot_features(add_market_features(calculate_rolling_features(df)))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    complete = df[BASE_FEATURE_COLS].notna().all(axis=1).sum()
    print(f"\nFeatures built. {len(df):,} matches, {complete:,} with complete features.")

    # --- the baseline every model has to beat -----------------------------
    market = df[MARKET_COLS].to_numpy(float)
    ok = np.isfinite(market).all(axis=1) & df["target"].notna()
    print("\nMarket baseline (the score to beat):")
    for k, v in scoring_report(df.loc[ok, "target"].astype(int),
                               market[ok.to_numpy()], "market").items():
        print(f"  {k:<32} {v}")

    # --- walk-forward -----------------------------------------------------
    try:
        preds, folds = run_walk_forward(df)
    except ValueError as exc:
        print("\n" + "!" * 70)
        print("STOPPED:", exc)
        print("!" * 70)
        return df, None, None

    print("\nPer-fold diagnostics:")
    print(folds.round(4).to_string(index=False))

    # --- how the corrections score out of sample --------------------------
    y = preds["target"].astype(int).to_numpy()
    print("\nOut-of-sample scoring:")
    rows = [scoring_report(y, preds[MARKET_COLS].to_numpy(float), "market")]
    if all(c in preds for c in outcome_cols("p_indep")):
        rows.append(scoring_report(y, preds[outcome_cols("p_indep")].to_numpy(float),
                                   "stage 1 (market-blind)"))
    rows += [scoring_report(y, preds[outcome_cols("p_model")].to_numpy(float), "model"),
             scoring_report(y, preds[outcome_cols("p_corr")].to_numpy(float), "corrected")]
    if all(c in preds for c in outcome_cols("p_unc")):
        rows.append(scoring_report(y, preds[outcome_cols("p_unc")].to_numpy(float),
                                   "learned uncertainty"))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # --- the headline figure ---------------------------------------------
    # The uncorrected arm must be the SAME estimate the correction was applied
    # to, or the two curves differ for two reasons at once and neither can be
    # attributed.
    raw_prefix = "p_indep" if all(c in preds for c in outcome_cols("p_indep")) else "p_model"
    curves = {}
    for prefix, label in ((raw_prefix, "uncorrected"), ("p_corr", "corrected")):
        c = risk_coverage_curve(preds, prefix)
        curves[label] = c
        print(f"\nRisk-coverage, {label}:  AURC = {aurc(c):+.3f}")
        print(c.round(4).to_string(index=False))

    # --- the gate ---------------------------------------------------------
    print("\nNon-vacuousness check:")
    nv = check_non_vacuousness(preds, "p_corr", raw_prefix)
    print(nv.round(4).to_string(index=False))
    if len(nv) and nv["overlap"].mean() > 0.95:
        print("\n  WARNING: the corrected and uncorrected rules are selecting "
              "almost the same bets.\n  The correction may be a relabelled "
              "threshold rather than a correction.")

    # --- by season --------------------------------------------------------
    print("\nBy season:")
    print(era_breakdown(preds, folds).round(4).to_string(index=False))

    # --- does the effect survive changing the model? ----------------------
    if pool:
        print("\n" + "=" * 70)
        print("MODEL POOL -- is the effect a property of the rule or of LightGBM?")
        print("=" * 70)
        scores, pool_curves, rankings = run_model_pool(df)
        print("\nScores (all market-blind, identical treatment):")
        print(scores.round(4).to_string(index=False))
        print("\nRankings -- note where they disagree:")
        print(rankings.to_string(index=False))

    print("\n" + "=" * 70)
    print("Read the intervals, not the middles. Overlapping intervals mean the")
    print("difference is not established, whatever the point estimates say.")
    print("=" * 70)
    return df, preds, folds


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Selective betting pipeline.")
    ap.add_argument("--mode", default="multiseason",
                    choices=["multiseason", "export", "local", "download"],
                    help="where the data comes from (default: multiseason)")
    ap.add_argument("--source", default=None,
                    help="file or folder for the chosen mode")
    ap.add_argument("--start", type=int, default=2005, help="first year, download mode")
    ap.add_argument("--end", type=int, default=2025, help="last year, download mode")
    ap.add_argument("--pool", action="store_true",
                    help="also run every installed model family and compare")
    args = ap.parse_args()

    defaults = {"multiseason": "matches_multiseason.csv",
                "export": "all-euro-data-2025-2026.csv",
                "local": "data/raw", "download": ""}
    main(source=args.source or defaults[args.mode], mode=args.mode,
         start_year=args.start, end_year=args.end, pool=args.pool)
