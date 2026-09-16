"""Baseline numeric features with explicit information-time boundaries."""
from pathlib import Path

import numpy as np
import pandas as pd

from selective import BASE_FEATURE_COLS, MARKET_COLS, TARGET_MAP


def shin_probabilities(odds):
    """Vectorized Shin solver, preserving H/D/A column order.

    For three outcomes, sum(sqrt(z^2+4(1-z)q_i^2/Q)) = 2+z.
    The legacy scripts instead solve against 2-z. Solve sum(p_i)=1
    directly using the rationalized formula, stable even near z=1.
    Non-overround books retain the legacy proportional normalization.
    """
    odds = np.asarray(odds, dtype=float)
    if odds.ndim != 2 or odds.shape[1] != 3 or not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError("Expected finite N-by-3 decimal odds greater than one")
    q = 1 / odds
    booksum = q.sum(axis=1, keepdims=True)
    p = q / booksum
    over = booksum[:, 0] > 1
    a = q[over] ** 2 / booksum[over]
    low, high = np.zeros((len(a), 1)), np.ones((len(a), 1))
    for _ in range(50):
        z = (low + high) / 2
        candidate = 2 * a / (np.sqrt(z * z + 4 * (1 - z) * a) + z)
        above = candidate.sum(axis=1, keepdims=True) > 1
        low, high = np.where(above, z, low), np.where(above, high, z)
    z = (low + high) / 2
    p[over] = 2 * a / (np.sqrt(z * z + 4 * (1 - z) * a) + z)
    return p / p.sum(axis=1, keepdims=True)


def load_matches(path: str | Path, market: str = "prematch") -> pd.DataFrame:
    """Read the repository's ordinary, ISO-date multi-season CSV.

    Closing mode uses closing benchmark AND closing execution prices. No
    fallback between timestamps or providers is made. Missing benchmark rows
    are excluded after histories are computed; they still contribute results
    to future fixtures' histories.
    """
    if market not in {"prematch", "closing"}:
        raise ValueError("market must be prematch or closing")
    df = pd.read_csv(path, dtype={"Season": str}, low_memory=False)
    benchmark = [f"Avg{'C' if market == 'closing' else ''}{o}" for o in "HDA"]
    execution = [f"Max{'C' if market == 'closing' else ''}{o}" for o in "HDA"]
    required = ["Date", "Season", "Div", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = set(required + benchmark + execution) - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    df = df[required + benchmark + execution].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="ISO8601", errors="raise").dt.normalize()
    if df[required].isna().any().any() or not df.FTR.isin(TARGET_MAP).all():
        raise ValueError("Expected completed matches with non-missing identifiers, scores and H/D/A results")
    for c in ["FTHG", "FTAG"] + benchmark + execution:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[["FTHG", "FTAG"]]).all().all() or (df[["FTHG", "FTAG"]] < 0).any().any():
        raise ValueError("Goals must be finite and nonnegative")
    expected = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    if not np.array_equal(expected, df.FTR):
        raise ValueError("FTR disagrees with the full-time scores")
    df = df.sort_values(["Date", "Div", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    df["match_key"] = (df.Date.dt.strftime("%Y-%m-%d") + "|" + df.Div + "|" + df.HomeTeam + "|" + df.AwayTeam)
    if df.match_key.duplicated().any():
        raise ValueError("Duplicate match keys; deduplicate the input explicitly")
    # Country prefix preserves histories across promotion/relegation while
    # keeping identically named clubs in different countries separate.
    df["country_key"] = df.Div.str.replace(r"\d+$", "", regex=True).replace({"EC": "E"})
    sides = []
    for side, team, gf, ga in [("home", "HomeTeam", "FTHG", "FTAG"), ("away", "AwayTeam", "FTAG", "FTHG")]:
        part = df[["Date", "country_key", team, gf, ga]].copy()
        part.columns = ["Date", "country_key", "Team", "GF", "GA"]
        part["side"], part["match_id"] = side, df.index
        sides.append(part)
    long = pd.concat(sides).sort_values(["Date", "match_id"]).reset_index(drop=True)
    # With only a calendar date, two appearances on one day have unknown order.
    if long.duplicated(["country_key", "Team", "Date"]).any():
        raise ValueError("A team appears twice on one day; kickoff timestamps are needed")
    groups = long.groupby(["country_key", "Team"], sort=False)
    long["rest"] = groups.Date.diff().dt.days.fillna(14).clip(0, 14)
    for span in (5, 10):
        for c in ("GF", "GA"):
            long[f"{c}_{span}"] = groups[c].transform(lambda s: s.shift().ewm(span=span, min_periods=3).mean())
        long[f"GD_{span}"] = long[f"GF_{span}"] - long[f"GA_{span}"]
    for side in ("home", "away"):
        stats = long[long.side == side].set_index("match_id")
        for c in ["rest"] + [f"{c}_{s}" for s in (5, 10) for c in ("GF", "GA", "GD")]:
            df[f"{side}_{c}"] = stats[c]
    df["diff_rest_days"] = df.home_rest - df.away_rest
    for c in ("GF", "GA"):
        df[f"diff_roll_{c}_5"] = df[f"home_{c}_5"] - df[f"away_{c}_5"]
    odds = df[benchmark].to_numpy(float)
    valid = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
    p = np.full_like(odds, np.nan)
    p[valid] = shin_probabilities(odds[valid])
    df[MARKET_COLS] = p
    # select_bets consumes MaxH/D/A and AvgH/D/A, so normalize names only
    # after taking copies from the explicitly selected timestamp.
    exec_values = df[execution].to_numpy(float).copy()
    exec_values[~np.isfinite(exec_values) | (exec_values <= 1)] = np.nan
    df[[f"Max{o}" for o in "HDA"]] = exec_values
    df[[f"Avg{o}" for o in "HDA"]] = odds
    df["target"] = df.FTR.map(TARGET_MAP).astype(int)
    df["market_timestamp"] = market
    df.attrs["input_rows"] = len(df)
    # Early-history feature missingness is imputed using training data only.
    out = df.loc[valid].copy()
    out.attrs["excluded_missing_benchmark"] = int((~valid).sum())
    return out


def split_by_date(frame: pd.DataFrame, fraction: float = 0.8):
    """Split on complete calendar days, never randomly or across one date."""
    dates = np.sort(frame.Date.unique())
    if len(dates) < 2 or not 0 < fraction < 1:
        raise ValueError("Chronological splitting needs >=2 dates and 0<fraction<1")
    boundary = dates[min(max(int(len(dates) * fraction), 1), len(dates) - 1)]
    return frame[frame.Date < boundary].copy(), frame[frame.Date >= boundary].copy()


def walk_forward(frame, min_season_matches=1000, test_seasons=None):
    """Earlier seasons -> validation season -> untouched test season.

    Season order follows actual dates. Overlapping season ranges are rejected.
    Small seasons remain usable history but cannot be validation/test seasons.
    """
    summary = frame.groupby("Season").Date.agg(["min", "max", "count"]).sort_values("min")
    seasons = summary.index.tolist()
    for left, right in zip(seasons, seasons[1:]):
        if summary.loc[left, "max"] >= summary.loc[right, "min"]:
            raise ValueError(f"Season date ranges overlap: {left}, {right}")
    requested = set(test_seasons or [])
    if requested - set(seasons):
        raise ValueError(f"Unknown test seasons: {sorted(requested - set(seasons))}")
    emitted = set()
    for i in range(2, len(seasons)):
        season, val_season = seasons[i], seasons[i - 1]
        if requested and season not in requested:
            continue
        if min(summary.loc[season, "count"], summary.loc[val_season, "count"]) < min_season_matches:
            continue
        tr = frame[frame.Season.isin(seasons[:i - 1])].copy()
        va = frame[frame.Season == val_season].copy()
        te = frame[frame.Season == season].copy()
        emitted.add(season)
        yield season, tr, va, te
    if requested - emitted:
        raise ValueError(f"Requested test seasons lack sufficient earlier/validation data: {sorted(requested-emitted)}")
    if not emitted:
        raise ValueError("No eligible folds; need at least three chronological seasons")


def recent_training_rows(frame, limit):
    """Common resource cap for EVERY model; retain entire most recent dates."""
    if limit < 0:
        raise ValueError("Training row limit must be nonnegative")
    if not limit or len(frame) <= limit:
        return frame.copy()
    counts = frame.groupby("Date").size().sort_index(ascending=False).cumsum()
    dates = counts[counts <= limit].index
    if not len(dates):
        raise ValueError("Training row cap is smaller than the most recent match day")
    return frame[frame.Date.isin(dates)].copy()
