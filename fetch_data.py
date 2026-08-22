"""Download football-data.co.uk match+odds CSVs across seasons and divisions,
cache them locally, and assemble a multi-season frame.

Adapts data_sources.pdf section 1 to this project: 22 divisions instead of
E0-E3, a configurable season range, and the reading hardened against what the
archive actually contains rather than what it is documented to contain.

Two-step by design. fetch_seasons() downloads and caches; load_matches()
assembles whatever is cached. Re-assembling after a partial fetch, or from a
cache built earlier, needs no network access.

parse_dates() here is the multi-season one and records which format won per
file so phase_a.py can audit it. load_data._parse_dates stays as it is for the
single-season path EDA.py uses.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import time
from collections import Counter

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# The same 22 divisions as EDA.py's TIERS dict -- keep the two lists in sync.
DIVISIONS = [
    "E0", "E1", "E2", "E3", "EC",          # England
    "SC0", "SC1", "SC2", "SC3",            # Scotland
    "D1", "D2",                            # Germany
    "SP1", "SP2",                          # Spain
    "I1", "I2",                            # Italy
    "F1", "F2",                            # France
    "N1", "B1", "P1", "T1", "G1",          # NL, BE, PT, TR, GR (top flight)
]

FIRST_SEASON_YEAR = 2005   # older seasons are thin on odds coverage anyway

# Pinnacle's feed has been unreliable from this date. Confirmed in EDA.py
# section 6: PSH exceeds the recorded market maximum in ~25% of matches from
# here on, and coverage falls to 39.3%. Masked by date, not by season, so
# earlier seasons keep the sharpest book in the file.
PINNACLE_STALE_FROM = "2025-07-23"

CACHE_DIR = "fd_cache"
OUT_PARQUET = "matches_multiseason.parquet"

# football-data renamed these clubs mid-corpus. Left alone, an Elo keyed on
# (division, team) starts the "new" club at 1500 partway through. Only
# confirmed renames of the SAME club belong here -- phase_a.py reports
# candidates, but similar names can be different clubs (Reggina/Reggiana).
TEAM_CANONICAL = {
    "AFC Telford United": "Telford United",
    "Atl. Madrid": "Ath Madrid",
}

# football-data spelling -> ClubElo spelling. Used only for the join; the
# match data keeps football-data's own names.
CLUBELO_ALIASES = {
    "Ankaragucu": "Ankaraguecue",
    "Extremadura UD": "Extremadura",
    "FeralpiSalo": "Feralpisalo",
    "Inverness C": "Inverness",
    "Kallithea": "Kalithea",
    "Kifisia": "Kifisias",
    "Sheffield Wed": "Sheffield Weds",
}

# Reviewed and rejected. Without these the same false positives surface on
# every run, and a report you learn to ignore is worse than no report.
CLUBELO_NOT_A_MATCH = {
    "Northwich",                      # Northwich Victoria is not Norwich City
}
NOT_RENAMES = {
    ("Reggina", "Reggiana"),          # different cities; both played 2020-2023
}

# columns that are legitimately text and must never be coerced to numeric
NON_NUMERIC = {"Div", "Date", "Season", "Time", "HomeTeam", "AwayTeam",
               "FTR", "HTR", "Referee", "Country"}

# default python-requests UA gets blocked by some hosts
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-data-fetch/1.0)"}


# --------------------------------------------------------------- seasons ----

def season_code(start_year):
    """football-data.co.uk's season code: 2025 -> '2526', 1999 -> '9900'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def default_seasons(first_year=FIRST_SEASON_YEAR, today=None):
    """Every season code from `first_year` up to the current season.

    A season starting in year Y runs from August of Y to May of Y+1, so before
    July the most recently *started* season is still last year's.
    """
    today = today or datetime.date.today()
    last_year = today.year if today.month >= 7 else today.year - 1
    return [season_code(y) for y in range(first_year, last_year + 1)]


def season_start_year(code):
    """'0506' -> 2005, '9900' -> 1999."""
    yy = int(code[:2])
    return 1900 + yy if yy >= 90 else 2000 + yy


# ------------------------------------------------------------ date parsing ----

_DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d")

# which format won for each (season, div) -- audited in phase_a.py
DATE_FORMAT_LOG = {}


def parse_dates(s, key=None):
    """Parse a date column by finding the one format that fits the whole column.

    Export eras differ (day-first vs month-first), so the format is fitted per
    file, before concatenating. One format picked for a mixed multi-season
    column mis-parses whichever era loses the vote, and a day/month swap is
    invisible downstream.
    """
    s = s.astype(str).str.strip()
    valid = s.ne("") & s.ne("nan")
    best, best_hits = None, -1
    for fmt in _DATE_FORMATS:
        hits = pd.to_datetime(s.where(valid), format=fmt, errors="coerce").notna().sum()
        if hits > best_hits:
            best, best_hits = fmt, hits
        if hits == valid.sum():
            break
    if key is not None:
        DATE_FORMAT_LOG[key] = (best, int(best_hits), int(valid.sum()))
    return pd.to_datetime(s.where(valid), format=best, errors="coerce")


# ----------------------------------------------------------------- fetch ----

def _cache_path(cache_dir, season, div):
    return os.path.join(cache_dir, season, f"{div}.csv")


def fetch_one(session, season, div, cache_dir, timeout=15):
    """Download one (season, division) CSV into the cache, or skip if cached.

    Returns a status string rather than a bool: a 404 and a dead network need
    different handling and shouldn't collapse into "skipped".
    """
    path = _cache_path(cache_dir, season, div)
    if os.path.exists(path):
        return "cached"

    try:
        r = session.get(f"{BASE_URL}/{season}/{div}.csv", headers=HEADERS, timeout=timeout)
    except requests.Timeout:
        return "timeout"
    except requests.RequestException as e:
        return f"neterr:{type(e).__name__}"

    if r.status_code != 200:
        return f"http{r.status_code}"
    if len(r.content) < 500:
        return "tiny"
    # block pages come back 200 with plenty of bytes -- don't cache them as csv
    if b"HomeTeam" not in r.content[:1000]:
        return "notcsv"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(r.content)
    return "ok"


def fetch_seasons(seasons=None, divisions=None, cache_dir=CACHE_DIR,
                  polite_delay=0.15, abort_after=8, verbose=True):
    """Download every (season, division) CSV not already cached.

    Returns a Counter of statuses. Combinations that 404 are expected: most
    divisions did not exist in the earliest seasons requested, and the current
    season is published a division at a time.
    """
    seasons = seasons or default_seasons()
    divisions = divisions or DIVISIONS
    session = requests.Session()
    tally, consecutive_net_fail = Counter(), 0

    for season in seasons:
        per = Counter()
        for div in divisions:
            status = fetch_one(session, season, div, cache_dir)
            per[status] += 1
            tally[status] += 1

            # bail early rather than burn 20 minutes of timeouts
            if status == "timeout" or status.startswith("neterr"):
                consecutive_net_fail += 1
                if consecutive_net_fail >= abort_after:
                    raise RuntimeError(
                        f"{consecutive_net_fail} consecutive network failures — aborting"
                    )
            else:
                consecutive_net_fail = 0

            if status == "ok":
                time.sleep(polite_delay)

        if verbose:
            print(f"{season}: {dict(per)}", flush=True)

    return tally


# -------------------------------------------------------------- assembly ----

def _read_any_encoding(path, **kw):
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False, **kw)


def _scan_fields(path):
    """Header row and the widest row in the file, in fields."""
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=enc, newline="") as fh:
                rows = csv.reader(fh)
                header = next(rows)
                return header, max([len(header)] + [len(r) for r in rows if r])
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def _read_cached(path, season, div, verbose=True):
    try:
        df = _read_any_encoding(path)
    except pd.errors.ParserError:
        # football-data appends columns without extending the header row, so
        # later rows carry more fields than the header declares. Pad to the
        # widest row; skipping them drops real matches. header=None +
        # skiprows=1 so the field count comes from `names`, not the short
        # header line.
        header, widest = _scan_fields(path)
        names = header + [f"Extra{i}" for i in range(len(header), widest)]
        df = _read_any_encoding(path, names=names, header=None, skiprows=1)
        if verbose:
            print(f"  {season}/{div}: ragged file, header padded "
                  f"{len(header)} -> {widest}")

    # old files pad team names with spaces; "Ajax " and "Ajax" would be two
    # different clubs to both our Elo and the ClubElo join
    for c in ("HomeTeam", "AwayTeam"):
        if c in df.columns:
            df[c] = df[c].str.strip()
    if "HomeTeam" in df.columns:
        df = df.dropna(subset=["HomeTeam"])
    df["Season"], df["Div"] = season, div
    if "Date" in df.columns:
        df["Date"] = parse_dates(df["Date"], key=(season, div))
        df = df.dropna(subset=["Date"])
    return df


def coerce_numeric(df, report=True):
    """Fix columns left as object dtype by a stray token.

    A single junk token (`#REF!`, a stray space) makes pandas read that whole
    file's column as strings. Concatenated against the same column read as
    float elsewhere it becomes a mixed object column: parquet rejects it, and
    comparisons against it are silently wrong.
    """
    fixed = []
    for c in df.columns:
        if c in NON_NUMERIC or df[c].dtype != object:
            continue
        nonnull = int(df[c].notna().sum())
        if not nonnull:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        got = int(conv.notna().sum())
        # only adopt if the column really is numeric; a text column converts
        # almost entirely to NaN
        if got >= 0.9 * nonnull:
            fixed.append((c, nonnull, nonnull - got))
            df[c] = conv

    if report and fixed:
        print(f"coerced {len(fixed)} object columns to numeric:")
        for c, n, lost in fixed:
            note = f"  ({lost} unparseable -> NaN)" if lost else ""
            print(f"  {c:<12} {n:>8,} values{note}")
    return df


def mixed_type_columns(df, sample=20000):
    """Columns still holding more than one Python type.

    Anything still mixed dies in to_parquet with an unreadable pyarrow error;
    name the column here instead.
    """
    bad = []
    for c in df.columns:
        if c in NON_NUMERIC or df[c].dtype != object:
            continue
        kinds = {type(v).__name__ for v in df[c].dropna().head(sample)}
        if len(kinds) > 1:
            bad.append((c, sorted(kinds)))
    return bad


def load_matches(seasons=None, divisions=None, cache_dir=CACHE_DIR, verbose=True):
    """Assemble every cached (season, division) CSV into one frame.

    Fetches nothing itself -- run fetch_seasons() first.
    """
    seasons = seasons or default_seasons()
    divisions = divisions or DIVISIONS

    frames = []
    for season in seasons:
        for div in divisions:
            path = _cache_path(cache_dir, season, div)
            if os.path.exists(path):
                frames.append(_read_cached(path, season, div, verbose=verbose))

    if not frames:
        raise FileNotFoundError(
            f"nothing cached in {cache_dir!r} -- run fetch_seasons() first"
        )

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = coerce_numeric(df, report=verbose)
    for c in ("HomeTeam", "AwayTeam"):
        df[c] = df[c].replace(TEAM_CANONICAL)
    return df.sort_values("Date").reset_index(drop=True)


# ------------------------------------------------------- Pinnacle masking ----

# PS* (1X2), PSC* (closing), P>/P< (O/U), PAH* (handicap) -- no other
# bookmaker code in this schema starts with "P".
_PINNACLE_PREFIXES = ("PS", "PC", "P>", "P<", "PAH")


def mask_stale_pinnacle(df, cutoff=PINNACLE_STALE_FROM):
    """Blank Pinnacle columns from `cutoff` onward, in place.

    Not a season-wide drop: Pinnacle is the sharpest book in the file before
    the feed broke, and the standard reference price in this literature.
    """
    pinnacle_cols = [c for c in df.columns if c.startswith(_PINNACLE_PREFIXES)]
    if not pinnacle_cols:
        return df
    mask = df["Date"] >= pd.Timestamp(cutoff)
    df.loc[mask, pinnacle_cols] = pd.NA
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--first-season-year", type=int, default=FIRST_SEASON_YEAR)
    ap.add_argument("--out", default=OUT_PARQUET)
    args = ap.parse_args()

    seasons = default_seasons(args.first_season_year)
    print(f"Fetching {len(seasons)} seasons x {len(DIVISIONS)} divisions ...")
    tally = fetch_seasons(seasons, DIVISIONS, args.cache_dir)
    usable = tally["ok"] + tally["cached"]
    absent = sum(v for k, v in tally.items() if k.startswith("http"))
    print(f"\n{usable} files usable, {absent} not published yet")

    df = mask_stale_pinnacle(load_matches(seasons, DIVISIONS, args.cache_dir))
    print(f"Assembled {len(df):,} matches, "
          f"{df.Date.min().date()} to {df.Date.max().date()}")
    df.to_parquet(args.out, index=False)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
