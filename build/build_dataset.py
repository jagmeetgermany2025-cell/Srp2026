"""
Assemble the modelling table: every match, every feature, nothing filled in.

Three rules decide the shape of this file.

Missing stays missing. A feature that does not reach a division is null there,
never zero and never an imputed average, because zero reads as "an average
team" to a model and that is a fabricated fact. Each block also carries a
has_<block> flag, so a model can use the absence itself -- which is real
information, since the blocks are missing by division, not by accident.

Division and tier are explicit columns. Without them a model would infer
"lower division" from the pattern of missing features, and that inference
would be invisible. Better to hand it over plainly.

Seasons are labelled, not filtered. 2019/20 pays for Elo's first fit and
2020/21 was played in empty stadiums, so both are marked burn_in rather than
deleted: a robustness check that includes them costs one filter, and the
covid seasons stay out of the headline by default.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

import argparse
from pathlib import Path

import pandas as pd

from two_stage import TIERS

HERE = DATA
KEY = ["Div", "Date", "HomeTeam", "AwayTeam"]
BURN_IN = {"1920", "2021"}          # covid: curtailed, then no crowds
MIN_SEASON_MATCHES = 1000           # drops the season in progress

BLOCKS = {
    "elo":          ("elo/elo_by_match.csv", "elo_diff"),
    "last_season":  ("history/prev_season.csv", "home_prev_ppg"),
    "xg_proxy":     ("history/xgproxy_features.csv", "home_pxg_for_5"),
    "xg":           ("understat/xg_features.csv", "home_xg_for_5"),
    "squad_value":  ("transfermarkt/squad_value_by_match.csv", "value_log_ratio"),
    "availability": ("history/lineup_availability.csv", "home_missing_regulars"),
    "weather":      ("weather/weather_by_match.csv", "temperature_2m_mean"),
}


# Left over from an earlier pipeline: flags with nothing behind them, present
# only in the seasons that came from the merged file. The names also collide
# with the has_<block> flags this script writes.
LEGACY = ["has_closing", "has_clubelo"]


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Drop what carries no information: spreadsheet debris, dead columns."""
    junk = [c for c in df.columns if str(c).startswith("Unnamed")]
    legacy = [c for c in LEGACY if c in df.columns]
    empty = [c for c in df.columns
             if c not in junk + legacy and df[c].notna().mean() < 0.01]
    dropped = junk + legacy + empty
    if dropped:
        print(f"dropped {len(dropped)} columns carrying almost nothing: "
              f"{len(junk)} spreadsheet artifacts, {len(legacy)} legacy flags, "
              f"{len(empty)} below 1% coverage")
        print(f"  under 1%: {', '.join(sorted(empty))}")
    return df.drop(columns=dropped)


def read_block(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    drop = [c for c in ("Season", "match_id", "month") if c in df.columns]
    return df.drop(columns=drop).drop_duplicates(KEY)


def main(source: str, dest: str, dictionary: str) -> None:
    df = pd.read_csv(HERE / source, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Season"].astype(str)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    counts = df["Season"].value_counts()
    part = sorted(counts[counts < MIN_SEASON_MATCHES].index)
    if part:
        print(f"dropping part-played season(s): {', '.join(part)}")
        df = df[~df["Season"].isin(part)]

    df = tidy(df)
    df["Country"] = df["Div"].map(lambda d: TIERS.get(d, ("?", 0))[0])
    df["Tier"] = df["Div"].map(lambda d: TIERS.get(d, ("?", 0))[1])
    df["window"] = df["Season"].map(lambda s: "burn_in" if s in BURN_IN else "evidence")
    df = df.sort_values("Date").reset_index(drop=True)
    provenance = {c: "match file" for c in df.columns}

    print(f"\n{len(df):,} matches, {df['Div'].nunique()} divisions, "
          f"{df['Season'].nunique()} seasons\n")
    for name, (rel, marker) in BLOCKS.items():
        block = read_block(HERE / rel)
        if block is None:
            print(f"  {name:<13} not built yet -- column omitted")
            continue
        before = set(df.columns)
        df = df.merge(block, on=KEY, how="left")
        added = [c for c in df.columns if c not in before]
        df[f"has_{name}"] = df[marker].notna().astype(int)
        provenance.update({c: name for c in added + [f"has_{name}"]})
        ev = df["window"] == "evidence"
        print(f"  {name:<13} {df.loc[ev, marker].notna().mean():>4.0%} of evidence "
              f"matches, {df.loc[df[marker].notna(), 'Div'].nunique():>2} divisions, "
              f"{len(added)} columns")

    df.to_csv(HERE / dest, index=False)

    rows = []
    ev = df["window"] == "evidence"
    for col in df.columns:
        rows.append({"column": col, "block": provenance.get(col, "derived"),
                     "dtype": str(df[col].dtype),
                     "coverage_all": round(df[col].notna().mean() * 100, 1),
                     "coverage_evidence": round(df.loc[ev, col].notna().mean() * 100, 1)})
    pd.DataFrame(rows).to_csv(HERE / dictionary, index=False)

    print(f"\nwrote {dest}: {len(df):,} rows x {df.shape[1]} columns")
    print(f"wrote {dictionary}: one row per column, with coverage")
    print("\nby window:")
    print(df.groupby(["window", "Season"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="matches_all.csv")
    ap.add_argument("--out", default="dataset_all.csv")
    ap.add_argument("--dictionary", default="dictionary.csv")
    a = ap.parse_args()
    main(a.source, a.out, a.dictionary)
