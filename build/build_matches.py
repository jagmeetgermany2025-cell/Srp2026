"""
One match table for every season we hold.

Two sources, two shapes. The four recent seasons arrive as a single clean CSV
with a Season column. The three earlier ones arrive as football-data's Excel
workbooks, one sheet per division and no Season column at all -- so the season
is taken from the file name and stamped on every row.

Columns differ across seasons because bookmakers come and go: Pinnacle is in
the older files and gone from 2025/26, Betfair Exchange is the other way
round. The union is kept and the gaps left as nulls, because a missing column
is a fact about that season, and filling it would hide exactly the kind of
coverage break the load checks exist to catch.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

import pandas as pd

HERE = DATA
DEST = HERE / "matches_all.csv"
WORKBOOKS = {"all-euro-data-2019-2020.xlsx": "1920",
             "all-euro-data-2020-2021.xlsx": "2021",
             "all-euro-data-2021-2022.xlsx": "2122"}


def from_workbook(path: Path, season: str) -> pd.DataFrame:
    book = pd.ExcelFile(path)
    frames = []
    for sheet in book.sheet_names:
        d = book.parse(sheet)
        if "HomeTeam" not in d.columns or not len(d):
            continue
        d = d[d["HomeTeam"].notna()].copy()
        d["Div"] = d["Div"].fillna(sheet) if "Div" in d.columns else sheet
        frames.append(d)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["Season"] = season
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    return out


def main() -> None:
    recent = pd.read_csv(HERE / "matches_multiseason.csv", low_memory=False)
    recent["Date"] = pd.to_datetime(recent["Date"], errors="coerce")
    recent["Season"] = recent["Season"].astype(str)
    print(f"recent file: {len(recent):,} rows, seasons "
          f"{', '.join(sorted(recent.Season.unique()))}")

    older = []
    for name, season in WORKBOOKS.items():
        path = HERE / name
        if not path.exists():
            print(f"  {name}: not here, skipped")
            continue
        d = from_workbook(path, season)
        print(f"  {name}: {len(d):,} matches, {d.Div.nunique()} divisions, "
              f"{d.Date.min().date()} to {d.Date.max().date()}")
        older.append(d)

    everything = pd.concat(older + [recent], ignore_index=True, sort=False)
    everything = everything.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    everything = everything.sort_values("Date").reset_index(drop=True)
    everything.to_csv(DEST, index=False)

    print(f"\nwrote {DEST}: {len(everything):,} matches, {everything.shape[1]} columns")
    summary = everything.groupby("Season").agg(
        matches=("Date", "size"), divisions=("Div", "nunique"),
        first=("Date", "min"), last=("Date", "max"))
    for col in ["AvgCH", "MaxCH", "PSCH", "BFECH", "HST", "AHCh"]:
        if col in everything.columns:
            summary[col] = (everything.groupby("Season")[col]
                            .apply(lambda s: s.notna().mean() * 100).round(0))
    print(summary.to_string())


if __name__ == "__main__":
    main()
