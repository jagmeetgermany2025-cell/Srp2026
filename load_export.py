"""
load_export.py — read the concatenated football-data export.

The file has three problems:
  1. semicolon-delimited, not comma
  2. 22 division sheets concatenated, each keeping its own header row, and the
     sheets have different column counts (124 / 131 / 132 / 133)
  3. dates are m/d/yy (US order), not the dd/mm/yyyy football-data normally uses

This splits the file into blocks at each embedded header, parses each block with
its own columns, and concatenates on the union of columns.
"""
import csv
import io

import pandas as pd

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


def load_export(path, season="2526", encoding="latin-1"):
    with open(path, encoding=encoding, newline="") as f:
        rows = [r for r in csv.reader(f, delimiter=";") if any(x.strip() for x in r)]

    # split into blocks at every header row
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
        rws = [r[:n] + [""] * (n - len(r)) for r in rws]   # pad or trim to header
        frames.append(pd.DataFrame(rws, columns=hdr))

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[df["HomeTeam"].astype(str).str.strip() != ""]

    # dates are m/d/yy in this export
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    bad = df["Date"].isna().sum()
    if bad:
        alt = pd.to_datetime(df.loc[df["Date"].isna(), "Date"],
                             dayfirst=True, errors="coerce")
        df.loc[df["Date"].isna(), "Date"] = alt

    # this export uses comma as the decimal separator: "1,44" not "1.44"
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


if __name__ == "__main__":
    df = load_export("/mnt/user-data/uploads/all-euro-data-2025-2026.csv")
    print(f"rows        : {len(df):,}")
    print(f"columns     : {df.shape[1]}")
    print(f"divisions   : {df.Div.nunique()}")
    print(f"date range  : {df.Date.min().date()} to {df.Date.max().date()}")
    print(f"unparsed    : {df.Date.isna().sum()}")
    print("\ncoverage of the columns the pipeline needs:")
    for c in ["PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA",
              "B365H", "AvgH", "FTR", "FTHG", "HS"]:
        if c in df:
            print(f"  {c:<7}{df[c].notna().mean()*100:6.1f}%")
    print("\nrows per division:")
    print(df.Div.value_counts().sort_index().to_string())
