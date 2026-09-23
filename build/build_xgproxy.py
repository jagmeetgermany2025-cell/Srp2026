"""
A stand-in for xG, built from shot counts, for the 17 divisions Understat
never covers.

Real xG weighs every shot by where it was taken from. We have no shot
locations, only counts: shots, shots on target, corners. Those counts still
carry most of the signal at match level, because a team that takes twenty
shots and eight on target has created more than one that takes four -- so a
Poisson fit of goals on counts gives a usable expected-goals figure.

How good a stand-in it is, is a measurable question rather than a matter of
opinion, and the five Understat leagues answer it: the same matches have both
numbers, and this script prints the correlation between them.

The leakage trap here is sharp. A match's shot counts are known only after it
is played, so they can never describe the match being predicted. They describe
the matches BEFORE it: every column produced here is a rolling average over a
team's earlier matches, shifted by one. The Poisson mapping itself is fitted
on the earliest season only, so it never sees the seasons it is used on.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = DATA
DEST = HERE / "history" / "xgproxy_features.csv"
WINDOWS = (5, 10)
COUNTS = ["shots", "sot", "corners"]


def long_form(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, with that team's counts and goals."""
    home = pd.DataFrame({
        "Div": df.Div, "Season": df.Season, "Date": df.Date,
        "team": df.HomeTeam, "opponent": df.AwayTeam, "home": 1.0,
        "goals": df.FTHG, "shots": df.HS, "sot": df.HST, "corners": df.HC,
        "goals_against": df.FTAG, "shots_against": df.AS, "sot_against": df.AST})
    away = pd.DataFrame({
        "Div": df.Div, "Season": df.Season, "Date": df.Date,
        "team": df.AwayTeam, "opponent": df.HomeTeam, "home": 0.0,
        "goals": df.FTAG, "shots": df.AS, "sot": df.AST, "corners": df.AC,
        "goals_against": df.FTHG, "shots_against": df.HS, "sot_against": df.HST})
    return pd.concat([home, away], ignore_index=True)


def fit_mapping(train: pd.DataFrame):
    """Poisson goals ~ counts. Fitted once, on the earliest season only."""
    d = train.dropna(subset=COUNTS + ["goals"])
    X = sm.add_constant(np.column_stack([np.log1p(d[c]) for c in COUNTS] +
                                        [d["home"].to_numpy()]))
    return sm.GLM(d["goals"], X, family=sm.families.Poisson()).fit()


def apply_mapping(model, rows: pd.DataFrame) -> np.ndarray:
    ok = rows[COUNTS].notna().all(axis=1)
    out = np.full(len(rows), np.nan)
    if ok.any():
        d = rows[ok]
        X = sm.add_constant(np.column_stack([np.log1p(d[c]) for c in COUNTS] +
                                            [d["home"].to_numpy()]), has_constant="add")
        out[ok.to_numpy()] = model.predict(X)
    return out


def rolling(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.sort_values(["team", "Date"]).copy()
    rows["pxg_luck"] = rows["goals"] - rows["pxg_for"]
    rows["pxg_diff"] = rows["pxg_for"] - rows["pxg_against"]
    out = rows[["Div", "Season", "Date", "team"]].copy()
    by = [rows["team"], rows["Season"]]
    for col in ["pxg_for", "pxg_against", "pxg_diff", "pxg_luck"]:
        shifted = rows.groupby(by, sort=False)[col].shift(1)
        for w in WINDOWS:
            out[f"{col}_{w}"] = (shifted.groupby(by).rolling(w, min_periods=2)
                                 .mean().reset_index(level=[0, 1], drop=True))
    return out


def main() -> None:
    df = pd.read_csv(HERE / "matches_all.csv", low_memory=False,
                     usecols=["Div", "Date", "Season", "HomeTeam", "AwayTeam",
                              "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Season"].astype(str)
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    counts = df["Season"].value_counts()
    df = df[df["Season"].isin(counts[counts >= 1000].index)]

    rows = long_form(df)
    first = sorted(rows["Season"].unique())[0]
    model = fit_mapping(rows[rows.Season == first])
    print(f"Poisson mapping fitted on {first}: "
          + ", ".join(f"{n}={v:+.3f}" for n, v in
                      zip(["const"] + [f"log1p({c})" for c in COUNTS] + ["home"],
                          model.params)))

    rows["pxg_for"] = apply_mapping(model, rows)
    opp = rows.rename(columns={"team": "opponent", "opponent": "team"})[
        ["Div", "Season", "Date", "team", "opponent", "pxg_for"]].rename(
        columns={"pxg_for": "pxg_against"})
    rows = rows.merge(opp, on=["Div", "Season", "Date", "team", "opponent"], how="left")

    feats = rolling(rows)
    home = feats.rename(columns={"team": "HomeTeam"}).rename(
        columns=lambda c: f"home_{c}" if c.startswith("pxg_") else c)
    away = feats.rename(columns={"team": "AwayTeam"}).rename(
        columns=lambda c: f"away_{c}" if c.startswith("pxg_") else c)
    out = (df[["Div", "Season", "Date", "HomeTeam", "AwayTeam"]]
           .merge(home, on=["Div", "Season", "Date", "HomeTeam"], how="left")
           .merge(away, on=["Div", "Season", "Date", "AwayTeam"], how="left"))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DEST, index=False)

    have = out["home_pxg_for_5"].notna() & out["away_pxg_for_5"].notna()
    print(f"\nwrote {DEST}: {len(out):,} matches, "
          f"{have.mean():.0%} with a rolling history, {out.Div.nunique()} divisions")
    print("by division, share with the feature:")
    by_div = out.assign(ok=have).groupby("Div")["ok"].mean().sort_values()
    print("  worst: " + ", ".join(f"{d} {v:.0%}" for d, v in by_div.head(3).items()))

    # how close is the stand-in to the real thing, where both exist?
    real = HERE / "understat" / "xg_by_match.csv"
    if real.exists():
        r = pd.read_csv(real)
        r["Date"] = pd.to_datetime(r["Date"])
        r["Season"] = r["Season"].astype(str)
        per_match = rows[rows.home == 1][["Div", "Season", "Date", "team",
                                          "opponent", "pxg_for", "pxg_against"]]
        per_match = per_match.rename(columns={"team": "HomeTeam", "opponent": "AwayTeam"})
        j = r.merge(per_match, on=["Div", "Season", "Date", "HomeTeam", "AwayTeam"])
        j = j.dropna(subset=["pxg_for", "pxg_against"])
        ch = np.corrcoef(j.xg_home, j.pxg_for)[0, 1]
        ca = np.corrcoef(j.xg_away, j.pxg_against)[0, 1]
        print(f"\nagainst real Understat xG on {len(j):,} matches: "
              f"home r={ch:.2f}, away r={ca:.2f}")
        print(f"  mean real xG {j.xg_home.mean():.2f} vs stand-in {j.pxg_for.mean():.2f}")


if __name__ == "__main__":
    main()
