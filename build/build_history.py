"""
What each team did LAST season, for every division.

Squad value was meant to answer one question: how good is this team, before
this season has told us anything? It answers it for 11 divisions out of 22.
The league table answers the same question for all of them, from data already
on disk, and it answers it in the units that matter -- points and goals rather
than euros.

The interesting cases are the teams that changed division. A promoted side
carries a fine record from a weaker league, and both the model and the market
have to guess how it translates; the tier it came from is the guess's starting
point. So each team gets last season's record, the tier it earned that record
in, and a flag for arriving from somewhere else.

Everything here is last season's, so nothing about the current match leaks.
A team with no previous season in the data gets nulls and an is_new flag,
never a filled-in average.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from two_stage import TIERS          # noqa: E402  (needs the path above)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

HERE = DATA
DEST = HERE / "history" / "prev_season.csv"
STATS = ["prev_ppg", "prev_gd_pg", "prev_gf_pg", "prev_ga_pg",
         "prev_played", "prev_tier", "tier_change", "is_promoted",
         "is_relegated", "is_new"]


def team_season_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per season: what its league campaign looked like."""
    home = pd.DataFrame({
        "Season": df.Season, "Div": df.Div, "team": df.HomeTeam,
        "gf": df.FTHG, "ga": df.FTAG,
        "pts": np.where(df.FTHG > df.FTAG, 3, np.where(df.FTHG == df.FTAG, 1, 0))})
    away = pd.DataFrame({
        "Season": df.Season, "Div": df.Div, "team": df.AwayTeam,
        "gf": df.FTAG, "ga": df.FTHG,
        "pts": np.where(df.FTAG > df.FTHG, 3, np.where(df.FTHG == df.FTAG, 1, 0))})
    both = pd.concat([home, away])
    out = (both.groupby(["Season", "team"])
                .agg(played=("pts", "size"), points=("pts", "sum"),
                     gf=("gf", "sum"), ga=("ga", "sum"),
                     Div=("Div", lambda s: s.mode().iat[0]))
                .reset_index())
    out["tier"] = out["Div"].map(lambda d: TIERS.get(d, ("?", 0))[1])
    return out


def previous(df: pd.DataFrame) -> pd.DataFrame:
    table = team_season_table(df)
    seasons = sorted(table["Season"].unique())
    nxt = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}

    prev = table.copy()
    prev["Season"] = prev["Season"].map(nxt)          # carry it forward one year
    prev = prev.dropna(subset=["Season"])
    prev = prev.assign(
        prev_ppg=prev.points / prev.played,
        prev_gd_pg=(prev.gf - prev.ga) / prev.played,
        prev_gf_pg=prev.gf / prev.played,
        prev_ga_pg=prev.ga / prev.played,
        prev_played=prev.played,
        prev_tier=prev.tier,
    )[["Season", "team"] + [c for c in STATS if c.startswith("prev_")]]

    now = table[["Season", "team", "tier"]]
    out = now.merge(prev, on=["Season", "team"], how="left")
    # a smaller tier number is a better league, so coming up is a positive move
    out["tier_change"] = out["prev_tier"] - out["tier"]
    out["is_promoted"] = (out["tier_change"] > 0).astype(float)
    out["is_relegated"] = (out["tier_change"] < 0).astype(float)
    out["is_new"] = out["prev_ppg"].isna().astype(float)
    out.loc[out.is_new == 1, ["tier_change", "is_promoted", "is_relegated"]] = np.nan
    return out[["Season", "team"] + STATS]


def main() -> None:
    df = pd.read_csv(HERE / "matches_all.csv", low_memory=False,
                     usecols=["Div", "Date", "Season", "HomeTeam", "AwayTeam",
                              "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Season"].astype(str)
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    counts = df["Season"].value_counts()
    df = df[df["Season"].isin(counts[counts >= 1000].index)]

    hist = previous(df)
    home = hist.rename(columns={"team": "HomeTeam", **{c: f"home_{c}" for c in STATS}})
    away = hist.rename(columns={"team": "AwayTeam", **{c: f"away_{c}" for c in STATS}})
    out = (df[["Div", "Season", "Date", "HomeTeam", "AwayTeam"]]
           .merge(home, on=["Season", "HomeTeam"], how="left")
           .merge(away, on=["Season", "AwayTeam"], how="left"))

    DEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DEST, index=False)

    have = out["home_prev_ppg"].notna() & out["away_prev_ppg"].notna()
    print(f"wrote {DEST}: {len(out):,} matches")
    print(f"both teams have last season: {have.sum():,} ({have.mean():.0%})")
    print("\nby season:")
    for s, g in out.groupby("Season"):
        ok = g.home_prev_ppg.notna() & g.away_prev_ppg.notna()
        newcomers = (g.home_is_new == 1).sum() + (g.away_is_new == 1).sum()
        print(f"  {s}  {ok.mean():>4.0%} of {len(g):,} matches, "
              f"{g.Div.nunique()} divisions, {newcomers:,} team-slots with no history")
    print("\npromoted teams, points per game in their new division vs their old:")
    prom = out[out.home_is_promoted == 1]
    print(f"  {len(prom):,} home matches involve a promoted side; "
          f"their previous-tier ppg averaged {prom.home_prev_ppg.mean():.2f}")


if __name__ == "__main__":
    main()
