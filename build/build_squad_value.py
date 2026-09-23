"""
Squad market value per match, for the divisions Transfermarkt covers.

Two steps, and the first is the one that needs care. Transfermarkt names clubs
one way ("Atletico de Madrid") and football-data another ("Ath Madrid"), and
matching those by spelling produces confident nonsense -- an early attempt
mapped Atletico to Real Madrid and PSG to Paris FC. So clubs are matched by
what they DID: league, date and exact score identify a fixture, hundreds of
fixtures identify a club, and two clubs cannot share a season's worth of
results.

The value itself is the last monthly valuation at or before kickoff, summed
over the players registered to that club, so nothing from after the match can
reach the row.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

import numpy as np
import pandas as pd

HERE = DATA
SRC = Path.home() / "Downloads" / "archive"
MAP_DEST = HERE / "transfermarkt" / "club_name_map.csv"
DEST = HERE / "transfermarkt" / "squad_value_by_match.csv"
DIV_TO_TM = {"E0": "GB1", "SC0": "SC1", "D1": "L1", "SP1": "ES1", "I1": "IT1",
             "F1": "FR1", "N1": "NL1", "B1": "BE1", "P1": "PO1", "T1": "TR1",
             "G1": "GR1"}


def club_map(ours: pd.DataFrame) -> pd.DataFrame:
    tm = pd.read_csv(SRC / "games.csv", parse_dates=["date"],
                     usecols=["competition_id", "date", "home_club_id",
                              "away_club_id", "home_club_goals", "away_club_goals"])
    tm = tm[tm.competition_id.isin(DIV_TO_TM.values())]
    j = ours.merge(tm, left_on=["competition_id", "Date", "FTHG", "FTAG"],
                   right_on=["competition_id", "date", "home_club_goals",
                             "away_club_goals"], how="inner")
    votes = pd.concat([
        j[["Div", "HomeTeam", "home_club_id"]].rename(
            columns={"HomeTeam": "fd_name", "home_club_id": "club_id"}),
        j[["Div", "AwayTeam", "away_club_id"]].rename(
            columns={"AwayTeam": "fd_name", "away_club_id": "club_id"})
    ]).value_counts().reset_index(name="n")
    best = votes.sort_values("n", ascending=False).drop_duplicates(["Div", "fd_name"])
    total = votes.groupby(["Div", "fd_name"])["n"].sum().rename("total")
    best = best.merge(total, on=["Div", "fd_name"])
    best["share"] = best.n / best.total
    names = pd.read_csv(SRC / "clubs.csv")[["club_id", "name"]]
    return best.merge(names, on="club_id", how="left")


def monthly_values() -> pd.DataFrame:
    v = pd.read_csv(SRC / "player_valuations.csv", parse_dates=["date"])
    v = v[v["date"] >= "2019-06-01"]
    v["month"] = v["date"].dt.to_period("M").dt.to_timestamp()
    v = v.sort_values("date").drop_duplicates(["player_id", "month"], keep="last")
    g = (v.groupby(["current_club_id", "month"])
           .agg(squad_value_eur=("market_value_in_eur", "sum"),
                n_valued=("player_id", "size")).reset_index()
           .rename(columns={"current_club_id": "club_id"}))
    g["club_id"] = g["club_id"].astype(float)
    g["month"] = g["month"].astype("datetime64[ns]")
    return g.sort_values("month")


def main() -> None:
    raw = pd.read_csv(HERE / "matches_all.csv", low_memory=False,
                      usecols=["Div", "Date", "Season", "HomeTeam", "AwayTeam",
                               "FTHG", "FTAG"])
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Season"] = raw["Season"].astype(str)
    raw = raw.dropna(subset=["Date"]).reset_index(drop=True)
    covered = raw[raw.Div.isin(DIV_TO_TM)].assign(
        competition_id=lambda d: d.Div.map(DIV_TO_TM)).dropna(subset=["FTHG", "FTAG"])

    mapping = club_map(covered)
    mapping.to_csv(MAP_DEST, index=False)
    print(f"mapped {len(mapping)} clubs by result, median confidence "
          f"{mapping.share.median():.2f}, "
          f"{(mapping.share < 0.6).sum()} below 0.6")

    vals = monthly_values()
    m = raw.copy()
    m["match_id"] = np.arange(len(m))
    m["month"] = m["Date"].dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")
    key = mapping[["Div", "fd_name", "club_id"]]

    for side in ("HomeTeam", "AwayTeam"):
        k = key.rename(columns={"fd_name": side})
        f = m[["match_id", "Div", side, "month"]].merge(k, on=["Div", side], how="left")
        f = f.dropna(subset=["club_id"]).sort_values("month")
        got = pd.merge_asof(f, vals[["club_id", "month", "squad_value_eur"]],
                            on="month", by="club_id", direction="backward")
        m[f"{'home' if side == 'HomeTeam' else 'away'}_value"] = \
            m["match_id"].map(got.set_index("match_id")["squad_value_eur"])

    both = m.home_value.notna() & m.away_value.notna()
    m["value_log_ratio"] = np.where(both, np.log(m.home_value / m.away_value), np.nan)
    med = m[both].groupby(["Div", "Season"])["home_value"].median().rename("league_median")
    m = m.merge(med, on=["Div", "Season"], how="left")
    m["home_value_rel"] = m.home_value / m.league_median
    m["away_value_rel"] = m.away_value / m.league_median

    cols = ["Div", "Season", "Date", "HomeTeam", "AwayTeam", "home_value",
            "away_value", "value_log_ratio", "home_value_rel", "away_value_rel"]
    m[cols].to_csv(DEST, index=False)
    print(f"\nwrote {DEST}: {both.sum():,} of {len(m):,} matches valued")
    print(m.assign(ok=both).groupby("Season")["ok"]
           .agg(matches="size", valued="sum",
                share=lambda s: f"{s.mean():.0%}").to_string())


if __name__ == "__main__":
    main()
