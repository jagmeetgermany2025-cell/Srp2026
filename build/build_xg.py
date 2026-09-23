"""
Turn Understat's per-match xG into features a model may legitimately use.

The xG of the match being predicted is not knowable before kickoff, so it can
never be a feature -- putting it in would be the purest form of leakage. What
is knowable is how a team has been performing: its recent xG created and
conceded, and the gap between the goals it scored and the goals its chances
deserved. That gap is the interesting one. A team scoring well above its xG has
been finishing luckily, and luck does not persist, so the market's view of that
team may be running ahead of its real level.

Every rolling figure is shifted by one match: a team's row for match n is built
from matches 1 to n-1 and nothing else. A team's first matches of a season use
what it did at the end of the season before, where we have it.

Team names are matched to the odds file by result, not by spelling: league,
date and exact score identify a fixture, and enough fixtures identify a club.
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

import pandas as pd

HERE = DATA
LEAGUE_DIV = {"EPL": "E0", "La_Liga": "SP1", "Serie_A": "I1",
              "Bundesliga": "D1", "Ligue_1": "F1"}
# Understat labels a season by the year it starts in; the odds file labels it
# by both years. 2019 and 2020 are listed so the covid seasons load if they
# ever arrive, even though they are only ever used as Elo burn-in.
SEASON_CODE = {"2019": "1920", "2020": "2021", "2021": "2122",
               "2022": "2223", "2023": "2324", "2024": "2425",
               "2025": "2526"}
WINDOWS = (5, 10)


def understat_matches() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(HERE / "understat" / "*.json"))):
        stem = Path(path).stem
        league, year = stem.rsplit("_", 1)
        if league not in LEAGUE_DIV or year not in SEASON_CODE:
            continue
        for m in json.load(open(path))["dates"]:
            if not m["isResult"]:
                continue
            rows.append({"Div": LEAGUE_DIV[league], "Season": SEASON_CODE[year],
                         "us_date": m["datetime"][:10],
                         "us_home": m["h"]["title"], "us_away": m["a"]["title"],
                         "gh": int(m["goals"]["h"]), "ga": int(m["goals"]["a"]),
                         "xg_home": float(m["xG"]["h"]), "xg_away": float(m["xG"]["a"])})
    us = pd.DataFrame(rows)
    us["us_date"] = pd.to_datetime(us["us_date"])
    return us


def link(us: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    """Pair the two sources on league + date + score, allowing a day either way."""
    pairs = []
    for shift in (0, 1, -1):
        pairs.append(ours.assign(key_date=ours.Date + pd.Timedelta(days=shift)).merge(
            us, left_on=["Div", "Season", "key_date", "FTHG", "FTAG"],
            right_on=["Div", "Season", "us_date", "gh", "ga"], how="inner"))
    return (pd.concat(pairs)
              .drop_duplicates(["Div", "Season", "Date", "HomeTeam", "AwayTeam"]))


def team_rows(linked: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, so rolling form is a single groupby."""
    home = linked.rename(columns={"HomeTeam": "team", "AwayTeam": "opponent",
                                  "xg_home": "xg_for", "xg_away": "xg_against",
                                  "FTHG": "goals_for", "FTAG": "goals_against"})
    away = linked.rename(columns={"AwayTeam": "team", "HomeTeam": "opponent",
                                  "xg_away": "xg_for", "xg_home": "xg_against",
                                  "FTAG": "goals_for", "FTHG": "goals_against"})
    cols = ["Div", "Season", "Date", "team", "opponent",
            "xg_for", "xg_against", "goals_for", "goals_against"]
    return pd.concat([home[cols], away[cols]]).sort_values(["team", "Date"])


def rolling_features(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["luck"] = rows["goals_for"] - rows["xg_for"]
    rows["xg_diff"] = rows["xg_for"] - rows["xg_against"]

    out = rows[["Div", "Season", "Date", "team"]].copy()
    grouped = rows.groupby(["team", "Season"], sort=False)
    for col in ["xg_for", "xg_against", "xg_diff", "luck"]:
        for w in WINDOWS:
            shifted = grouped[col].shift(1)
            out[f"{col}_{w}"] = (shifted.groupby([rows["team"], rows["Season"]])
                                 .rolling(w, min_periods=1).mean()
                                 .reset_index(level=[0, 1], drop=True))
    # what a team looked like at the end of last season, for its first games
    last = (rows.groupby(["team", "Season"])[["xg_for", "xg_against", "xg_diff", "luck"]]
              .mean().reset_index())
    order = sorted(rows["Season"].unique())
    nxt = {s: order[i + 1] for i, s in enumerate(order[:-1])}
    last["Season"] = last["Season"].map(nxt)
    last = last.dropna(subset=["Season"]).rename(columns=lambda c:
        f"prev_{c}" if c in ("xg_for", "xg_against", "xg_diff", "luck") else c)
    out = out.merge(last, on=["team", "Season"], how="left")
    for col in ["xg_for", "xg_against", "xg_diff", "luck"]:
        for w in WINDOWS:
            out[f"{col}_{w}"] = out[f"{col}_{w}"].fillna(out[f"prev_{col}"])
    return out


def main() -> None:
    ours = pd.read_csv(HERE / "matches_all.csv", low_memory=False,
                       usecols=["Div", "Date", "Season", "HomeTeam", "AwayTeam",
                                "FTHG", "FTAG"])
    ours["Date"] = pd.to_datetime(ours["Date"], errors="coerce")
    ours["Season"] = ours["Season"].astype(str)
    ours = ours.dropna(subset=["Date", "FTHG", "FTAG"])
    ours = ours[ours["Div"].isin(LEAGUE_DIV.values())]

    us = understat_matches()
    linked = link(us, ours)
    print(f"linked {len(linked):,} of {len(ours):,} matches in the five leagues")
    for season, n in linked.groupby("Season").size().items():
        total = (ours.Season == season).sum()
        print(f"  {season}: {n:,} of {total:,}")

    linked[["Div", "Season", "Date", "HomeTeam", "AwayTeam", "xg_home", "xg_away"]] \
        .to_csv(HERE / "understat" / "xg_by_match.csv", index=False)

    feats = rolling_features(team_rows(linked))
    home = feats.rename(columns={"team": "HomeTeam"}).rename(
        columns=lambda c: f"home_{c}" if c.startswith(("xg_", "luck", "prev_")) else c)
    away = feats.rename(columns={"team": "AwayTeam"}).rename(
        columns=lambda c: f"away_{c}" if c.startswith(("xg_", "luck", "prev_")) else c)
    table = (linked[["Div", "Season", "Date", "HomeTeam", "AwayTeam"]]
             .merge(home, on=["Div", "Season", "Date", "HomeTeam"], how="left")
             .merge(away, on=["Div", "Season", "Date", "AwayTeam"], how="left"))
    dest = HERE / "understat" / "xg_features.csv"
    table.to_csv(dest, index=False)

    filled = table["home_xg_for_5"].notna().mean()
    print(f"\nwrote {dest}: {len(table):,} matches, "
          f"{filled:.0%} with a rolling xG history")


if __name__ == "__main__":
    main()
