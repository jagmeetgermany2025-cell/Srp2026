"""
Who is missing, taken from line-ups rather than from an injury feed.

The injury API's free tier stops at 2024/25, which is exactly the season we
most want. Transfermarkt's line-up file covers every season including the
current one, and it answers a better question anyway: not "is this player
injured" but "is this player in today's squad", which also catches
suspensions, rotation, fallings-out and transfers out.

A club's regulars are defined by what it has actually been doing: the eleven
players with the most starts in its previous ten matches. Today's absence
count is how many of those eleven are not in today's squad -- neither starting
nor on the bench.

Two honest caveats. First, line-ups are published about an hour before kickoff,
so this is knowable before a closing price but NOT before an opening one; a
model that bets at opening prices may not use it. Second, Transfermarkt covers
first divisions only, so this reaches 11 of the 22 divisions.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

import pandas as pd

HERE = DATA
SRC = Path.home() / "Downloads" / "archive"
DEST = HERE / "history" / "lineup_availability.csv"
DIV_TO_TM = {"E0": "GB1", "SC0": "SC1", "D1": "L1", "SP1": "ES1", "I1": "IT1",
             "F1": "FR1", "N1": "NL1", "B1": "BE1", "P1": "PO1", "T1": "TR1",
             "G1": "GR1"}
SQUAD = 11          # how many regulars define a first-choice side
LOOKBACK = 10       # how many recent matches decide who those regulars are
START = "2019-06-01"


def club_match_squads() -> pd.DataFrame:
    games = pd.read_csv(SRC / "games.csv", parse_dates=["date"],
                        usecols=["game_id", "competition_id", "season", "date",
                                 "home_club_id", "away_club_id"])
    games = games[games.competition_id.isin(DIV_TO_TM.values())
                  & (games.date >= START)]
    wanted = set(games.game_id)
    print(f"{len(games):,} games in the covered leagues since {START}", flush=True)

    keep = []
    reader = pd.read_csv(SRC / "game_lineups.csv", chunksize=500_000,
                         usecols=["game_id", "club_id", "player_id", "type"])
    for chunk in reader:
        keep.append(chunk[chunk.game_id.isin(wanted)])
    lineups = pd.concat(keep, ignore_index=True)
    print(f"{len(lineups):,} line-up rows for those games", flush=True)

    dates = games.set_index("game_id")["date"]
    lineups["date"] = lineups["game_id"].map(dates)
    return lineups, games


def availability(lineups: pd.DataFrame) -> pd.DataFrame:
    """For each club and match: how many of its recent regulars are absent."""
    starts = lineups[lineups["type"] == "starting_lineup"]
    squads = (lineups.groupby(["club_id", "game_id", "date"])["player_id"]
              .apply(set).reset_index(name="squad"))
    start_sets = (starts.groupby(["club_id", "game_id"])["player_id"]
                  .apply(list).to_dict())

    rows = []
    for club, group in squads.sort_values("date").groupby("club_id", sort=False):
        history: list[list] = []
        for game_id, date, squad in zip(group.game_id, group.date, group.squad):
            if len(history) >= 3:
                recent = [p for game in history[-LOOKBACK:] for p in game]
                regulars = pd.Series(recent).value_counts().head(SQUAD).index
                missing = sum(p not in squad for p in regulars)
                rows.append({"club_id": club, "game_id": game_id, "date": date,
                             "missing_regulars": float(missing),
                             "available_regulars": float(SQUAD - missing)})
            history.append(start_sets.get((club, game_id), []))
    return pd.DataFrame(rows)


def main() -> None:
    lineups, games = club_match_squads()
    avail = availability(lineups)
    print(f"{len(avail):,} club-match availability rows", flush=True)

    m = pd.read_csv(HERE / "matches_all.csv", low_memory=False,
                    usecols=["Div", "Date", "Season", "HomeTeam", "AwayTeam"])
    m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
    m["Season"] = m["Season"].astype(str)
    m = m[m.Div.isin(DIV_TO_TM)].dropna(subset=["Date"])

    club_map = pd.read_csv(HERE / "transfermarkt" / "club_name_map.csv")
    club_map = club_map[["Div", "fd_name", "club_id"]]

    side = avail.rename(columns={"date": "Date"})
    out = m.copy()
    for prefix, name_col in (("home", "HomeTeam"), ("away", "AwayTeam")):
        key = club_map.rename(columns={"fd_name": name_col})
        joined = (out[["Div", "Date", name_col]]
                  .merge(key, on=["Div", name_col], how="left")
                  .merge(side, on=["club_id", "Date"], how="left"))
        out[f"{prefix}_missing_regulars"] = joined["missing_regulars"].to_numpy()
        out[f"{prefix}_available_regulars"] = joined["available_regulars"].to_numpy()

    DEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DEST, index=False)

    have = out.home_missing_regulars.notna() & out.away_missing_regulars.notna()
    print(f"\nwrote {DEST}: {len(out):,} matches in {out.Div.nunique()} divisions")
    print(f"both sides measured: {have.sum():,} ({have.mean():.0%})")
    print("\nby season:")
    for s, g in out.groupby("Season"):
        ok = g.home_missing_regulars.notna() & g.away_missing_regulars.notna()
        if ok.sum():
            print(f"  {s}  {ok.mean():>4.0%} of {len(g):,}   "
                  f"mean regulars missing {g.loc[ok, 'home_missing_regulars'].mean():.1f}")


if __name__ == "__main__":
    main()
