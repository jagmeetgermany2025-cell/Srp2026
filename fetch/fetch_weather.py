"""
Attach kickoff-day weather to every match, from Open-Meteo's historical archive.

The archive is free, needs no key, and answers a whole date range per location
in one request, so this is one request per club rather than one per match:
around 480 requests for 31,000 matches.

Daily figures, not hourly. A day of rain at a ground is what shifts a match
towards fewer goals, and daily responses are forty times smaller than hourly
ones, which keeps the whole pull to a few minutes. Hourly can be layered on
later for the matches whose kickoff time is known, if the daily signal proves
worth refining.

Weather is taken at the HOME club's coordinates, which is where the match is
played -- bar the occasional neutral ground, which this ignores.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parents[1] / "data"
CACHE = HERE / "weather" / "by_venue"
DEST = HERE / "weather" / "weather_by_match.csv"
API = "https://archive-api.open-meteo.com/v1/archive"
FIELDS = ["temperature_2m_mean", "precipitation_sum", "snowfall_sum",
          "wind_speed_10m_max"]
DELAY = 3.0            # their minute limit counts data pulled, not calls


def pull(team: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """One club's whole date range, cached. A slow answer is retried, not fatal:
    the first run died on a single read timeout after 300 clubs."""
    dest = CACHE / f"{team.replace('/', '_')}.json"
    if not dest.exists():
        body = None
        for attempt in range(6):
            try:
                r = requests.get(API, timeout=120, params={
                    "latitude": lat, "longitude": lon, "start_date": start,
                    "end_date": end, "daily": ",".join(FIELDS), "timezone": "UTC"})
            except Exception as exc:
                print(f"  {team}: {str(exc)[:60]} (attempt {attempt + 1})", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            time.sleep(DELAY)
            if r.status_code == 200:
                body = r.json()
                break
            hourly = "Hourly" in r.text            # a much longer wall
            print(f"  {team}: http {r.status_code} {r.text[:70]}", flush=True)
            time.sleep(900 if hourly else 45)
        if body is None:
            return None
        dest.write_text(json.dumps(body))
    body = json.loads(dest.read_text())
    daily = body.get("daily")
    if not daily:
        return None
    out = pd.DataFrame(daily).rename(columns={"time": "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    out.insert(0, "HomeTeam", team)
    return out


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    venues = pd.read_csv(HERE / "geo" / "venues.csv").dropna(subset=["lat", "lon"])

    m = pd.read_csv(HERE / "matches_all.csv", low_memory=False,
                    usecols=["Div", "Date", "Season", "HomeTeam", "AwayTeam"])
    m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
    m = m.dropna(subset=["Date"])
    span = m.groupby("HomeTeam")["Date"].agg(["min", "max"])

    frames = []
    todo = [t for t in venues["team"] if t in span.index]
    print(f"{len(todo)} clubs to fetch", flush=True)
    for i, team in enumerate(todo, 1):
        row = venues.loc[venues.team == team].iloc[0]
        lo, hi = span.loc[team, "min"], span.loc[team, "max"]
        got = pull(team, row.lat, row.lon, str(lo.date()), str(hi.date()))
        if got is not None:
            frames.append(got)
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)

    weather = pd.concat(frames, ignore_index=True)
    out = m.merge(weather, on=["HomeTeam", "Date"], how="left")
    out.to_csv(DEST, index=False)

    have = out["temperature_2m_mean"].notna()
    print(f"\nwrote {DEST}")
    print(f"{have.sum():,} of {len(out):,} matches have weather ({have.mean():.0%})")
    print(out.loc[have, ["temperature_2m_mean", "precipitation_sum",
                         "wind_speed_10m_max"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
