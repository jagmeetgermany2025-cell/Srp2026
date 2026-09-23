"""
Collect API-Football injury reports for the 22 divisions in the odds file.

One request returns a whole league-season (the Premier League's 2022/23 comes
back as 3,056 records in a single unpaginated response), so the entire pull is
66 requests and fits inside the free plan's 100 a day.

Two limits of the free plan shape what this can be used for:

  * Only seasons 2022, 2023 and 2024 are available -- that is 2022/23 to
    2024/25. The 2025/26 season, which is currently the test season, is not
    included, so an injury feature can inform training but cannot be tested
    on the most recent season without moving the evaluation window back.
  * 100 requests a day AND 10 a minute. The minute limit is the binding one
    here: a first run at one request a second was throttled after fifteen
    files. Files already downloaded are skipped, so an interrupted run
    continues where it stopped.

Records are stored raw, one JSON file per league-season. Turning them into
per-match availability counts happens later, against the match table.
"""
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parents[1] / "data"
OUT = HERE / "api_football" / "injuries"
KEY = (Path(__file__).resolve().parents[1] / ".api_football_key").read_text().strip()
BASE = "https://v3.football.api-sports.io"

# football-data division code -> API-Football league id
LEAGUES = {
    "E0": 39, "E1": 40, "E2": 41, "E3": 42, "EC": 43,
    "SC0": 179, "SC1": 180, "SC2": 183, "SC3": 184,
    "D1": 78, "D2": 79, "SP1": 140, "SP2": 141,
    "I1": 135, "I2": 136, "F1": 61, "F2": 62,
    "N1": 88, "B1": 144, "P1": 94, "T1": 203, "G1": 197,
}
SEASONS = [2022, 2023, 2024]          # what the free plan allows
DELAY = 7.0            # 10 requests a minute is the plan's ceiling
RATE_WAIT = 65.0       # a throttled request is retried after the minute turns


def pull(div: str, league_id: int, season: int) -> str:
    dest = OUT / f"{div}_{season}.json"
    if dest.exists():
        n = len(json.loads(dest.read_text()).get("response", []))
        return f"cached ({n:,})"

    for attempt in range(4):
        r = requests.get(f"{BASE}/injuries",
                         headers={"x-apisports-key": KEY},
                         params={"league": league_id, "season": season},
                         timeout=60)
        body = r.json() if r.status_code == 200 else {}
        errors = body.get("errors") or {}
        throttled = r.status_code == 429 or "rateLimit" in errors
        if not throttled:
            break
        time.sleep(RATE_WAIT)
    if r.status_code != 200:
        return f"FAILED http {r.status_code}"
    if errors:
        return f"FAILED {errors}"
    dest.write_text(json.dumps(body))
    return f"{len(body.get('response', [])):,} records"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for div, league_id in LEAGUES.items():
        for season in SEASONS:
            status = pull(div, league_id, season)
            print(f"{div:<4} {season}  {status}", flush=True)
            if "cached" not in status:
                time.sleep(DELAY)

if __name__ == "__main__":
    main()
