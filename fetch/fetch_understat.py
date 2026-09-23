"""
Download Understat season files, in the same shape as the ones already here.

Understat used to embed its data in the league page, which is how the existing
files were made. It does not any more: the page now asks a separate endpoint,
getLeagueData/{league}/{season}, which the site's own league.min.js reveals.
That endpoint returns exactly the teams/players/dates structure the rest of
this project already reads, so nothing downstream has to change.

Fifteen requests in total, several seconds apart. There is no API to be polite
to here beyond an ordinary web server, and this asks it for less than a person
browsing the same pages would.
"""
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parents[1] / "data"
DEST = HERE / "understat"
BASE = "https://understat.com/getLeagueData"
# endpoint slug -> the file name the rest of the project expects
LEAGUES = {"EPL": "EPL", "La_liga": "La_Liga", "Serie_A": "Serie_A",
           "Bundesliga": "Bundesliga", "Ligue_1": "Ligue_1"}
SEASONS = [2019, 2020, 2021]
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0 Safari/537.36",
           "X-Requested-With": "XMLHttpRequest"}
DELAY = 3.0


def pull(slug: str, name: str, season: int) -> str:
    dest = DEST / f"{name}_{season}.json"
    if dest.exists():
        return f"cached ({len(json.loads(dest.read_text())['dates']):,} matches)"
    r = requests.get(f"{BASE}/{slug}/{season}", headers={
        **HEADERS, "Referer": f"https://understat.com/league/{slug}/{season}"},
        timeout=60)
    time.sleep(DELAY)
    if r.status_code != 200:
        return f"FAILED http {r.status_code}"
    try:
        body = r.json()
    except ValueError:
        return "FAILED: not json"
    if "dates" not in body:
        return f"FAILED: unexpected keys {list(body)[:4]}"
    played = sum(1 for m in body["dates"] if m.get("isResult"))
    dest.write_text(json.dumps(body))
    return f"{len(body['dates']):,} fixtures, {played:,} played"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        for slug, name in LEAGUES.items():
            print(f"{name:<12} {season}  {pull(slug, name, season)}", flush=True)


if __name__ == "__main__":
    main()
