"""
Download football-data.co.uk season files through the Wayback Machine.

The site itself is blocked by the local ISP, so every request goes to
web.archive.org with the id_ suffix, which returns the original bytes rather
than an archived page wrapper.

Two things the archive forces on us:

  * It rate-limits hard. A first attempt with four parallel workers got eleven
    files and was then refused at the TCP level for every request afterwards.
    So this fetches one file at a time, waits between requests, and backs off
    for minutes when the refusal comes back.
  * A snapshot taken DURING a season holds a part-played file, which would look
    like a complete season with matches missing. Each file is checked against
    the season's real end date and retried against a later snapshot if short.

Already-downloaded files are re-checked and skipped, so the script can be
stopped and restarted without losing work.
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "fd_raw"
DIVS = ["E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "D1", "D2",
        "SP1", "SP2", "I1", "I2", "F1", "F2", "N1", "B1", "P1", "T1", "G1"]
# season code -> (end year, snapshot timestamps to try, earliest to latest)
SEASONS = {
    "1920": (2020, ["20201101", "20210301", "20211101"]),
    "2021": (2021, ["20211101", "20220301", "20221101"]),
    "2122": (2022, ["20221101", "20230301", "20231101"]),
    "2223": (2023, ["20231101", "20240301", "20241101"]),
    "2324": (2024, ["20241101", "20250301", "20251101"]),
    "2425": (2025, ["20251101", "20260301", "20260901"]),
}
# 2019/20 ran into July 2020 because of the covid shutdown; every other season
# is finished by June.
MIN_END = {"1920": "2020-07-01"}

DELAY = 4.0                      # seconds between requests, politeness
BACKOFF = [90, 240, 600, 900]    # waits after the archive starts refusing us

session = requests.Session()
session.headers["User-Agent"] = "SRP-2109 academic research data collection"


def season_end(df: pd.DataFrame):
    return pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").max()


def get(url: str):
    """One request, waiting out the archive's rate limiting if it blocks us."""
    for wait in [0] + BACKOFF:
        if wait:
            print(f"    blocked; waiting {wait}s", flush=True)
            time.sleep(wait)
        try:
            r = session.get(url, timeout=90)
        except Exception as exc:
            last = str(exc)[:60]
            continue
        if r.status_code == 429:
            last = "http 429"
            continue
        return r, ""
    return None, last


def fetch(season: str, div: str) -> dict:
    end_year, stamps = SEASONS[season]
    need = pd.Timestamp(MIN_END.get(season, f"{end_year}-05-01"))
    dest = OUT / season / f"{div}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():                      # keep only if complete, else refetch
        try:
            d = pd.read_csv(dest, encoding="latin-1")
            if season_end(d) >= need:
                return {"season": season, "div": div, "rows": len(d),
                        "last": str(season_end(d).date()), "status": "cached"}
        except Exception:
            pass
        dest.unlink()

    best, best_max, note = None, None, ""
    for stamp in stamps:
        url = (f"https://web.archive.org/web/{stamp}id_/"
               f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv")
        r, err = get(url)
        time.sleep(DELAY)
        if r is None:
            note = err
            continue
        if r.status_code != 200 or len(r.content) < 500:
            note = f"http {r.status_code}"
            continue
        try:
            d = pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
            end = season_end(d)
        except Exception as exc:
            note = f"parse: {str(exc)[:40]}"
            continue
        if best is None or end > best_max:
            best, best_max = r.content, end
        if end >= need:                                   # season finished
            break
        note = f"ends {end.date()}"

    if best is None:
        return {"season": season, "div": div, "rows": 0, "last": "",
                "status": f"FAILED {note}"}

    dest.write_bytes(best)
    d = pd.read_csv(dest, encoding="latin-1")
    return {"season": season, "div": div, "rows": len(d),
            "last": str(best_max.date()),
            "status": "ok" if best_max >= need else f"PARTIAL ({note})"}


def main() -> None:
    results = []
    for season in SEASONS:
        for div in DIVS:
            r = fetch(season, div)
            results.append(r)
            print(f"{r['season']} {r['div']:<4} {r['rows']:>4} rows  "
                  f"{r['last']:<10} {r['status']}", flush=True)
            pd.DataFrame(results).to_csv(OUT / "_download_log.csv", index=False)

    res = pd.DataFrame(results)
    print("\n" + res.groupby("status")["rows"].agg(["count", "sum"]).to_string())
    bad = res[~res["status"].isin(["ok", "cached"])]
    if len(bad):
        print("\nnot clean:")
        print(bad.to_string(index=False))
    print(f"\ntotal rows: {res['rows'].sum():,} in {len(res)} files")

if __name__ == "__main__":
    main()
