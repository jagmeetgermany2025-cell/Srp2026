"""
Find a coordinate for every club, so that weather can be attached to a match.

Kickoff weather needs a place. The odds file names teams, not grounds, so each
club is looked up once in OpenStreetMap: first as a stadium, then as a football
club, then as a bare name, each search confined to the club's own country. The
first answer that lands inside that country's bounding box is kept.

City-level accuracy is enough here. Rain and wind at a ground are what matter,
and a few kilometres of error does not change them.

One club, one request, one a second, and every answer cached -- so a stopped
run resumes without asking the same question twice.
"""
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parents[1] / "data"
DEST = HERE / "geo" / "venues.csv"
AGENT = "SRP-2109-academic-research/1.0"
DELAY = 1.1                     # OpenStreetMap asks for no more than one a second

# football-data writes short names that OpenStreetMap has never heard of.
# Everything here was in the not-found list of the first pass.
ALIAS = {
    "Sheffield Weds": "Sheffield Wednesday", "Nott'm Forest": "Nottingham Forest",
    "Bristol Rvs": "Bristol Rovers", "Albion Rvs": "Albion Rovers",
    "Raith Rvs": "Raith Rovers", "Dag and Red": "Dagenham and Redbridge",
    "Peterboro": "Peterborough United", "Boston Utd": "Boston United",
    "Airdrie Utd": "Airdrieonians", "Ath Bilbao": "Athletic Bilbao",
    "Ath Madrid": "Atletico Madrid", "Sp Gijon": "Sporting Gijon",
    "Sp Braga": "Sporting Braga", "Sp Lisbon": "Sporting Lisbon",
    "Sociedad B": "Real Sociedad",     "For Sittard": "Fortuna Sittard", "Buyuksehyr": "Istanbul Basaksehir",
    "Goztep": "Goztepe Izmir",
    "OFI Crete": "Heraklion Crete", "Volos NFC": "Volos", "Larisa": "AEL Larissa",
    "Panserraikos": "Serres",
    "Athens Kallithea": "Kallithea", "RWD Molenbeek": "Molenbeek Brussels",
    "RAAL La Louviere": "La Louviere", "NAC Breda": "Breda",
    "Willem II": "Tilburg", "Telstar": "Velsen", "FeralpiSalo": "Salo",
    "Virtus Entella": "Chiavari", "Juve Stabia": "Castellammare di Stabia",
    "Carrarese": "Carrara", "Red Star": "Saint-Ouen", "Dresden": "Dynamo Dresden",
    "Nacional": "Funchal Madeira", "Truro": "Truro City", "Cultural Leonesa": "Leon",
    "Ceuta": "Ceuta Spain", "Genclerbirligi": "Ankara", "Umraniyespor": "Umraniye",
    "Eyupspor": "Eyup Istanbul", "Kocaelispor": "Izmit", "Bodrumspor": "Bodrum",
    "Giresunspor": "Giresun", "Alanyaspor": "Alanya", "Hatayspor": "Hatay",
    "Istanbulspor": "Istanbul", "Las Palmas": "Las Palmas Gran Canaria",
    "Tenerife": "Santa Cruz de Tenerife", "Ad. Demirspor": "Adana",
    "Levadeiakos": "Livadeia", "Cercle Brugge": "Brugge",
    "AFC Wimbledon": "Wimbledon London",
}

COUNTRY = {"E0": "gb", "E1": "gb", "E2": "gb", "E3": "gb", "EC": "gb",
           "SC0": "gb", "SC1": "gb", "SC2": "gb", "SC3": "gb",
           "D1": "de", "D2": "de", "SP1": "es", "SP2": "es",
           "I1": "it", "I2": "it", "F1": "fr", "F2": "fr",
           "N1": "nl", "B1": "be", "P1": "pt", "T1": "tr", "G1": "gr"}
# (min_lat, max_lat, min_lon, max_lon) -- a sanity check, not a border.
# Spain and Portugal reach a long way past the mainland: Las Palmas and
# Tenerife are in the Canaries, Nacional plays in Madeira, Ceuta is in North
# Africa. Mainland-only boxes silently discarded all four as wrong-country.
BOX = {"gb": (49.8, 61.0, -8.7, 2.0), "de": (47.2, 55.1, 5.8, 15.1),
       "es": (27.5, 43.8, -18.5, 4.4), "it": (36.6, 47.1, 6.6, 18.6),
       "fr": (41.3, 51.1, -5.2, 9.6), "nl": (50.7, 53.6, 3.3, 7.3),
       "be": (49.4, 51.6, 2.5, 6.4), "pt": (36.9, 42.2, -9.6, -6.1),
       "tr": (35.8, 42.2, 25.6, 44.9), "gr": (34.7, 41.8, 19.3, 28.3)}


def lookup(team: str, code: str) -> dict | None:
    name = ALIAS.get(team, team)
    for suffix in (" stadium", " football club", ""):
        for attempt in range(3):
            r = requests.get("https://nominatim.openstreetmap.org/search",
                             params={"q": f"{name}{suffix}", "countrycodes": code,
                                     "format": "json", "limit": 5},
                             headers={"User-Agent": AGENT}, timeout=45)
            time.sleep(DELAY)
            if r.status_code != 429:      # throttled: wait it out and repeat
                break
            time.sleep(20)
        if r.status_code != 200:
            continue
        lo_lat, hi_lat, lo_lon, hi_lon = BOX[code]
        for hit in r.json():
            lat, lon = float(hit["lat"]), float(hit["lon"])
            if lo_lat <= lat <= hi_lat and lo_lon <= lon <= hi_lon:
                return {"lat": lat, "lon": lon,
                        "matched": hit.get("display_name", "")[:120],
                        "query": f"{name}{suffix}".strip()}
    return None


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    done = (pd.read_csv(DEST) if DEST.exists()
            else pd.DataFrame(columns=["team", "country_code", "lat", "lon",
                                       "matched", "query"]))
    seen = set(done.loc[done["lat"].notna(), "team"])
    done = done[done["lat"].notna()]

    m = pd.read_csv(HERE / "matches_multiseason.csv", low_memory=False,
                    usecols=["Div", "HomeTeam", "Season"])
    m = m[m["Season"].astype(str) != "2627"]
    pairs = (m[["Div", "HomeTeam"]].drop_duplicates()
             .assign(code=lambda d: d["Div"].map(COUNTRY))
             .dropna(subset=["code"])
             .drop_duplicates("HomeTeam"))

    rows = list(done.to_dict("records"))
    todo = [(t, c) for t, c in zip(pairs.HomeTeam, pairs.code) if t not in seen]
    print(f"{len(seen)} cached, {len(todo)} to look up", flush=True)

    for i, (team, code) in enumerate(todo, 1):
        try:
            hit = lookup(team, code)
        except Exception as exc:
            hit, note = None, str(exc)[:50]
            print(f"  {team}: {note}", flush=True)
        rows.append({"team": team, "country_code": code,
                     **(hit or {"lat": None, "lon": None, "matched": "", "query": ""})})
        if i % 10 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(DEST, index=False)
            found = sum(r["lat"] is not None and r["lat"] == r["lat"] for r in rows)
            print(f"  {i}/{len(todo)} done, {found} located", flush=True)

    pd.DataFrame(rows).to_csv(DEST, index=False)
    out = pd.read_csv(DEST)
    print(f"\n{out['lat'].notna().sum()} of {len(out)} clubs located")
    missing = out[out["lat"].isna()]["team"].tolist()
    if missing:
        print("not found:", ", ".join(missing))


if __name__ == "__main__":
    main()
