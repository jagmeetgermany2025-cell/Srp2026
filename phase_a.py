"""Phase A — build the multi-season corpus and validate it.

Items 1, 2 and 4 of the build order, plus the ClubElo join check (item 3).
Fetches, assembles, masks the stale Pinnacle feed, writes the parquet, then
runs the checks that justify trusting a 22-season corpus.

The fetch only proves the plumbing works. Everything under VALIDATION targets a
failure mode that a single season, or a few adjacent ones, structurally cannot
surface -- each one below caught something real on this data.

    python phase_a.py                 # a few recent seasons, fast
    python phase_a.py --full          # everything since FIRST_SEASON_YEAR
    python phase_a.py --no-fetch      # assemble from the cache only

Reads/writes beside this module. Report goes to phase_a_report.txt.
"""
from __future__ import annotations

import argparse
import difflib
import os
import warnings

import pandas as pd

import fetch_data
from fetch_data import (CLUBELO_ALIASES, CLUBELO_NOT_A_MATCH, DIVISIONS,
                        NOT_RENAMES, PINNACLE_STALE_FROM, season_start_year)
from pipeline import coverage_table

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
pd.set_option("display.width", 200)

XGABORA_DIR = "Club-Football-Match-Data-2000-2025"
REPORT = "phase_a_report.txt"

OUT = []


def section(title):
    line = "=" * 78
    OUT.append(f"\n{line}\n{title}\n{line}")


def add(text=""):
    OUT.append(str(text))


def table(df):
    add(df.to_string())


# ============================================================ 1. FETCH ======

def build_corpus(seasons, cache_dir, out_parquet, do_fetch=True):
    section("1. FETCH AND ASSEMBLY")

    if do_fetch:
        tally = fetch_data.fetch_seasons(seasons, DIVISIONS, cache_dir)
        usable = tally["ok"] + tally["cached"]
        # any http* = no file published for that season/division, expected for
        # seasons not yet played
        absent = sum(v for k, v in tally.items() if k.startswith("http"))
        add(f"totals: {dict(tally)}")
        add(f"{usable} files usable, {absent} not published yet")
        if tally["notcsv"]:
            add(f"WARNING: {tally['notcsv']} responses were not CSV -- likely a "
                f"block page. Check with:")
            add(f"  curl -k -sS {fetch_data.BASE_URL}/2425/E0.csv | head -c 300")
    else:
        add("--no-fetch: assembling from the cache only")

    df = fetch_data.load_matches(seasons, DIVISIONS, cache_dir)
    add(f"\n{len(df):,} matches, {df.Div.nunique()} divisions, "
        f"{df.Date.min().date()} to {df.Date.max().date()}")

    df = fetch_data.mask_stale_pinnacle(df)
    if "PSH" in df.columns:
        after = df[df.Date >= PINNACLE_STALE_FROM]
        before = df[df.Date < PINNACLE_STALE_FROM]
        add(f"Pinnacle after cutoff : {after.PSH.notna().sum():,} / {len(after):,} (want 0)")
        add(f"Pinnacle before cutoff: {before.PSH.notna().sum():,} / {len(before):,}")

    # drop matches with no result or no opening 1X2 price
    df = df.dropna(subset=["FTR"])
    odds = [c for c in ("B365H", "B365D", "B365A") if c in df.columns]
    if odds:
        df = df.dropna(subset=odds)
    df = df.sort_values("Date").reset_index(drop=True)

    bad = fetch_data.mixed_type_columns(df)
    if bad:
        add("mixed-type columns parquet will reject:")
        for c, kinds in bad:
            add(f"  {c}: {kinds}")
        raise TypeError("fix the columns above before writing")

    df.to_parquet(out_parquet, index=False)
    add(f"{len(df):,} usable matches -> {out_parquet}")
    return df


# ========================================================= 2. COVERAGE ======

def report_coverage(df):
    section("2. COVERAGE BY SEASON")
    add("Per-season completeness by column group. This is what fixes the")
    add("modelling window, not the changelog.\n")
    tbl, flags = coverage_table(df)
    table(tbl.round(3))
    add("\nflags (a group usable over only part of the corpus):")
    for k, v in (flags or {}).items():
        add(f"  {k}: {v}")
    if not flags:
        add("  none -- widen the season range to see this properly")
    return flags


# ========================================================== 3. CLUBELO ======

def report_clubelo(df):
    section("3. CLUBELO JOIN CHECK (item 3)")

    if not os.path.isdir(XGABORA_DIR):
        add(f"'{XGABORA_DIR}' not found. Clone it once:")
        add("  git clone https://github.com/xgabora/Club-Football-Match-Data-2000-2025")
        return None

    elo = pd.read_csv(os.path.join(XGABORA_DIR, "data", "EloRatings.csv"))
    elo_recent = set(elo[elo.date >= "2024-08-01"].club.unique())
    elo_ever = set(elo.club.unique())

    def to_clubelo(t):
        return CLUBELO_ALIASES.get(t, t)

    rows = []
    for div, g in df.groupby("Div"):
        teams = set(pd.concat([g.HomeTeam, g.AwayTeam]).dropna().unique())
        if not teams:
            continue
        matched = sum(to_clubelo(t) in elo_recent for t in teams)
        rows.append((div, len(teams), matched, matched / len(teams) * 100))
    join_check = pd.DataFrame(rows, columns=["Div", "Teams", "Matched", "MatchRate%"])
    table(join_check.sort_values("MatchRate%", ascending=False).round(1))

    # Split the unmatched. A club ClubElo never tracked is expected; one it
    # tracks under a different spelling is a broken join, and on a big club
    # that is a silently missing feature.
    all_teams = set(pd.concat([df.HomeTeam, df.AwayTeam]).dropna().unique())
    rows = []
    for t in sorted(t for t in all_teams if to_clubelo(t) not in elo_recent):
        if to_clubelo(t) in elo_ever:
            rows.append((t, "historic only", ""))       # tracked, not top-500 now
            continue
        close = difflib.get_close_matches(t, elo_ever, n=1, cutoff=0.85)
        if close and t not in CLUBELO_NOT_A_MATCH:
            rows.append((t, "NAME MISMATCH", close[0]))
        else:
            rows.append((t, "not tracked", ""))
    unmatched = pd.DataFrame(rows, columns=["Team", "Status", "ClosestInClubElo"])
    add("\n" + unmatched.Status.value_counts().to_string())

    mismatches = unmatched[unmatched.Status == "NAME MISMATCH"]
    if len(mismatches):
        add(f"\n{len(mismatches)} likely broken joins -- check these by hand:")
        table(mismatches)

    # MatchRate% above asks whether a club sits in ClubElo's *current* top 500,
    # which over 22 seasons counts every long-gone club as a failure. Item 23
    # needs coverage at the match date, so measure per match.
    span = (elo.assign(date=pd.to_datetime(elo.date))
              .groupby("club").date.agg(["min", "max"]))
    lo, hi = span["min"].to_dict(), span["max"].to_dict()

    def rated(team, when):
        t = to_clubelo(team)
        return t in lo and lo[t] <= when <= hi[t]

    both = [rated(h, d) and rated(a, d)
            for h, a, d in zip(df.HomeTeam, df.AwayTeam, df.Date)]

    add(f"\nClubElo snapshots run {span['min'].min().date()} to {span['max'].max().date()}")
    # static snapshot: anything after its last date has no ClubElo at all,
    # which is a coverage limit rather than a join failure
    late = int((df.Date > span["max"].max()).sum())
    if late:
        add(f"{late:,} matches fall after that and can never match -- run "
            f"clubelo_topup.ipynb for the current season")

    cov = pd.DataFrame({"Div": df.Div.values, "both": both})
    add(f"matches with a ClubElo rating for both teams at kickoff: "
        f"{cov.both.mean() * 100:.1f}%")
    per_div = cov.groupby("Div").both.agg(["mean", "size"])
    per_div["mean"] = (per_div["mean"] * 100).round(1)
    table(per_div.rename(columns={"mean": "Covered%", "size": "Matches"})
          .sort_values("Covered%", ascending=False))
    return both


# ==================================================== 4. NAME STABILITY =====

# "Malaga B" after "Malaga" is a reserve side, not a rename
_RESERVE_SUFFIXES = (" B", " II", " C")


def _reserve_pair(a, b):
    short, long_ = sorted((a, b), key=len)
    return any(long_ == short + suf for suf in _RESERVE_SUFFIXES)


def suspected_renames(df, cutoff=0.85):
    """Departures whose name matches an arrival in the same division next season.

    football-data renames clubs mid-corpus (`Ath Madrid` -> `Atl. Madrid`).
    That breaks the ClubElo join and splits our own Elo, which is keyed on
    (division, team) and restarts the "new" club at 1500. Promotion and
    relegation churn teams legitimately, so a departure alone means nothing.
    """
    out = []
    for div, g in df.groupby("Div"):
        per_season = {s: set(pd.concat([x.HomeTeam, x.AwayTeam]).dropna())
                      for s, x in g.groupby("Season")}
        seasons = sorted(per_season)
        for a, b in zip(seasons, seasons[1:]):
            gone, arrived = per_season[a] - per_season[b], per_season[b] - per_season[a]
            for t in sorted(gone):
                close = difflib.get_close_matches(t, arrived, n=1, cutoff=cutoff)
                if (close and not _reserve_pair(t, close[0])
                        and (t, close[0]) not in NOT_RENAMES):
                    out.append((div, a, b, t, close[0]))
    return pd.DataFrame(out, columns=["Div", "LastSeen", "ThenAppears",
                                      "OldName", "NewName"])


# ======================================================== 5. VALIDATION =====

def validate(df):
    section("4. VALIDATION")

    add("4.1 Date parsing")
    add("The riskiest silent failure: old exports use 2-digit years, new ones")
    add("4-digit, and a day/month swap is invisible downstream.\n")
    fmt = pd.DataFrame(
        [(s, d, f, hits, total)
         for (s, d), (f, hits, total) in fetch_data.DATE_FORMAT_LOG.items()],
        columns=["Season", "Div", "Format", "Parsed", "Rows"])
    if fmt.empty:
        add("  no files parsed through parse_dates this run")
    else:
        partial = fmt[fmt.Parsed < fmt.Rows]
        add(f"  files where the chosen format left rows unparsed: {len(partial)}")
        if len(partial):
            table(partial)
        add("\n  formats used per season:")
        table(fmt.groupby("Season").Format.agg(lambda s: sorted(set(s))).to_frame())

    rows = []
    for season, g in df.groupby("Season"):
        y = season_start_year(season)
        lo, hi = pd.Timestamp(f"{y}-07-01"), pd.Timestamp(f"{y + 1}-06-30")
        out = g.Date[(g.Date < lo) | (g.Date > hi)]
        rows.append((season, g.Date.min().date(), g.Date.max().date(), len(g),
                     len(out),
                     out.min().date() if len(out) else "",
                     out.max().date() if len(out) else ""))
    span = pd.DataFrame(rows, columns=["Season", "First", "Last", "Matches",
                                       "OutsideWindow", "OutsideFrom", "OutsideTo"])
    add("")
    table(span)
    bad = span[span.OutsideWindow > 0]
    add(f"\n  seasons with dates outside their own window: {len(bad)}")
    if len(bad):
        add("  Check OutsideFrom/To. A misparse scatters them randomly; a")
        add("  contiguous run past June is a real calendar change -- 2019/20")
        add("  ran to August 2020 because of the COVID suspension.")

    add("\n\n4.2 Column lifespans")
    add("A feature that only appears halfway through the corpus can't be used")
    add("across it without introducing sample selection. All-null = absent.\n")
    watch = ["B365H", "B365CH", "MaxH", "AvgH", "PSH", "BFEH",
             "AHh", "B365AHH", "HS", "HST", "HC", "HY", "Referee"]
    rows = []
    for c in watch:
        if c not in df.columns:
            rows.append((c, "-", "-", 0))
            continue
        s = df.groupby("Season")[c].apply(lambda x: x.notna().any())
        present = s[s].index.tolist()
        rows.append((c, present[0] if present else "-",
                     present[-1] if present else "-", int(s.sum())))
    table(pd.DataFrame(rows, columns=["Column", "FirstSeason", "LastSeason",
                                      "SeasonsPresent"]).sort_values("SeasonsPresent"))
    add(f"\n  total seasons in corpus: {df.Season.nunique()}")

    add("\n\n4.3 Pinnacle masking, both branches")
    add("Masking by date only earns its complexity if it KEEPS Pinnacle before")
    add("the cutoff. That branch needs a season ending before the cutoff.\n")
    if "PSH" in df.columns:
        before = df[df.Date < PINNACLE_STALE_FROM]
        after = df[df.Date >= PINNACLE_STALE_FROM]
        kept, leaked = int(before.PSH.notna().sum()), int(after.PSH.notna().sum())
        add(f"  before cutoff: {kept:,} / {len(before):,} rows keep Pinnacle")
        add(f"  after cutoff : {leaked:,} / {len(after):,} still have it (want 0)")
        if len(before) == 0:
            add("  no rows before the cutoff -- widen the season range")
        elif kept == 0:
            add("  Pinnacle absent before the cutoff too -- check the source columns")

    key = ["Div", "Date", "HomeTeam", "AwayTeam"]
    dupes = int(df.duplicated(subset=key).sum())
    add(f"\n  duplicate matches on {key}: {dupes} (want 0)")

    add("\n\n4.4 Fetch gaps")
    add("A division that runs for years, vanishes for one season, then returns")
    add("is a failed download, not a format change.\n")
    counts = df.pivot_table(index="Div", columns="Season", values="Date",
                            aggfunc="count", fill_value=0)
    table(counts)
    gaps = []
    for div, row in counts.iterrows():
        nz = row.to_numpy().nonzero()[0]
        if len(nz) < 2:
            continue
        for i in range(nz[0], nz[-1]):
            if row.iloc[i] == 0:
                gaps.append((div, row.index[i]))
    add(f"\n  interior gaps: {gaps or 'none'}")

    add("\n\n4.5 Team name stability")
    renames = suspected_renames(df)
    if len(renames):
        add(f"  {len(renames)} suspected renames -- each splits one club's Elo history:")
        table(renames)
        add("  Confirmed ones belong in fetch_data.TEAM_CANONICAL; look-alikes")
        add("  that are different clubs belong in fetch_data.NOT_RENAMES.")
    else:
        add("  no suspected renames")


# =================================================== 6. ANALYSIS WINDOWS ====

def analysis_windows(df, has_clubelo, out_parquet):
    section("5. ANALYSIS WINDOWS")
    add("Closing odds are missing before 2019/20; ClubElo is missing for clubs")
    add("outside Europe's top ~500 and after the dump's last date.")
    add("")
    add("Decided against dropping those matches. ClubElo coverage tracks")
    add("division tier almost exactly, so dropping deletes the lower divisions")
    add("wholesale -- the soft, high-margin markets this project is about, and")
    add("selecting on a variable that correlates with the outcome is the same")
    add("class of bias the study is measuring. Carry availability as flags;")
    add("each analysis restricts itself and reports its own n.\n")

    closing = [c for c in ("B365CH", "B365CD", "B365CA") if c in df.columns]
    df["has_closing"] = df[closing].notna().all(axis=1) if closing else False
    df["has_clubelo"] = has_clubelo if has_clubelo is not None else False

    w = pd.DataFrame([
        ("full corpus", len(df)),
        ("with closing odds", int(df.has_closing.sum())),
        ("with ClubElo", int(df.has_clubelo.sum())),
        ("with both", int((df.has_closing & df.has_clubelo).sum())),
    ], columns=["Window", "Matches"])
    w["% of corpus"] = (w.Matches / len(df) * 100).round(1)
    table(w)

    # per-division cost of dropping, i.e. the reason not to
    loss = df.groupby("Div").agg(Matches=("has_clubelo", "size"),
                                 Kept=("has_clubelo", "sum"))
    loss["Lost%"] = ((1 - loss.Kept / loss.Matches) * 100).round(1)
    add("")
    table(loss.sort_values("Lost%", ascending=False))
    add(f"\ndropping uncovered matches would remove {int((~df.has_clubelo).sum()):,} "
        f"matches ({(~df.has_clubelo).mean() * 100:.1f}%), concentrated in the lower tiers")

    df.to_parquet(out_parquet, index=False)
    add(f"\nflags added, {out_parquet} rewritten")


# ============================================================== main ========

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="fetch everything since FIRST_SEASON_YEAR")
    ap.add_argument("--no-fetch", action="store_true",
                    help="assemble from the cache without downloading")
    ap.add_argument("--cache-dir", default=fetch_data.CACHE_DIR)
    ap.add_argument("--out", default=fetch_data.OUT_PARQUET)
    args = ap.parse_args()

    seasons = (fetch_data.default_seasons() if args.full
               else fetch_data.default_seasons(first_year=2022))
    add(f"{len(seasons)} seasons x {len(DIVISIONS)} divisions: {seasons}")

    df = build_corpus(seasons, args.cache_dir, args.out, do_fetch=not args.no_fetch)
    report_coverage(df)
    has_clubelo = report_clubelo(df)
    validate(df)
    analysis_windows(df, has_clubelo, args.out)

    text = "\n".join(OUT)
    print(text)
    with open(REPORT, "w") as fh:
        fh.write(text)
    print(f"\nreport written to {REPORT}")


if __name__ == "__main__":
    main()
