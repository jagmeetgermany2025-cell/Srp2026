"""
Write per-match Elo features for every match, fitted walk-forward.

The constants are refitted for each season on that season's past and nothing
else, so the ratings a season is scored with never saw that season. The first
season has no past to fit on and is left out of the table rather than scored
with constants fitted on itself.

Fitting on a single season leaves `carry` -- how much of a rating survives the
summer -- unidentified, since there is no summer inside one season, and the
optimiser then parks it at zero, which throws away every rating each August.
An expanding window fixes that from the second season onwards.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

import numpy as np
import pandas as pd

import elo

SOURCE = str(DATA / "matches_all.csv")
DEST = str(DATA) + "/elo/elo_by_match.csv"
KEEP = ["Div", "Season", "Date", "HomeTeam", "AwayTeam"]
MIN_SEASON_MATCHES = 1000


def main() -> None:
    raw = pd.read_csv(SOURCE, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Season"] = raw["Season"].astype(str)
    df = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    df = elo.add_country(df).sort_values("Date").reset_index(drop=True)

    counts = df["Season"].value_counts()
    seasons = sorted(s for s in df["Season"].unique()
                     if counts[s] >= MIN_SEASON_MATCHES)
    played = df["FTR"].isin(["H", "D", "A"]).to_numpy()
    onehot = np.column_stack([df.FTR == "H", df.FTR == "D", df.FTR == "A"]).astype(float)

    pieces, report = [], []
    for season in seasons[1:]:
        past = df[df["Season"] < season]
        params, train_loss = elo.fit_elo(past)
        # the walk has to run over the history as well, or the ratings this
        # season starts with would be wrong; only this season is then kept
        full = elo.run_elo(df, params)
        mask = (df["Season"] == season).to_numpy()
        pieces.append(pd.concat([df.loc[mask, KEEP], full[mask]], axis=1))

        scored = mask & played
        p = full.loc[scored, ["elo_p_H", "elo_p_D", "elo_p_A"]].to_numpy()
        loss = float(-np.mean(np.log(np.clip((p * onehot[scored]).sum(1), 1e-12, 1))))
        base = past.loc[past.FTR.isin(list("HDA")), "FTR"].value_counts(normalize=True)
        bp = np.tile([base.get("H", 0), base.get("D", 0), base.get("A", 0)],
                     (scored.sum(), 1))
        base_loss = float(-np.mean(np.log(np.clip((bp * onehot[scored]).sum(1), 1e-12, 1))))
        report.append((season, scored.sum(), params, train_loss, loss, base_loss))
        print(f"{season}: fitted on {', '.join(sorted(past.Season.unique()))} "
              f"(train {train_loss:.4f}) -> test {loss:.4f} vs base {base_loss:.4f}",
              flush=True)

    out = pd.concat(pieces, ignore_index=True)
    out.to_csv(DEST, index=False)

    print(f"\nwrote {DEST}: {len(out):,} matches, seasons "
          f"{', '.join(seasons[1:])}")
    print("\nfitted constants by season:")
    for season, n, p, _, loss, base_loss in report:
        print(f"  {season}  n={n:>6,}  k={p.k:>5.1f}  home_adv={p.home_adv:>5.1f}  "
              f"gamma={p.gamma:.2f}  carry={p.carry:.2f}  beta={p.beta:.5f}  "
              f"delta={p.delta:.2f}  loss {loss:.4f} (base {base_loss:.4f})")


if __name__ == "__main__":
    main()
