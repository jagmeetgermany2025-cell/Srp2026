"""
Elo ratings built from results alone, with every constant fitted rather than
chosen.

Textbook Elo hides half a dozen hand-set numbers: the update size K, how much a
home ground is worth, how much a three-goal win counts over a one-goal win, how
much of a rating survives the summer. Picking those by eye is exactly the thing
this project refuses to do elsewhere, so here they are parameters, fitted on
training seasons by log loss and then held fixed for the seasons that follow.

The ratings are strictly pre-match: a match is predicted from the ratings the
two teams carried INTO it, and only afterwards do those ratings move. Walking
the fixtures in date order is what guarantees that, so the walk is the whole
implementation.

Two league-specific details matter for a 22-division file:

  * Teams are keyed by country and name, not by division, so a promoted side
    carries its rating up with it. That carried rating is most of the value:
    a promoted team is usually weaker than the division it joins, and nothing
    else in the feature set knows that.
  * A team seen for the first time starts at the current average of the
    division it appears in, not at a fixed 1500, because a Greek second-tier
    side and a Bundesliga side do not belong on the same scale.
"""
from dataclasses import dataclass, astuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

START = 1500.0            # the scale's origin; any constant would do

# (low, high) for each fitted parameter, in the order of EloParams
BOUNDS = [(1.0, 120.0),   # k          update size
          (0.0, 250.0),   # home_adv   rating points a home ground is worth
          (0.0, 1.5),     # gamma      how much a bigger margin counts
          (0.0, 1.0),     # carry      fraction of a rating kept over a summer
          (0.0005, 0.05), # beta       rating points -> latent strength
          (0.05, 3.0)]    # delta      half-width of the drawing band


@dataclass
class EloParams:
    k: float = 20.0
    home_adv: float = 60.0
    gamma: float = 0.5
    carry: float = 0.75
    beta: float = 0.0058          # ln(10)/400, the textbook scale
    delta: float = 0.5

    def clipped(self) -> "EloParams":
        return EloParams(*[float(np.clip(v, lo, hi))
                           for v, (lo, hi) in zip(astuple(self), BOUNDS)])


def outcome_probs(strength: np.ndarray, p: EloParams) -> np.ndarray:
    """
    Turn a rating difference into home/draw/away probabilities.

    An ordered logit, which is the natural choice: the three results are
    ordered, and a draw is what happens when the latent difference in strength
    is too small to decide the match either way. delta is the width of that
    indecisive band, fitted like everything else.
    """
    s = p.beta * strength
    p_away = expit(-p.delta - s)
    p_home = 1.0 - expit(p.delta - s)
    return np.column_stack([p_home, 1.0 - p_home - p_away, p_away])


def add_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure a Country column exists, because that is half the team key.

    Divisions carry it implicitly (E0 and E1 are both England), and teams move
    between divisions inside a country every summer. Keying on the division
    instead would reset a promoted team to its new division's average and throw
    away the very thing we want to measure.
    """
    if "Country" in df.columns and df["Country"].notna().all():
        return df
    from two_stage import TIERS                     # imported late: heavy module
    out = df.copy()
    out["Country"] = out["Div"].map(lambda x: TIERS.get(x, (str(x), 0))[0])
    return out


def run_elo(df: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    """
    Walk the fixtures in date order and return the PRE-match state of each one.

    Everything returned for a match is known before it kicks off, so these
    columns can go straight into a market-blind model without leaking.
    """
    p = params.clipped()
    d = add_country(df).sort_values(["Date", "Div", "HomeTeam"], kind="mergesort")

    home_key = d["Country"].astype(str) + "|" + d["HomeTeam"].astype(str)
    away_key = d["Country"].astype(str) + "|" + d["AwayTeam"].astype(str)
    divs = d["Div"].astype(str).to_numpy()
    seasons = d["Season"].astype(str).to_numpy()
    gh = d["FTHG"].to_numpy(dtype=float)
    ga = d["FTAG"].to_numpy(dtype=float)

    rating: dict[str, float] = {}
    div_members: dict[str, set] = {}
    eh = np.empty(len(d))
    ea = np.empty(len(d))
    season_now = seasons[0] if len(d) else ""

    def rating_of(team: str, div: str) -> float:
        if team not in rating:
            peers = [rating[t] for t in div_members.get(div, ()) if t in rating]
            rating[team] = float(np.mean(peers)) if peers else START
        div_members.setdefault(div, set()).add(team)
        return rating[team]

    for i, (h, a, div, season) in enumerate(zip(home_key, away_key, divs, seasons)):
        if season != season_now:            # summer: regress toward the mean
            season_now = season
            if rating:
                m = float(np.mean(list(rating.values())))
                for t in rating:
                    rating[t] = m + p.carry * (rating[t] - m)

        rh, ra = rating_of(h, div), rating_of(a, div)
        eh[i], ea[i] = rh, ra

        if np.isnan(gh[i]) or np.isnan(ga[i]):       # unplayed: no update
            continue
        expected = 1.0 / (1.0 + 10.0 ** (-(rh + p.home_adv - ra) / 400.0))
        score = 1.0 if gh[i] > ga[i] else (0.5 if gh[i] == ga[i] else 0.0)
        move = p.k * (1.0 + abs(gh[i] - ga[i])) ** p.gamma * (score - expected)
        rating[h], rating[a] = rh + move, ra - move

    strength = eh + p.home_adv - ea
    probs = outcome_probs(strength, p)
    return pd.DataFrame({
        "elo_home": eh,
        "elo_away": ea,
        "elo_diff": strength,
        "elo_p_H": probs[:, 0],
        "elo_p_D": probs[:, 1],
        "elo_p_A": probs[:, 2],
    }, index=d.index).reindex(df.index)


def log_loss_of(df: pd.DataFrame, params: EloParams) -> float:
    """Mean log loss of the Elo probabilities against the actual results."""
    out = run_elo(df, params)
    played = df["FTR"].isin(["H", "D", "A"]).to_numpy()
    if not played.any():
        return np.inf
    cols = {"H": "elo_p_H", "D": "elo_p_D", "A": "elo_p_A"}
    p_actual = np.array([out.loc[i, cols[r]]
                         for i, r in df.loc[played, "FTR"].items()])
    return float(-np.mean(np.log(np.clip(p_actual, 1e-12, 1.0))))


def fit_elo(train: pd.DataFrame, seed_params: EloParams = EloParams(),
            verbose: bool = False) -> tuple[EloParams, float]:
    """
    Fit the six constants on training matches by log loss.

    Nelder-Mead, because the walk is not differentiable in any useful sense:
    a change in K changes every later rating through a chain of updates. The
    search is small enough (six parameters) that this is not a problem.
    """
    def objective(x: np.ndarray) -> float:
        loss = log_loss_of(train, EloParams(*x))
        if verbose:
            print(f"    {np.round(x, 4)} -> {loss:.5f}", flush=True)
        return loss

    res = minimize(objective, np.array(astuple(seed_params)),
                   method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-6})
    return EloParams(*res.x).clipped(), float(res.fun)


# =========================================================================
# A standalone check: does Elo know anything the base rate does not?
# =========================================================================
def _report(path: str = "matches_multiseason.csv") -> None:
    import sys
    from two_stage import shin_de_vig

    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Season"].astype(str)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    counts = df["Season"].value_counts()
    df = df[df["Season"].isin(counts[counts >= 1000].index)]
    df = add_country(df).sort_values("Date").reset_index(drop=True)

    seasons = sorted(df["Season"].unique())
    test_season = seasons[-1]
    train = df[df["Season"] < test_season]
    print(f"{len(df):,} matches, seasons {', '.join(seasons)}; "
          f"fitting on {', '.join(seasons[:-1])}, testing on {test_season}")

    params, train_loss = fit_elo(train, verbose="-v" in sys.argv)
    print(f"\nfitted: k={params.k:.1f}  home_adv={params.home_adv:.1f}  "
          f"gamma={params.gamma:.2f}  carry={params.carry:.2f}  "
          f"beta={params.beta:.5f}  delta={params.delta:.2f}")
    print(f"train log loss {train_loss:.4f}")

    # Ratings must be carried through the training seasons to be right for the
    # test season, so the walk runs over everything and is then sliced.
    out = run_elo(df, params)
    mask = (df["Season"] == test_season).to_numpy()
    actual = df.loc[mask, "FTR"].to_numpy()
    onehot = np.column_stack([actual == "H", actual == "D", actual == "A"]).astype(float)

    def loss(p3: np.ndarray) -> float:
        return float(-np.mean(np.log(np.clip((p3 * onehot).sum(1), 1e-12, 1.0))))

    elo_p = out.loc[mask, ["elo_p_H", "elo_p_D", "elo_p_A"]].to_numpy()
    base = train["FTR"].value_counts(normalize=True)
    base_p = np.tile([base.get("H", 0), base.get("D", 0), base.get("A", 0)],
                     (mask.sum(), 1))

    rows = [("base rate (train frequencies)", loss(base_p)),
            ("elo", loss(elo_p))]

    cols = ["AvgCH", "AvgCD", "AvgCA"]
    if all(c in df.columns for c in cols):
        sub = df.loc[mask, cols]
        ok = sub.notna().all(axis=1).to_numpy()
        market = np.array([shin_de_vig(h, d, a)
                           for h, d, a in sub[ok].to_numpy()])
        m_onehot = onehot[ok]
        m_loss = float(-np.mean(np.log(np.clip((market * m_onehot).sum(1), 1e-12, 1.0))))
        rows.append((f"market closing ({ok.sum():,} of {mask.sum():,} matches)", m_loss))

    print(f"\nout-of-sample log loss on {test_season} ({mask.sum():,} matches)")
    for label, value in rows:
        print(f"  {label:<40} {value:.4f}")


if __name__ == "__main__":
    _report()
