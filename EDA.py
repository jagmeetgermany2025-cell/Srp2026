"""
Exploratory Data Analysis — football-data.co.uk 2025/26, 22 divisions.
    Part 1  (below)      sections 1-3    data quality, coverage, structure
    Part 2  (line ~160)  sections 4-8    outcomes, market efficiency, signals
    Part 3  (line ~410)  sections 9-10   the threshold anomaly, plus figures
"""
import warnings

import matplotlib
matplotlib.use("Agg")   # headless backend, must be set before pyplot is imported
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from load_data import load

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

# Report lines accumulate here, then get printed and written out at the end.
OUT = []


def section(title):
    line = "=" * 78
    OUT.append(f"\n{line}\n{title}\n{line}")


def add(text=""):
    OUT.append(str(text))


df = load("csv_out")
df["Date"] = pd.to_datetime(df["Date"])

# Division code -> (country, tier). Tier 1 is the top flight. Used throughout to
# group divisions by country and to test whether league level carries signal.
TIERS = {
    "E0": ("England", 1), "E1": ("England", 2), "E2": ("England", 3),
    "E3": ("England", 4), "EC": ("England", 5),
    "SC0": ("Scotland", 1), "SC1": ("Scotland", 2), "SC2": ("Scotland", 3),
    "SC3": ("Scotland", 4),
    "D1": ("Germany", 1), "D2": ("Germany", 2),
    "SP1": ("Spain", 1), "SP2": ("Spain", 2),
    "I1": ("Italy", 1), "I2": ("Italy", 2),
    "F1": ("France", 1), "F2": ("France", 2),
    "N1": ("Netherlands", 1), "B1": ("Belgium", 1), "P1": ("Portugal", 1),
    "T1": ("Turkey", 1), "G1": ("Greece", 1),
}
df["Country"] = df.League.map(lambda x: TIERS[x][0])
df["Tier"] = df.League.map(lambda x: TIERS[x][1])

# ---------------------------------------------------------------- 1. OVERVIEW
section("1. DATASET OVERVIEW")
add(f"Rows                : {len(df):,}")
add(f"Columns             : {df.shape[1]}")
add(f"Divisions           : {df.League.nunique()}")
add(f"Countries           : {df.Country.nunique()}")
add(f"Date range          : {df.Date.min().date()} to {df.Date.max().date()}")
add(f"Span                : {(df.Date.max() - df.Date.min()).days} days")
add(f"Unique teams        : {pd.concat([df.HomeTeam, df.AwayTeam]).nunique():,}")
add(f"Unique referees     : {df.Referee.nunique()}")

add("\nMatches per division:")
tbl = (df.groupby(["Country", "Tier", "League"])
         .size().reset_index(name="Matches")
         .sort_values(["Country", "Tier"]))
# A team can appear as home or away, so count distinct names across both columns.
teams = df.groupby("League").apply(
    lambda g: pd.concat([g.HomeTeam, g.AwayTeam]).nunique())
tbl["Teams"] = tbl.League.map(teams)
# n(n-1) is one full home-and-away season. Dividing actual matches by it gives a
# format-independent completeness measure: ~1.00 normal, >1 play-offs, <1 gaps.
tbl["RoundRobin"] = tbl.Teams * (tbl.Teams - 1)
tbl["Rounds"] = (tbl.Matches / tbl.RoundRobin).round(2)
add(tbl[["Country","Tier","League","Teams","Matches","Rounds"]].to_string(index=False))

add("\n'Rounds' = matches / n(n-1), i.e. how many times the full double")
add("round-robin was played. 1.00 is a standard home-and-away season. Scotland")
add("plays roughly four rounds in its smaller divisions; Belgium and Greece")
add("exceed 1.00 because of end-of-season play-off phases. Values slightly")
add("below 1.00 (0.93-0.97) indicate fixtures absent from the file rather than")
add("a different format.")

# --------------------------------------------------------- 2. DATA COMPLETENESS
section("2. DATA COMPLETENESS")

# Columns grouped into families that are useful (or useless) together — a row is
# only counted complete if every column in its family is present.
groups = {
    "Match identity": ["Div", "Date", "Time", "HomeTeam", "AwayTeam"],
    "Result": ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"],
    "Officials": ["Referee"],
    "Shots": ["HS", "AS", "HST", "AST"],
    "Discipline": ["HF", "AF", "HY", "AY", "HR", "AR"],
    "Corners": ["HC", "AC"],
    "1X2 opening": ["B365H", "B365D", "B365A"],
    "1X2 closing": ["B365CH", "B365CD", "B365CA"],
    "Exchange": ["BFEH", "BFED", "BFEA"],
    "Pinnacle": ["PSH", "PSD", "PSA"],
    "Aggregates": ["MaxH", "AvgH"],
    "Over/under": ["B365>2.5", "B365<2.5"],
    "Asian handicap": ["AHh", "B365AHH", "B365AHA"],
}
add(f"{'Group':<18}{'Columns':>9}{'Complete':>10}   Missing where")
for name, cols in groups.items():
    cols = [c for c in cols if c in df.columns]
    if not cols:
        add(f"{name:<18}{'--':>9}{'ABSENT':>10}")
        continue
    cov = df[cols].notna().all(axis=1)
    # Name the worst-affected divisions: missingness here is concentrated by
    # division (e.g. no referee in Spain/Italy), not spread evenly.
    miss = df.loc[~cov, "League"].value_counts()
    where = ", ".join(f"{k}({v})" for k, v in miss.head(4).items()) if len(miss) else "-"
    add(f"{name:<18}{len(cols):>9}{cov.mean()*100:>9.1f}%   {where}")

add("\nPer-bookmaker coverage, 1X2 opening:")
BOOKS = {"B365": "Bet365", "BFD": "Betfred", "BMGM": "BetMGM", "BV": "Betvictor",
         "BW": "Bet&Win", "CL": "Coral", "LB": "Ladbrokes", "PS": "Pinnacle",
         "Max": "Market max", "Avg": "Market avg", "BFE": "Betfair Exch"}
for code, label in BOOKS.items():
    c = f"{code}H"
    if c in df.columns:
        add(f"  {label:<14}{df[c].notna().mean()*100:6.1f}%")

add("\nPinnacle sits far below the rest. Football-data.co.uk reports that since")
add("23/07/2025 Pinnacle's public API has been unreliable, and Pinnacle has been")
add("removed from the Max/Avg calculations. This is confirmed independently in")
add("Section 6 below.")

# ------------------------------------------------------------ 3. INTEGRITY
section("3. INTERNAL CONSISTENCY CHECKS")
add("Verifying column meanings from the data rather than from documentation.\n")

# Each check is (label, pass rate). Anything below 100% means a column does not
# mean what its name implies — cheaper to find here than downstream in a model.
checks = []

# Result letter must agree with the goal columns.
d = df.dropna(subset=["FTHG", "FTAG", "FTR"])
derived = np.where(d.FTHG > d.FTAG, "H", np.where(d.FTHG < d.FTAG, "A", "D"))
checks.append(("FTR equals sign(FTHG - FTAG)", (derived == d.FTR).mean()))

# Shots on target are a subset of shots, so ST can never exceed S.
s = df.dropna(subset=["HS", "HST", "AS", "AST"])
checks.append(("HST <= HS", (s.HST <= s.HS).mean()))
checks.append(("AST <= AS", (s.AST <= s.AS).mean()))

# Half-time goals are cumulative into full-time, so HT can never exceed FT.
h = df.dropna(subset=["HTHG", "FTHG", "HTAG", "FTAG"])
checks.append(("HTHG <= FTHG", (h.HTHG <= h.FTHG).mean()))
checks.append(("HTAG <= FTAG", (h.HTAG <= h.FTAG).mean()))

# Implied probabilities must sum above 1 (that excess is the bookmaker margin);
# above ~1.20 would suggest the odds are on a different scale than assumed.
o = df.dropna(subset=["B365H", "B365D", "B365A"])
ov = 1/o.B365H + 1/o.B365D + 1/o.B365A
checks.append(("1X2 implied probs sum > 1", (ov > 1).mean()))
checks.append(("1X2 overround < 1.20", (ov < 1.20).mean()))

# MaxH is billed as the best price across the market, so no member of that set
# can beat it. Tested against the bookmakers and Pinnacle separately, because
# Pinnacle is the one suspected of being excluded from the aggregate.
# (1e-9 tolerance absorbs float error on prices stored to 2 decimals.)
bm = [f"{c}H" for c in ["B365", "BFD", "BMGM", "BV", "BW", "CL", "LB"]
      if f"{c}H" in df.columns]
m = df.dropna(subset=["MaxH"])
checks.append(("MaxH >= all bookmakers (excl. Pinnacle)",
               (m.MaxH >= m[bm].max(axis=1) - 1e-9).mean()))
if "PSH" in df.columns:
    mp = df.dropna(subset=["MaxH", "PSH"])
    checks.append(("MaxH >= Pinnacle", (mp.MaxH >= mp.PSH - 1e-9).mean()))

# Sanity-check the handicap sign convention: a higher (less negative) line means
# a weaker home side, which should come with longer home odds.
a = df.dropna(subset=["AHh", "B365H"])
checks.append(("corr(AHh, B365H) > 0.5",
               1.0 if np.corrcoef(a.AHh, a.B365H)[0, 1] > 0.5 else 0.0))

for name, rate in checks:
    flag = "PASS" if rate > 0.995 else ("PARTIAL" if rate > 0.90 else "FAIL")
    add(f"  {name:<42}{rate*100:6.1f}%   {flag}")

add("\nThe Pinnacle row is the diagnostic one. Every other bookmaker respects the")
add("recorded market maximum; Pinnacle does not, which is what an excluded and")
add("stale feed looks like from inside the data.")

print("\n".join(OUT))
with open("eda_part1.txt", "w") as f:
    f.write("\n".join(OUT))
# =============================================================================
# PART 2 — outcomes, market efficiency, signal testing (sections 4-8)
# Was a standalone script; the string below is its old module docstring and is
# now just an unused expression. Reloads the data and redefines OUT/section/add.
# Its imports now live in the single block at the top of the file.
# =============================================================================
"""
Exploratory Data Analysis — Part 2.
Outcome structure, market efficiency, and signal testing.
"""
OUT = []
def section(t):
    OUT.append("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def add(t=""):
    OUT.append(str(t))

# df = pd.read_pickle("all.pkl")
df = load("csv_out")
df["Date"] = pd.to_datetime(df.Date)
TIERS = {"E0":("England",1),"E1":("England",2),"E2":("England",3),"E3":("England",4),
         "EC":("England",5),"SC0":("Scotland",1),"SC1":("Scotland",2),"SC2":("Scotland",3),
         "SC3":("Scotland",4),"D1":("Germany",1),"D2":("Germany",2),"SP1":("Spain",1),
         "SP2":("Spain",2),"I1":("Italy",1),"I2":("Italy",2),"F1":("France",1),
         "F2":("France",2),"N1":("Netherlands",1),"B1":("Belgium",1),"P1":("Portugal",1),
         "T1":("Turkey",1),"G1":("Greece",1)}
df["Country"] = df.League.map(lambda x: TIERS[x][0])
df["Tier"] = df.League.map(lambda x: TIERS[x][1])

# Analysis set: played matches that also carry opening Bet365 prices.
d = df[df.FTR.isin(["H", "D", "A"])].dropna(
    subset=["B365H", "B365D", "B365A", "FTHG", "FTAG"]).copy()


def devig(h, dd, a):
    """Odds -> fair probabilities, stripping the margin by proportional scaling.

    1/odds sums above 1; dividing by that sum renormalises to 1. Simple, but it
    spreads the margin evenly across outcomes — section 7 shows that assumption
    is wrong (the margin is heavier on longshots), which is why Shin and power
    de-vigging are flagged there as alternatives worth comparing.
    """
    s = 1/h + 1/dd + 1/a
    return (1/h)/s, (1/dd)/s, (1/a)/s


d["oH"], d["oD"], d["oA"] = devig(d.B365H, d.B365D, d.B365A)   # o = opening
cH, cD, cA = devig(d.B365CH, d.B365CD, d.B365CA)               # c = closing
d["cH"], d["cD"], d["cA"] = cH, cD, cA
# Outcome as an integer class, fixed as H=0 D=1 A=2 everywhere below.
y = d.FTR.map({"H": 0, "D": 1, "A": 2}).values.astype(int)
# Overround = the raw 1/odds sum; the excess over 1 is the bookmaker's margin.
d["overround"] = 1/d.B365H + 1/d.B365D + 1/d.B365A

# ------------------------------------------------------------ 4. OUTCOMES
section("4. OUTCOME STRUCTURE")
add(f"Usable matches: {len(d):,}\n")
vc = d.FTR.value_counts(normalize=True)
add(f"Home win  {vc.get('H',0)*100:5.1f}%")
add(f"Draw      {vc.get('D',0)*100:5.1f}%")
add(f"Away win  {vc.get('A',0)*100:5.1f}%")
add(f"\nMean goals: home {d.FTHG.mean():.2f}, away {d.FTAG.mean():.2f}, "
    f"total {(d.FTHG+d.FTAG).mean():.2f}")
add(f"Home advantage in goals: {(d.FTHG-d.FTAG).mean():+.3f}")

add("\nHome-win rate and draw rate by division (sorted by home advantage):")
g = d.groupby(["Country", "Tier", "League"]).agg(
    n=("FTR", "size"),
    home=("FTR", lambda s: (s == "H").mean()*100),
    draw=("FTR", lambda s: (s == "D").mean()*100),
    away=("FTR", lambda s: (s == "A").mean()*100),
    goals=("FTHG", lambda s: 0.0)).reset_index()
# `goals` needs both goal columns, which .agg() cannot see at once, so it is
# reserved as a zero placeholder above and overwritten here. Relies on both
# groupbys emitting the same key order — true because pandas sorts keys by
# default, but it is the fragile part of this block.
g["goals"] = d.groupby(["Country","Tier","League"]).apply(
    lambda x: (x.FTHG+x.FTAG).mean()).values
g = g.sort_values("home", ascending=False)
add(f"{'Div':<6}{'Country':<13}{'T':>2}{'n':>7}{'Home%':>8}{'Draw%':>8}{'Away%':>8}{'Goals':>8}")
for _, r in g.iterrows():
    add(f"{r.League:<6}{r.Country:<13}{r.Tier:>2}{r.n:>7.0f}{r.home:>8.1f}"
        f"{r.draw:>8.1f}{r.away:>8.1f}{r.goals:>8.2f}")

add("\nThe draw rate is remarkably stable (roughly a quarter of matches everywhere)")
add("while home advantage varies substantially by division. Any model pooling")
add("divisions needs a league term for home advantage.")

# ---------------------------------------------------- 5. MARKET STRUCTURE
section("5. MARKET STRUCTURE")

def ll(P, yy):
    """Multiclass log loss. Clipped away from 0 so a confident miss stays finite."""
    return log_loss(yy, np.clip(P, 1e-9, 1), labels=[0, 1, 2])


add("Market quality by price source (all divisions):\n")
add(f"{'Source':<26}{'n':>7}{'Margin':>9}{'LogLoss':>10}{'RPS':>8}")


def rps(P, yy):
    """Ranked probability score — squared error between cumulative distributions.

    Unlike log loss this respects the H < D < A ordering, so predicting a draw
    when the home side wins is penalised less than predicting an away win.
    Summing the first 2 of 3 cumulative terms and halving is the standard
    1/(r-1) normalisation; the final cumulative term is always 1 on both sides.
    """
    Y = np.eye(3)[yy]
    return np.mean(np.sum((np.cumsum(P, 1) - np.cumsum(Y, 1))[:, :2]**2, 1) / 2)

for label, (h, dd, a) in {
    "Bet365 opening": ("B365H", "B365D", "B365A"),
    "Bet365 closing": ("B365CH", "B365CD", "B365CA"),
    "Market average opening": ("AvgH", "AvgD", "AvgA"),
    "Market average closing": ("AvgCH", "AvgCD", "AvgCA"),
    "Best price opening": ("MaxH", "MaxD", "MaxA"),
    "Betfair Exchange opening": ("BFEH", "BFED", "BFEA"),
    "Betfair Exchange closing": ("BFECH", "BFECD", "BFECA"),
    "Pinnacle opening": ("PSH", "PSD", "PSA"),
}.items():
    if h not in d.columns:
        continue
    # Each source is scored only where it quotes, so `n` differs between rows and
    # the log-loss column is not strictly like-for-like across sources.
    m = d[[h, dd, a]].notna().all(axis=1)
    if m.sum() < 50:
        continue
    s = 1/d.loc[m, h] + 1/d.loc[m, dd] + 1/d.loc[m, a]
    P = np.c_[(1/d.loc[m, h])/s, (1/d.loc[m, dd])/s, (1/d.loc[m, a])/s]
    add(f"{label:<26}{m.sum():>7}{(s.mean()-1)*100:>8.2f}%"
        f"{ll(P, y[m.values]):>10.4f}{rps(P, y[m.values]):>8.4f}")

add("\nClosing prices are sharper than opening, as expected. The exchange carries")
add("a far smaller margin than any bookmaker. Pinnacle's figures are computed on")
add("a reduced and unreliable sample and should not be used for 2025/26.")

add("\n\nMarket predictive quality by division (de-vigged Bet365 opening):\n")
add(f"{'Div':<6}{'Country':<13}{'T':>2}{'n':>7}{'Margin':>9}{'LogLoss':>10}{'RPS':>8}")
rows = []
for (c, t, L), gg in d.groupby(["Country", "Tier", "League"]):
    P = gg[["oH", "oD", "oA"]].values
    yy = gg.FTR.map({"H": 0, "D": 1, "A": 2}).values.astype(int)
    rows.append((L, c, t, len(gg), (gg.overround.mean()-1)*100,
                 ll(P, yy), rps(P, yy)))
rows.sort(key=lambda r: -r[5])          # worst-forecast division first
for L, c, t, n, mg, l, r in rows:
    add(f"{L:<6}{c:<13}{t:>2}{n:>7}{mg:>8.2f}%{l:>10.4f}{r:>8.4f}")

# Correlate margin against log loss across the 22 divisions: does the bookmaker
# charge more where it forecasts worse? Tuple fields are (4) margin, (5) log loss.
mg = np.array([r[4] for r in rows]); lo = np.array([r[5] for r in rows])
cc = np.corrcoef(mg, lo)[0, 1]
add(f"\ncorr(margin, market log loss) across divisions = {cc:+.3f}")
add("\nThis is the central structural finding. Divisions the bookmaker predicts")
add("worst are also the divisions where it charges most. The soft markets are")
add("not cheap markets: the extra margin is priced against exactly the")
add("uncertainty that might otherwise be exploitable.")

# ------------------------------------------- 6. PINNACLE ANOMALY, CONFIRMED
section("6. THE PINNACLE ANOMALY")
# Follows up the failed MaxH >= Pinnacle check from section 3. If Pinnacle were
# still in the aggregate it could never beat the market maximum; how often it
# does, and by how much, measures how stale the feed is.
# Caveat: the two rates printed below use different denominators — Pinnacle over
# matches where both prices exist, the bookmakers over all matches with MaxH.
if "PSH" in d.columns:
    m = d[["MaxH", "PSH"]].notna().all(axis=1)
    viol = (d.loc[m, "PSH"] > d.loc[m, "MaxH"] + 1e-9)
    add(f"Matches with both MaxH and Pinnacle : {m.sum():,}")
    add(f"Pinnacle price exceeds market maximum: {viol.sum():,} ({viol.mean()*100:.1f}%)")
    bm = [f"{c}H" for c in ["B365","BFD","BMGM","BV","BW","CL","LB"] if f"{c}H" in d.columns]
    m2 = d["MaxH"].notna()
    v2 = (d.loc[m2, bm].max(axis=1) > d.loc[m2, "MaxH"] + 1e-9)
    add(f"Any other bookmaker exceeds maximum  : {v2.sum():,} ({v2.mean()*100:.1f}%)")
    add(f"\nMean absolute excess where violated  : "
        f"{(d.loc[m,'PSH'][viol] - d.loc[m,'MaxH'][viol]).mean():.3f} odds points")
    add("\nA price cannot exceed the maximum of the set it belongs to. Pinnacle")
    add("does so in a quarter of matches while no other bookmaker ever does.")
    add("This independently confirms the reported exclusion, and means Pinnacle")
    add("columns must be dropped for this season.")

# ------------------------------------------- 7. FAVOURITE-LONGSHOT BIAS
section("7. FAVOURITE-LONGSHOT BIAS")
# Stack all three outcomes of every match into one long frame of bet-outcomes, so
# calibration can be judged across the whole probability range at once rather
# than per outcome type. p = de-vigged probability, o = odds, w = did it happen.
rows = []
for side, odds, hit in [("H", d.B365H, d.FTR == "H"),
                        ("D", d.B365D, d.FTR == "D"),
                        ("A", d.B365A, d.FTR == "A")]:
    rows.append(pd.DataFrame({"p": (1/odds)/d.overround, "o": odds,
                              "w": hit.astype(float), "side": side}))
b = pd.concat(rows, ignore_index=True)
# Narrow bins at the long-odds end, where the bias is expected to be strongest.
b["bin"] = pd.cut(b.p, [0, .05, .10, .20, .30, .40, .50, .60, .70, 1.0])
add(f"All {len(b):,} bet-outcomes, binned by de-vigged implied probability:\n")
add(f"{'Implied probability':<22}{'n':>7}{'Predicted':>11}{'Observed':>10}"
    f"{'Ratio':>8}{'Flat ROI':>10}")
for bb, gg in b.groupby("bin", observed=True):
    # Flat 1-unit stake: win returns (odds - 1), a loss returns -1.
    roi = np.where(gg.w > 0, gg.o - 1, -1).mean()*100
    # Ratio < 1 means the bin wins less often than priced (overpriced longshots).
    ratio = gg.w.mean()/gg.p.mean()
    add(f"{str(bb):<22}{len(gg):>7}{gg.p.mean():>11.3f}{gg.w.mean():>10.3f}"
        f"{ratio:>8.2f}{roi:>9.1f}%")

lowp = b[b.p < 0.15]; hip = b[b.p > 0.60]
add(f"\nLongshots (p<0.15): predicted {lowp.p.mean():.3f}, observed "
    f"{lowp.w.mean():.3f}, ratio {lowp.w.mean()/lowp.p.mean():.2f}")
add(f"Favourites (p>0.60): predicted {hip.p.mean():.3f}, observed "
    f"{hip.w.mean():.3f}, ratio {hip.w.mean()/hip.p.mean():.2f}")
add("\nThe ratio rises monotonically with implied probability: longshots win")
add("less often than priced, favourites more often. This is the classical")
add("favourite-longshot bias, and its presence here means the bookmaker's")
add("margin is not distributed evenly across outcomes. Naive de-vigging by")
add("proportional normalisation therefore mis-states the fair price, most")
add("severely at long odds. Shin and power methods should be compared.")

# --------------------------------------------------- 8. SIGNAL TESTING
section("8. SIGNAL TESTING AGAINST THE MARKET RESIDUAL")
add("Target: (home win indicator) - (de-vigged opening market probability).")
add("A feature with predictive value beyond the market correlates with this.\n")
# What the market failed to predict. Correlating a feature with this residual,
# rather than with the outcome itself, tests for information the price has not
# already absorbed — the only kind that could be profitable.
resid = (d.FTR == "H").astype(float) - d.oH


def test(name, x, note=""):
    """Correlate one feature against the market residual and report significance.

    `x` is positional, not index-aligned: it is rebuilt with a fresh RangeIndex
    and masked with a boolean array, so it must arrive in the same row order as
    `resid`. Note `resid` is reassigned before the rolling-feature tests below.
    """
    x = pd.to_numeric(pd.Series(np.asarray(x, dtype=float)), errors="coerce")
    m = x.notna().values & np.isfinite(x.fillna(0)).values
    if m.sum() < 100:
        add(f"{name:<40}{'insufficient data':>28}")
        return
    c = np.corrcoef(x[m], resid[m])[0, 1]
    k = int(m.sum())
    # t-test on a correlation coefficient, k-2 degrees of freedom.
    t = c*np.sqrt((k-2)/(1-c**2))
    p = 2*(1 - st.t.cdf(abs(t), k-2))
    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    add(f"{name:<40}{k:>7}{c:>+9.4f}{p:>11.3g}  {star:<4}{note}")

add(f"{'Feature':<40}{'n':>7}{'corr':>9}{'p-value':>11}")
add("-"*78)

test("Line movement (close - open, home)", (d.cH - d.oH).values, "LEAKS if betting at open")
test("Asian handicap line (AHh)", d.AHh.values)
test("Bookmaker disagreement (sd of probs)",
     (1/d[[c for c in ["B365H","BFDH","BMGMH","BVH","BWH","CLH","LBH"]
           if c in d.columns]]).std(axis=1).values)
test("Number of bookmakers quoting",
     d[[c for c in ["B365H","BFDH","BMGMH","BVH","BWH","CLH","LBH"]
        if c in d.columns]].notna().sum(axis=1).values)
test("Overround (market margin)", d.overround.values)
test("Exchange minus bookmaker prob",
     (((1/d.BFEH)/(1/d.BFEH+1/d.BFED+1/d.BFEA)) - d.oH).values)
test("Market implied P(over 2.5)", (1/d["Avg>2.5"]).values if "Avg>2.5" in d else None)
test("Best-price premium (Max/Avg, home)", (d.MaxH/d.AvgH).values)
test("League tier (1 = top flight)", d.Tier.values)

# --- rolling form features -------------------------------------------------
# Team form needs a team-centric view, so each match is exploded into two rows
# (one per team) with goal difference signed from that team's perspective. After
# grouping by team the rolling mean is per-team; shift(1) excludes the current
# match, which is what keeps the feature free of look-ahead.
# Every match index `i` therefore appears exactly twice, and the home/away rows
# are recovered positionally below — hence the stable sort.
d2 = d.sort_values("Date").reset_index(drop=True)
long = pd.concat([
    pd.DataFrame({"i": d2.index, "L": d2.League, "T": d2.HomeTeam,
                  "gd": d2.FTHG-d2.FTAG, "sot": d2.HST, "date": d2.Date}),
    pd.DataFrame({"i": d2.index, "L": d2.League, "T": d2.AwayTeam,
                  "gd": d2.FTAG-d2.FTHG, "sot": d2.AST, "date": d2.Date}),
]).sort_values("i", kind="stable")   # stable: iloc[0::2]/[1::2] must stay home/away
gp = long.groupby(["L", "T"])
long["f_gd"] = gp.gd.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
long["f_sot"] = gp.sot.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
long["rest"] = gp.date.transform(lambda s: s.diff().dt.days)
# Pairs sit adjacent after the stable sort: even rows home, odd rows away.
# Re-indexing on `i` makes the H - A subtractions below align by match.
H = long.iloc[0::2].set_index("i"); A = long.iloc[1::2].set_index("i")

# Rebuilt for d2's chronological row order, which the features now follow.
resid = (d2.FTR == "H").astype(float) - d2.oH
test("Rolling goal-difference form (6 match)", (H.f_gd - A.f_gd).values)
test("Rolling shots-on-target form (6 match)", (H.f_sot - A.f_sot).values)
test("Rest-days differential", (H.rest - A.rest).values)

add("\n*** p<0.001   ** p<0.01   * p<0.05")
add("\nRead this table with care. Correlations of 0.03-0.07 explain a fraction")
add("of one percent of variance. They indicate the presence of signal, not its")
add("exploitability: a margin of 5-8% must be overcome before any of it is")
add("profitable. Line movement is the strongest by a wide margin but is")
add("observable only at kick-off, so it cannot be used as a feature when")
add("simulating bets placed at opening prices.")

print("\n".join(OUT))
with open("eda_part2.txt", "w") as f:
    f.write("\n".join(OUT))
# =============================================================================
# PART 3 — the threshold anomaly and figures (sections 9-10)
# Third standalone script; same pattern as Part 2. Builds an actual forecasting
# model, simulates betting against opening prices, and writes eda_figures.png.
# =============================================================================
"""
Exploratory Data Analysis — Part 3.
The threshold anomaly (motivating observation) and figures.
"""
# Plot styling, applied globally for the figures at the end of this part.
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": "#5A6472", "axes.labelcolor": "#1A2332",
    "text.color": "#1A2332", "xtick.color": "#5A6472", "ytick.color": "#5A6472",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.grid": True,
    "grid.color": "#E4E9ED", "grid.linewidth": 0.7,
})
ACC, DARK, RED, GREEN = "#2D6A8F", "#1A2332", "#9B3D3D", "#2E6F4E"

OUT = []
def section(t):
    OUT.append("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def add(t=""):
    OUT.append(str(t))

# df = pd.read_pickle("all.pkl")
df = load("csv_out")
df["Date"] = pd.to_datetime(df.Date)
d = df[df.FTR.isin(["H", "D", "A"])].dropna(
    subset=["B365H", "B365D", "B365A", "FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)
s = 1/d.B365H + 1/d.B365D + 1/d.B365A
d["oH"], d["oD"], d["oA"] = (1/d.B365H)/s, (1/d.B365D)/s, (1/d.B365A)/s
d["overround"] = s
y = d.FTR.map({"H": 0, "D": 1, "A": 2}).values.astype(int)

# ---- chronological Elo -------------------------------------------------
# One pass in date order. Each match is rated using only the ratings standing
# before it, then those ratings are updated — so elo_diff never sees its own
# result and needs no separate train/test handling.
# K = update rate, HFA = home advantage in rating points, teams start at 1500.
elo, K, HFA = {}, 20, 60
eh, ea = np.zeros(len(d)), np.zeros(len(d))
for i, r in enumerate(d.itertuples()):
    # Keyed by (league, team): promoted and relegated sides are treated as new.
    kh, ka = (r.League, r.HomeTeam), (r.League, r.AwayTeam)
    Rh, Ra = elo.get(kh, 1500.), elo.get(ka, 1500.)
    eh[i], ea[i] = Rh, Ra            # record pre-match ratings, then update
    e = 1/(1 + 10**(-((Rh + HFA) - Ra)/400))   # expected home score
    S = 1.0 if r.FTR == "H" else 0.5 if r.FTR == "D" else 0.0   # actual
    # Margin-of-victory multiplier, damped by log so routs do not dominate.
    m = np.log(max(abs(r.FTHG - r.FTAG), 1) + 1)
    elo[kh], elo[ka] = Rh + K*m*(S - e), Ra - K*m*(S - e)   # zero-sum update
d["elo_diff"] = (eh + HFA) - ea

long = pd.concat([
    pd.DataFrame({"i": d.index, "L": d.League, "T": d.HomeTeam, "gd": d.FTHG - d.FTAG}),
    pd.DataFrame({"i": d.index, "L": d.League, "T": d.AwayTeam, "gd": d.FTAG - d.FTHG}),
]).sort_values("i", kind="stable")   # stable: f[0::2]/f[1::2] must stay home/away
f = long.groupby(["L", "T"]).gd.transform(
    lambda x: x.shift(1).rolling(6, min_periods=1).mean()).fillna(0).values
d["form_diff"] = f[0::2] - f[1::2]

# Market probabilities are features, not just a benchmark: the model starts from
# the price and asks whether Elo and form justify moving away from it.
FEATS = ["elo_diff", "form_diff", "oH", "oD", "oA"]
# Chronological 70/30 split — never random, which would leak future form into
# past matches. Note the cut lands mid-day, so one date straddles both sets.
cut = int(len(d)*0.7)
tr, te = np.arange(cut), np.arange(cut, len(d))
# Features are left unscaled; elo_diff spans hundreds while oH/oD/oA span 0-1,
# so the default L2 penalty does not bear on them evenly.
X = np.nan_to_num(d[FEATS].values.astype(float))
model = LogisticRegression(max_iter=3000).fit(X[tr], y[tr])
P = model.predict_proba(X[te])                      # model probabilities
O = d[["B365H", "B365D", "B365A"]].values[te]       # prices actually available
yt = y[te]

# ------------------------------------------------ 9. THRESHOLD ANOMALY
section("9. THE THRESHOLD ANOMALY")
add(f"Model    : multinomial logistic on {FEATS}")
add(f"Training : {cut:,} matches, {d.Date[0].date()} to {d.Date[cut-1].date()}")
add(f"Test     : {len(te):,} matches, {d.Date[cut].date()} to {d.Date.iloc[-1].date()}")

def ll(Pm, yy):
    return log_loss(yy, np.clip(Pm, 1e-9, 1), labels=[0, 1, 2])
add(f"\nTest log loss, de-vigged market : {ll(d[['oH','oD','oA']].values[te], yt):.4f}")
add(f"Test log loss, model            : {ll(P, yt):.4f}")

# Expected value per unit staked. Positive means the model thinks the price is
# too long. Shape (matches, 3): every outcome of every test match is a candidate.
EV = P*O - 1
add("\nFlat-stake betting on the test period, opening Bet365 prices:\n")
add(f"{'Threshold':<12}{'Bets':>7}{'Hit rate':>10}{'ROI':>10}{'95% CI':>22}{'P(ROI<=0)':>12}")
rng = np.random.default_rng(0)
ths = [0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
curve = []
for th in ths:
    # ri = match row, ci = which outcome. A match can qualify on more than one
    # outcome, in which case it contributes several bets to the same tally.
    sel = EV >= th
    ri, ci = np.where(sel)
    if len(ri) < 5:
        add(f"{'EV >= '+format(th,'.3f'):<12}{len(ri):>7}   (too few)")
        continue
    won = (ci == yt[ri])
    pnl = np.where(won, O[ri, ci] - 1, -1.0)
    # Bootstrap the mean P/L to get an interval. Resamples individual bets, so
    # it assumes independence and ignores bets sharing a match.
    bs = np.array([rng.choice(pnl, len(pnl), replace=True).mean() for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])*100
    curve.append((th, len(pnl), pnl.mean()*100, lo, hi))
    add(f"{'EV >= '+format(th,'.3f'):<12}{len(pnl):>7}{won.mean()*100:>9.1f}%"
        f"{pnl.mean()*100:>9.2f}%   [{lo:>6.1f}, {hi:>6.1f}]{(bs<=0).mean():>12.2f}")

add("\nReturn does not improve as the threshold rises. Under the assumption that")
add("the filter selects genuine edge, tightening it should raise returns; the")
add("observed pattern is the opposite. This is the motivating observation for")
add("the project. The confidence intervals are wide on a single season, so the")
add("pattern is suggestive rather than established, and confirming it on a")
add("larger sample is the first task of the study.")

# ------------------------------------------------ 10. SHRINKAGE WEIGHT
section("10. THE SHRINKAGE WEIGHT")
add("Regressing the market residual on the model's deviation, test period:")
add("    (y_home - q_home) = w * (p_home - q_home) + noise\n")
# How much of the model's disagreement with the market is real? Regress what the
# market got wrong on how far the model moved from it. The slope w is the share
# of that move worth keeping: 0 = all noise, shrink fully; 1 = take it at face
# value. Home outcome only (class 0).
dev = P[:, 0] - d.oH.values[te]                      # model minus market
res = (yt == 0).astype(float) - d.oH.values[te]      # truth minus market
w, b = np.polyfit(dev, res, 1)
n = len(dev)
# Textbook standard error of an OLS slope: residual variance over spread in x.
se = np.sqrt(np.sum((res - (w*dev + b))**2)/(n-2) / np.sum((dev - dev.mean())**2))
add(f"  n                    : {n:,}")
add(f"  estimated w          : {w:+.4f}")
add(f"  standard error       : {se:.4f}")
add(f"  t statistic          : {w/se:+.2f}")
add(f"  mean |deviation|     : {np.abs(dev).mean():.4f}")
lo95, hi95 = w - 1.96*se, w + 1.96*se
add(f"  95% CI               : [{lo95:+.3f}, {hi95:+.3f}]")
add("")
add("Interpretation. w = 0 would mean the model's disagreements with the market")
add("are pure noise and should be shrunk away entirely. w = 1 would mean they")
add("should be taken at face value. The estimate here exceeds 1, which would")
add("imply the model is under-confident rather than over-confident.")
add("")
add("Three cautions before reading anything into that. The confidence interval")
add("is wide and comfortably contains 1, so no shrinkage and moderate shrinkage")
add("are both consistent with the data. The regressor has mean absolute value")
add(f"{np.abs(dev).mean():.4f}, so the slope is estimated across a very narrow")
add("range of x and is correspondingly unstable. And this is a single season")
add("with one train/test split.")
add("")
add("H3 is therefore weakly supported in sign but the magnitude is not usable.")
add("Estimating w reliably, on a larger sample and with a heteroscedastic")
add("specification, is a primary task of the study rather than a settled result.")

# ------------------------------------------------------------ FIGURES
# Four panels summarising sections 5, 7, 9 and 4, written to eda_figures.png.
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

# (a) market quality vs margin — one point per division, recomputing section 5's
#     margin/log-loss pair so the panel stands alone.
ax = axes[0, 0]
rows = []
for L, g in d.groupby("League"):
    yy = g.FTR.map({"H": 0, "D": 1, "A": 2}).values.astype(int)
    rows.append((L, (g.overround.mean()-1)*100,
                 ll(g[["oH", "oD", "oA"]].values, yy), len(g)))
R = pd.DataFrame(rows, columns=["L", "margin", "ll", "n"])
ax.scatter(R.margin, R.ll, s=28, c=ACC, alpha=.8, edgecolors="white", linewidth=1)
for _, r in R.iterrows():
    ax.annotate(r.L, (r.margin, r.ll), fontsize=6.5, ha="left", va="bottom",
                xytext=(3.5, 2.5), textcoords="offset points", color=DARK)
z = np.polyfit(R.margin, R.ll, 1)   # trend line through the divisions
xs = np.linspace(R.margin.min(), R.margin.max(), 50)
ax.plot(xs, np.polyval(z, xs), "--", c=RED, lw=1.3)
ax.set_xlabel("Bookmaker margin (%)")
ax.set_ylabel("Market log loss  (higher = worse forecast)")
ax.set_title(f"Soft markets charge more   (r = {np.corrcoef(R.margin, R.ll)[0,1]:+.2f})",
             fontsize=10, weight="bold", loc="left")

# (b) favourite-longshot bias — calibration curve against the diagonal; marker
#     area scales with bin count, so sparse long-odds bins read as less certain.
ax = axes[0, 1]
rows = []
for odds, hit in [(d.B365H, d.FTR == "H"), (d.B365D, d.FTR == "D"), (d.B365A, d.FTR == "A")]:
    rows.append(pd.DataFrame({"p": (1/odds)/d.overround, "w": hit.astype(float)}))
b = pd.concat(rows, ignore_index=True)
b["bin"] = pd.cut(b.p, [0, .05, .1, .2, .3, .4, .5, .6, .7, 1.0])
gb = b.groupby("bin", observed=True).agg(p=("p", "mean"), w=("w", "mean"), n=("w", "size"))
ax.plot([0, .85], [0, .85], "--", c="#9AA5AF", lw=1.2, label="perfect calibration")
ax.scatter(gb.p, gb.w, s=np.sqrt(gb.n)*4, c=ACC, zorder=3,
           edgecolors="white", linewidth=1.2)
ax.plot(gb.p, gb.w, c=ACC, lw=1.4, zorder=2)
ax.set_xlabel("De-vigged implied probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Favourite-longshot bias", fontsize=10, weight="bold", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")

# (c) THE anomaly — ROI against selection threshold, the project's motivating
#     observation. Tuple layout is (threshold, n bets, ROI, CI low, CI high).
ax = axes[1, 0]
plot_curve = [c for c in curve if c[1] >= 25]   # drop thresholds too thin to read
cx = [c[0]*100 for c in plot_curve]; cy = [c[2] for c in plot_curve]
clo = [c[3] for c in plot_curve]; chi = [c[4] for c in plot_curve]
ax.fill_between(cx, clo, chi, color=ACC, alpha=.13, label="95% CI")
ax.plot(cx, cy, "o-", c=ACC, lw=1.8, ms=5, zorder=3, label="realised ROI")
ax.axhline(0, c="#9AA5AF", lw=1, ls="--")
for (t_, n_, r_, _, _), xx, yy2 in zip(plot_curve, cx, cy):
    ax.annotate(f"n={n_}", (xx, yy2), textcoords="offset points",
                xytext=(0, -13), fontsize=6.5, ha="center", color="#5A6472")
ax.set_xlabel("Selection threshold  $\\tau$  (%)")
ax.set_ylabel("Return on investment (%)")
ax.set_title("Return does not improve with selectivity", fontsize=10,
             weight="bold", loc="left")
ax.set_ylim(min(clo)-8, max(chi)+8)
ax.legend(frameon=False, fontsize=8)

# (d) outcome mix by division — stacked to 100%, sorted by home-win share to
#     show how much home advantage varies while the draw band stays flat.
ax = axes[1, 1]
hw = d.groupby("League").apply(lambda g: pd.Series({
    "home": (g.FTR == "H").mean()*100, "draw": (g.FTR == "D").mean()*100,
    "away": (g.FTR == "A").mean()*100})).sort_values("home")
yp = np.arange(len(hw))
ax.barh(yp, hw.home, color=ACC, label="home win")
ax.barh(yp, hw.draw, left=hw.home, color="#9AA5AF", label="draw")
ax.barh(yp, hw.away, left=hw.home + hw.draw, color=RED, alpha=.75, label="away win")
ax.set_yticks(yp); ax.set_yticklabels(hw.index, fontsize=7)
ax.set_xlabel("Share of matches (%)"); ax.set_xlim(0, 100)
ax.grid(axis="y", visible=False)
ax.set_title("Outcome mix by division", fontsize=10, weight="bold", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
          bbox_to_anchor=(.5, -.22))

plt.tight_layout(pad=1.6)
plt.savefig("eda_figures.png", dpi=190, bbox_inches="tight")
add("\n\nFigures written to eda_figures.png")

print("\n".join(OUT))
with open("eda_part3.txt", "w") as f:
    f.write("\n".join(OUT))