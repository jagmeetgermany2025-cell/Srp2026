"""Phase B — threshold sweep on the full corpus, and the item 6 decision.

Items 5 and 6. Re-runs EDA.py's section 9 sweep on the multi-season corpus and
decides whether the motivating anomaly survives the larger sample.

The point is comparability, so the model, features and thresholds are the ones
EDA.py used. Three deviations, all strictly-more-correct and none of which
changes the estimator:

  * the train/test cut lands on a date boundary instead of mid-day (EDA.py had
    31 matches from one round straddling the split)
  * the bootstrap resamples MATCHES, not bets -- bets on the same fixture are
    not independent and resampling bets understates the interval
  * form is recovered with a `side` column instead of iloc[0::2], with an
    equivalence check against the old slicing

Everything else is deliberately unchanged. Walk-forward splits, calibration and
shrinkage belong to later phases; mixing them in here would make it impossible
to tell whether a change in the curve came from more data or from the changes.

    python phase_b.py

Reads matches_multiseason.parquet beside this module. Writes
phase_b_report.txt and phase_b_threshold_sweep.png.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")   # headless, must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

pd.set_option("display.width", 200)

THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]   # EDA.py's grid
N_BOOT = 2000
SEED = 0
MIN_BETS = 200          # thresholds thinner than this are not worth a verdict
FEATS = ["elo_diff", "form_diff", "oH", "oD", "oA"]
ODDS = ["B365H", "B365D", "B365A"]

REPORT = "phase_b_report.txt"
FIGURE = "phase_b_threshold_sweep.png"

OUT = []


def section(title):
    line = "=" * 78
    OUT.append(f"\n{line}\n{title}\n{line}")


def add(text=""):
    OUT.append(str(text))


def table(df):
    OUT.append(df.to_string())


# ============================================================== metrics =====

def ll(P, yy):
    return log_loss(yy, np.clip(P, 1e-9, 1), labels=[0, 1, 2])


def rps(P, yy):
    """Ranked probability score: respects the H < D < A ordering."""
    Y = np.eye(3)[yy]
    return np.mean(np.sum((np.cumsum(P, 1) - np.cumsum(Y, 1))[:, :2] ** 2, 1) / 2)


def spearman(x, y):
    rx = x.argsort().argsort() + 1.0
    ry = y.argsort().argsort() + 1.0
    a, b = rx - rx.mean(), ry - ry.mean()
    return (a @ b) / np.sqrt((a ** 2).sum() * (b ** 2).sum())


# =============================================================== data =======

def load_analysis_set(parquet):
    section("0. ANALYSIS SET")
    df = pd.read_parquet(parquet)
    add(f"{len(df):,} matches, {df.Div.nunique()} divisions, "
        f"{df.Date.min().date()} to {df.Date.max().date()}")

    d = df[df.FTR.isin(["H", "D", "A"])].dropna(subset=ODDS + ["FTHG", "FTAG"]).copy()

    # dropna misses these: older files write 0 for "no price" instead of leaving
    # the cell blank, and 1/0 -> inf -> NaN after normalising. A negative or
    # sub-1 price is worse, since it yields a finite but nonsense probability
    # that never errors. Decimal odds must exceed 1.
    valid = (d[ODDS] > 1.0).all(axis=1)
    if (~valid).any():
        add(f"\ndropping {(~valid).sum():,} matches with invalid odds (<= 1.0), by season:")
        add(d.loc[~valid, "Season"].value_counts().sort_index().to_string())
    d = d[valid].sort_values("Date").reset_index(drop=True)

    s = 1 / d.B365H + 1 / d.B365D + 1 / d.B365A
    d["oH"], d["oD"], d["oA"] = (1 / d.B365H) / s, (1 / d.B365D) / s, (1 / d.B365A) / s
    d["overround"] = s
    assert np.isfinite(d[["oH", "oD", "oA"]].values).all(), \
        "non-finite de-vigged probabilities"

    y = d.FTR.map({"H": 0, "D": 1, "A": 2}).values.astype(int)
    add(f"\nanalysis set: {len(d):,} matches")
    add(f"mean overround: {(d.overround.mean() - 1) * 100:.2f}%")
    return d, y


# ==================================================== 1. MARKET BASELINE ====

def market_baseline(d, y):
    section("1. MARKET BASELINE")
    add("Sanity gate. The de-vigged market should score a log loss near 1.00 on")
    add("1X2; far from that means the de-vigging or the outcome coding is wrong")
    add("and every later figure would be wrong with it.\n")
    Q = d[["oH", "oD", "oA"]].values
    add(f"market log loss : {ll(Q, y):.4f}   (want ~1.00)")
    add(f"market RPS      : {rps(Q, y):.4f}")
    add(f"outcome mix     : H {(y == 0).mean() * 100:.1f}%  "
        f"D {(y == 1).mean() * 100:.1f}%  A {(y == 2).mean() * 100:.1f}%")

    rows = []
    for season, g in d.groupby("Season"):
        yy = g.FTR.map({"H": 0, "D": 1, "A": 2}).values.astype(int)
        P = g[["oH", "oD", "oA"]].values
        rows.append((season, len(g), (g.overround.mean() - 1) * 100, ll(P, yy), rps(P, yy)))
    add("\nper season, to see whether the market's own quality drifts:\n")
    table(pd.DataFrame(rows, columns=["Season", "n", "Margin%", "LogLoss", "RPS"]).round(4))


# ============================================ 2. FAVOURITE-LONGSHOT BIAS ====

def longshot_bias(d):
    section("2. FAVOURITE-LONGSHOT BIAS")
    add("EDA.py measured 0.113 priced against 0.076 observed at the long end on")
    add("one season -- ratio 0.68. Item 16 uses that as the acceptance test for")
    add("the de-vigging comparison, so re-measure it here before building on it.\n")
    rows = []
    for odds, hit in [(d.B365H, d.FTR == "H"), (d.B365D, d.FTR == "D"),
                      (d.B365A, d.FTR == "A")]:
        rows.append(pd.DataFrame({"p": (1 / odds) / d.overround, "o": odds,
                                  "w": hit.astype(float)}))
    b = pd.concat(rows, ignore_index=True)
    b["bin"] = pd.cut(b.p, [0, .05, .10, .20, .30, .40, .50, .60, .70, 1.0])

    tbl = b.groupby("bin", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "Predicted": g.p.mean(), "Observed": g.w.mean(),
        "Ratio": g.w.mean() / g.p.mean(),
        "FlatROI%": np.where(g.w > 0, g.o - 1, -1).mean() * 100,
    }))
    table(tbl.round(3))

    lo, hi = b[b.p < 0.15], b[b.p > 0.60]
    add(f"\nlongshots  (p<0.15): priced {lo.p.mean():.3f}, won {lo.w.mean():.3f}, "
        f"ratio {lo.w.mean() / lo.p.mean():.2f}")
    add(f"favourites (p>0.60): priced {hi.p.mean():.3f}, won {hi.w.mean():.3f}, "
        f"ratio {hi.w.mean() / hi.p.mean():.2f}")


# ========================================================== 3. FEATURES =====

def build_features(d):
    section("3. FEATURES")
    add("Chronological Elo and 6-match form, exactly as EDA.py built them.")
    add("Ratings are recorded before the update, so a match never informs its")
    add("own feature.\n")

    elo, K, HFA = {}, 20, 60
    eh, ea = np.zeros(len(d)), np.zeros(len(d))
    for i, r in enumerate(d.itertuples()):
        kh, ka = (r.Div, r.HomeTeam), (r.Div, r.AwayTeam)
        Rh, Ra = elo.get(kh, 1500.), elo.get(ka, 1500.)
        eh[i], ea[i] = Rh, Ra                       # before the update
        e = 1 / (1 + 10 ** (-((Rh + HFA) - Ra) / 400))
        S = 1.0 if r.FTR == "H" else 0.5 if r.FTR == "D" else 0.0
        m = np.log(max(abs(r.FTHG - r.FTAG), 1) + 1)
        elo[kh], elo[ka] = Rh + K * m * (S - e), Ra - K * m * (S - e)
    d["elo_diff"] = (eh + HFA) - ea
    add(f"elo_diff: mean {d.elo_diff.mean():.1f}, sd {d.elo_diff.std():.1f}, "
        f"range [{d.elo_diff.min():.0f}, {d.elo_diff.max():.0f}]")

    # side column rather than iloc[0::2] -- positional recovery depends on the
    # sort staying stable and silently swaps home/away if it doesn't
    long = pd.concat([
        pd.DataFrame({"i": d.index, "L": d.Div, "T": d.HomeTeam,
                      "gd": d.FTHG - d.FTAG, "side": "H"}),
        pd.DataFrame({"i": d.index, "L": d.Div, "T": d.AwayTeam,
                      "gd": d.FTAG - d.FTHG, "side": "A"}),
    ]).sort_values("i", kind="stable")
    long["f"] = (long.groupby(["L", "T"]).gd
                 .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
                 .fillna(0))
    H = long[long.side == "H"].set_index("i").f
    A = long[long.side == "A"].set_index("i").f
    d["form_diff"] = (H - A).reindex(d.index).values

    f = long.f.values
    agree = np.allclose(f[0::2] - f[1::2], d.form_diff.values)
    add(f"positional slicing agrees with the side filter: {agree}")
    if not agree:
        n = int((~np.isclose(f[0::2] - f[1::2], d.form_diff.values)).sum())
        add(f"  {n:,} of {len(d):,} matches differ -- EDA.py's form_diff was wrong")
    return d


# ==================================================== 4. MODEL AND SPLIT ====

def fit_model(d, y):
    section("4. MODEL AND SPLIT")
    add("Same features and model as EDA.py. The cut is moved to a date boundary")
    add("so no round straddles train and test.\n")

    cut = int(len(d) * 0.7)
    cut = int((d.Date < d.Date.iloc[cut]).sum())        # snap to the boundary
    tr, te = np.arange(cut), np.arange(cut, len(d))

    X = np.nan_to_num(d[FEATS].values.astype(float))
    model = LogisticRegression(max_iter=3000).fit(X[tr], y[tr])
    P = model.predict_proba(X[te])
    O = d[ODDS].values[te]
    yt = y[te]
    Qte = d[["oH", "oD", "oA"]].values[te]

    add(f"train {cut:,} matches, {d.Date.iloc[0].date()} to {d.Date.iloc[cut - 1].date()}")
    add(f"test  {len(te):,} matches, {d.Date.iloc[cut].date()} to {d.Date.iloc[-1].date()}")
    add(f"straddling dates: {len(set(d.Date.iloc[tr]) & set(d.Date.iloc[te]))} (want 0)")
    add("")
    add(f"test log loss, market : {ll(Qte, yt):.4f}")
    add(f"test log loss, model  : {ll(P, yt):.4f}")
    add(f"test RPS,      market : {rps(Qte, yt):.4f}")
    add(f"test RPS,      model  : {rps(P, yt):.4f}")
    return P, O, yt, Qte, te


# ================================================== 5. THRESHOLD SWEEP ======

def sweep(EV, O, yt, one_per_match, n_boot=N_BOOT, seed=SEED):
    """Returns (table, bootstrap replicates).

    The bootstrap resamples fixtures and reuses the same resampled fixtures
    across every threshold, so differences between thresholds are paired.
    """
    n = len(yt)
    S = np.zeros((len(THRESHOLDS), n))     # per-match P&L sum, per threshold
    C = np.zeros((len(THRESHOLDS), n))     # per-match bet count
    rows = []

    for k, th in enumerate(THRESHOLDS):
        sel = EV >= th
        if one_per_match:
            ri = np.where(sel.any(1))[0]
            ci = (np.argmax(np.where(sel, EV, -np.inf)[ri], axis=1)
                  if len(ri) else np.array([], int))
        else:
            ri, ci = np.where(sel)
        if len(ri) == 0:
            rows.append((th, 0, np.nan, np.nan))
            continue
        won = ci == yt[ri]
        pnl = np.where(won, O[ri, ci] - 1, -1.0)
        S[k] = np.bincount(ri, weights=pnl, minlength=n)
        C[k] = np.bincount(ri, minlength=n)
        rows.append((th, len(ri), won.mean() * 100, pnl.mean() * 100))

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(THRESHOLDS)))
    for bnum in range(n_boot):
        idx = rng.integers(0, n, n)
        num, den = S[:, idx].sum(1), C[:, idx].sum(1)
        boot[bnum] = np.where(den > 0, num / np.maximum(den, 1) * 100, np.nan)

    out = pd.DataFrame(rows, columns=["tau", "Bets", "HitRate%", "ROI%"])
    out["CIlow"] = np.nanpercentile(boot, 2.5, axis=0)
    out["CIhigh"] = np.nanpercentile(boot, 97.5, axis=0)
    out["P(ROI<=0)"] = np.nanmean(boot <= 0, axis=0)
    return out.round(3), boot


# ======================================================= 6. THE DECISION ====

def decision(sweep_df, boot, label):
    """Item 6.

    H1 says ROI is *decreasing* in tau -- a monotonicity claim, not a linearity
    one, so the primary test is a bootstrapped Spearman correlation across
    every usable threshold. A linear slope is reported as a magnitude only:
    fitting a straight line through a concave curve understates the decline,
    and a noisy upturn at the last threshold inflates its variance enough to
    drag the interval over zero. Comparing the two endpoints is worse still --
    the tightest threshold carries the fewest bets.
    """
    k = sweep_df.index[sweep_df.Bets >= MIN_BETS].values
    if len(k) < 3:
        add(f"{label}: fewer than three thresholds clear {MIN_BETS} bets")
        return
    taus, roi = sweep_df.tau.values[k], sweep_df["ROI%"].values[k]
    B = boot[:, k]
    B = B[~np.isnan(B).any(1)]

    def ranks(M):
        order = M.argsort(1)
        r = np.empty_like(order)
        np.put_along_axis(r, order, np.arange(M.shape[1]), axis=1)
        return r + 1.0

    rb = ranks(B) - (len(taus) + 1) / 2
    xr = (taus.argsort().argsort() + 1.0) - (len(taus) + 1) / 2
    rho = (rb @ xr) / np.sqrt((rb ** 2).sum(1) * (xr ** 2).sum())
    r_lo, r_hi = np.percentile(rho, [2.5, 97.5])

    x = taus - taus.mean()
    slope = (B - B.mean(1, keepdims=True)) @ x / (x @ x)
    s_lo, s_hi = np.percentile(slope, [2.5, 97.5])

    add(f"--- {label} ---")
    add(f"thresholds used : {', '.join(f'{t:.3f}' for t in taus)}")
    add(f"Spearman rho    : {spearman(taus, roi):+.3f}   "
        f"95% CI [{r_lo:+.3f}, {r_hi:+.3f}]   P(rho<0) = {(rho < 0).mean():.3f}")
    add(f"linear slope    : {np.polyfit(taus, roi, 1)[0] / 100:+.2f} ROI% per 1pp   "
        f"95% CI [{s_lo / 100:+.2f}, {s_hi / 100:+.2f}]   (magnitude only)")
    add(f"monotone to the minimum: "
        f"{bool(np.all(np.diff(roi[:int(np.argmin(roi)) + 1]) < 0))}")

    if r_hi < 0:
        add("=> decline confirmed at this sample size")
    elif r_lo > 0:
        add("=> return RISES with selectivity -- the opposite of H1")
    else:
        add("=> rho interval spans zero; read the per-threshold table below")

    # where the gap against tau=0 first excludes zero. More useful than any
    # single number, and independent of functional form.
    gap = B[:, [0]] - B
    rows = []
    for j in range(1, len(taus)):
        l, h = np.percentile(gap[:, j], [2.5, 97.5])
        rows.append((taus[j], roi[0] - roi[j], l, h, "yes" if l > 0 else "no"))
    add("")
    table(pd.DataFrame(rows, columns=["tau", "ROI(0)-ROI(tau)", "CIlow",
                                      "CIhigh", "excludes 0"]).round(2))
    add("")


# ========================================================= 7. ROBUSTNESS ====

def era_split(EV, O, yt):
    section("7. DOES IT HOLD IN BOTH HALVES?")
    add("If the decline is a property of the selection rule it should appear in")
    add("both halves. If only in one, it is an era effect and the single-season")
    add("result was picking up whichever era it sat in.\n")
    mid = len(yt) // 2
    rows = []
    for name, sub in {"first half": np.arange(mid),
                      "second half": np.arange(mid, len(yt))}.items():
        for th in (0.0, 0.05, 0.10):
            ri, ci = np.where(EV[sub] >= th)
            if len(ri) < 50:
                rows.append((name, th, len(ri), np.nan))
                continue
            won = ci == yt[sub][ri]
            pnl = np.where(won, O[sub][ri, ci] - 1, -1.0)
            rows.append((name, th, len(ri), pnl.mean() * 100))
    table(pd.DataFrame(rows, columns=["Half", "tau", "Bets", "ROI%"]).round(2))


def shrinkage_weight(P, Qte, yt):
    section("8. SHRINKAGE WEIGHT (H3)")
    add("EDA.py estimated w on one season and got a wide interval containing 1,")
    add("which it called 'not usable'. Same regression on the full test set.\n")
    dev = P[:, 0] - Qte[:, 0]                    # model minus market, home
    res = (yt == 0).astype(float) - Qte[:, 0]    # truth minus market, home
    w, b0 = np.polyfit(dev, res, 1)
    n = len(dev)
    se = np.sqrt(np.sum((res - (w * dev + b0)) ** 2) / (n - 2)
                 / np.sum((dev - dev.mean()) ** 2))
    add(f"n                : {n:,}")
    add(f"estimated w      : {w:+.4f}")
    add(f"standard error   : {se:.4f}")
    add(f"t                : {w / se:+.2f}")
    add(f"95% CI           : [{w - 1.96 * se:+.3f}, {w + 1.96 * se:+.3f}]")
    add(f"mean |deviation| : {np.abs(dev).mean():.4f}   (EDA.py: 0.019)")
    add("")
    add("w=0: the model's disagreements are noise, shrink them away entirely.")
    add("w=1: take them at face value.")


# ============================================================= 9. FIGURE ====

def figure(sweep_all, sweep_one, n_test):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": "#5A6472", "axes.labelcolor": "#1A2332",
        "text.color": "#1A2332", "xtick.color": "#5A6472", "ytick.color": "#5A6472",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.grid": True,
        "grid.color": "#E4E9ED", "grid.linewidth": 0.7,
    })
    ACC, RED = "#2D6A8F", "#9B3D3D"

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for sw, colour, label in [(sweep_all, ACC, "all qualifying outcomes"),
                              (sweep_one, RED, "one bet per match")]:
        ok = sw[sw.Bets >= MIN_BETS]
        ax.fill_between(ok.tau * 100, ok.CIlow, ok.CIhigh, color=colour, alpha=.12)
        ax.plot(ok.tau * 100, ok["ROI%"], "o-", c=colour, lw=1.8, ms=5, label=label)
        for _, r in ok.iterrows():
            ax.annotate(f"n={int(r.Bets):,}", (r.tau * 100, r["ROI%"]),
                        textcoords="offset points", xytext=(0, -14),
                        fontsize=6.5, ha="center", color="#5A6472")

    ax.axhline(0, c="#9AA5AF", lw=1, ls="--")
    ax.set_xlabel("Selection threshold  $\\tau$  (%)")
    ax.set_ylabel("Return on investment (%)")
    ax.set_title(f"ROI against selectivity, {n_test:,} test matches",
                 fontsize=10, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE, dpi=190, bbox_inches="tight")
    add(f"\nfigure written to {FIGURE}")


# ============================================================== main ========

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="matches_multiseason.parquet")
    args = ap.parse_args()

    d, y = load_analysis_set(args.parquet)
    market_baseline(d, y)
    longshot_bias(d)
    d = build_features(d)
    P, O, yt, Qte, te = fit_model(d, y)

    section("5. THRESHOLD SWEEP")
    add("`all` takes every outcome clearing the threshold, which is what EDA.py")
    add("did and lets one fixture contribute several bets. `one` keeps only the")
    add("highest-EV outcome per match, the coherent strategy. Both are reported")
    add("so it is visible whether the pattern depends on that choice.\n")
    EV = P * O - 1
    sweep_all, boot_all = sweep(EV, O, yt, one_per_match=False)
    sweep_one, boot_one = sweep(EV, O, yt, one_per_match=True)
    add("all qualifying outcomes (EDA.py's rule):")
    table(sweep_all)
    add("\none bet per match:")
    table(sweep_one)

    section("6. THE DECISION (item 6)")
    decision(sweep_all, boot_all, "all qualifying outcomes")
    decision(sweep_one, boot_one, "one bet per match")

    era_split(EV, O, yt)
    shrinkage_weight(P, Qte, yt)
    figure(sweep_all, sweep_one, len(te))

    text = "\n".join(OUT)
    print(text)
    with open(REPORT, "w") as fh:
        fh.write(text)
    print(f"\nreport written to {REPORT}")


if __name__ == "__main__":
    main()
