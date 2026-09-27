"""
Run the two-stage pipeline under many seeds and ask what was luck.

One seed gives one number. Whether that number is the pipeline or the draw
cannot be read off it, so this runs the whole walk-forward N times, changing
nothing but the seed, and reports distributions instead of points:

  1. THE WEIGHT. w and w_band for every seed and every fold. Is w zero every
     time, or does it move? Does the weight fitted only where the strategy
     bets (w_band) disagree with the one fitted on every match (w)?

  2. THE TAIL. Log loss on the betting band alone -- the 10% of test matches
     where stage 1 disagrees most with the market. Averaged over every match,
     log loss is dominated by games no rule would bet, so a small edge that
     lives only in the tail can hide there. This looks for it directly.

  3. LUCK, three kinds:
       seed      -- the spread of the operating-point ROI across seeds
       sample    -- a bootstrap over whole WEEKS, since matches on one
                    weekend share news and move together
       selection -- the share of odds-matched random draws each seed beats
     and how much of each seed's profit came from its single best fold.

Nothing in the pipeline changes. Each seed calls two_stage_torch.run() as is,
with the stake model switched off (it does not affect selection) and w(x)
optional because it is the slowest piece.

    python seed_sweep.py --seeds 100 --learner catboost --ensemble-k 5
"""
import argparse
import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import two_stage_torch as ts                     # noqa: E402  (sets OpenMP env)

import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402
from joblib import Parallel, delayed             # noqa: E402
from sklearn.metrics import log_loss             # noqa: E402

COVERAGE = 0.10
ARMS = ("p_indep", "p_corr", "p_band", "p_js", "p_cal")


def week_bootstrap(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """ROI interval resampling whole weeks rather than single matches."""
    if bets.empty:
        return {"wk_lo": np.nan, "wk_hi": np.nan, "wk_p_le_zero": np.nan}
    week = pd.to_datetime(bets["Date"]).dt.to_period("W").astype(str)
    g = bets.groupby(week.values)
    stake, pnl = g["Stake"].sum().to_numpy(), g["PnL"].sum().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(stake), (n_boot, len(stake)))
    roi = pnl[idx].sum(1) / stake[idx].sum(1) * 100
    return {"wk_lo": float(np.percentile(roi, 2.5)),
            "wk_hi": float(np.percentile(roi, 97.5)),
            "wk_p_le_zero": float((roi <= 0).mean())}


def band_log_loss(te: pd.DataFrame) -> dict:
    """Log loss on the whole test unit and on its betting band, per arm."""
    y = te["target"].astype(int).to_numpy()
    p_ref, p_ind = ts.probs(te, "p_ref"), ts.probs(te, "p_indep")
    band = ts.selection_band(p_ref, p_ind, coverage=COVERAGE)
    out = {"n_band": int(band.sum())}
    for name, prefix in (("market", "p_ref"), ("stage1", "p_indep"),
                         ("corr", "p_corr"), ("band", "p_band")):
        p = ts.probs(te, prefix)
        out[f"ll_all_{name}"] = log_loss(y, p, labels=[0, 1, 2])
        out[f"ll_band_{name}"] = log_loss(y[band], p[band], labels=[0, 1, 2])
    return out


def roi_of(bets: pd.DataFrame) -> float:
    stake = bets["Stake"].sum() if len(bets) else 0.0
    return float(bets["PnL"].sum() / stake * 100) if stake > 0 else np.nan


def one_seed(df: pd.DataFrame, seed: int, learner: str, k: int,
             n_blocks: int, learn_w: bool) -> tuple:
    """The full walk-forward under one seed, reduced to what the sweep needs."""
    ts.configure(learner=learner, ensemble_k=k)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, folds, _ = ts.run(df, n_blocks=n_blocks, seed=seed,
                                 learn_w=learn_w, learn_stakes=False)
    unit_col = "Block" if "Block" in preds.columns else "Season"

    # The operating point, chosen exactly as main() chooses it: the corrected
    # arm while it can still rank anything, the uncorrected one once it cannot.
    op_prefix = ("p_corr" if ts.rankable(preds, "p_corr", ts.OP_CRITERION)
                 else "p_indep")
    tau_col = "op_tau" if op_prefix == "p_corr" else "op_tau_indep"

    fold_rows, op_frames = [], []
    for _, f in folds.iterrows():
        te = preds[preds[unit_col] == f["test_unit"]]
        bets = ts.select_bets(te, f[tau_col], op_prefix, criterion=ts.OP_CRITERION)
        op_frames.append(bets.assign(fold=f["test_unit"]))
        fold_rows.append({
            "seed": seed, "fold": f["test_unit"],
            "w": f["w"], "w_band": f["w_band"], "c_js": f["c_js"],
            **band_log_loss(te),
            "op_bets": len(bets), "op_pnl": float(bets["PnL"].sum()) if len(bets) else 0.0,
            "op_stake": float(bets["Stake"].sum()) if len(bets) else 0.0,
            "op_roi": roi_of(bets)})
    op_bets = pd.concat(op_frames, ignore_index=True).assign(seed=seed)

    fold_pnl = pd.Series([r["op_pnl"] for r in fold_rows])
    total = fold_pnl.sum()
    rnd = ts.odds_matched_random(preds, op_bets, n_draws=300, seed=seed)

    summary = {"seed": seed, "learner": learner, "ensemble_k": k,
               "op_arm": op_prefix, "op_bets": len(op_bets),
               "op_roi": roi_of(op_bets),
               **week_bootstrap(op_bets, seed=seed),
               "random_mean": rnd.get("random_mean", np.nan),
               "beats_random": rnd.get("model_beats_random", np.nan),
               # Above 1 means one fold made more than the whole run did.
               "best_fold_share": float(fold_pnl.max() / total) if total > 0 else np.nan,
               "w_median": float(folds["w"].median()),
               "w_band_median": float(folds["w_band"].median()),
               "seconds": time.time() - t0}

    # Every arm at matched coverage, thresholded on the pooled test scores the
    # way betting_report() does it. Comparable across arms; not a strategy.
    for prefix in ARMS:
        if ts.has_arm(preds, prefix) and ts.rankable(preds, prefix, "edge"):
            tau = float(np.quantile(ts.score_pool(preds, prefix, criterion="edge"),
                                    1.0 - COVERAGE))
            summary[f"roi10_{prefix}"] = roi_of(
                ts.select_bets(preds, tau, prefix, criterion="edge"))
        else:
            summary[f"roi10_{prefix}"] = np.nan        # collapsed onto the market

    keep = ["seed", "fold", "match_key", "Date", "Div", "Selection", "Odds",
            "p_est", "p_ref", "Score", "IsWin", "Stake", "PnL"]
    return fold_rows, summary, op_bets[[c for c in keep if c in op_bets.columns]]


# -------------------------------------------------------------------------
def q(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "n/a"
    return (f"{s.median():+.3f}  [{s.quantile(0.05):+.3f}, "
            f"{s.quantile(0.95):+.3f}]")


def report(fold_df: pd.DataFrame, seed_df: pd.DataFrame) -> str:
    out = []
    say = out.append
    n_seeds = seed_df["seed"].nunique()
    say("=" * 72)
    say(f"SEED SWEEP  {n_seeds} seeds x {fold_df['fold'].nunique()} folds   "
        f"learner={seed_df['learner'].iloc[0]}  "
        f"ensemble_k={seed_df['ensemble_k'].iloc[0]}")
    say("Figures are median [5th, 95th percentile] unless marked otherwise.")
    say("=" * 72)

    # 1. the weight -------------------------------------------------------
    say("\n1. THE SHRINKAGE WEIGHT  (0 = trust the market, 1 = trust stage 1)")
    for col, label in (("w", "w, fitted on every match"),
                       ("w_band", "w_band, fitted on the betting band only")):
        s = fold_df[col]
        say(f"  {label}")
        say(f"     all seed x fold : {q(s)}   mean {s.mean():.3f}")
        say(f"     share < 0.05    : {(s < ts.W_COLLAPSE_LEVEL).mean():.0%}   "
            f"share > 0.5 : {(s > ts.W_WARNING_LEVEL).mean():.0%}")
    gap = fold_df["w_band"] - fold_df["w"]
    say(f"  w_band - w        : {q(gap)}   w_band higher in "
        f"{(gap > 0.05).mean():.0%} of seed x fold")

    per_fold = fold_df.groupby("fold").agg(
        w_med=("w", "median"), w_p5=("w", lambda s: s.quantile(0.05)),
        w_p95=("w", lambda s: s.quantile(0.95)),
        wb_med=("w_band", "median"), wb_p5=("w_band", lambda s: s.quantile(0.05)),
        wb_p95=("w_band", lambda s: s.quantile(0.95)),
        roi_med=("op_roi", "median"))
    say("\n  By fold (across seeds):")
    say(per_fold.round(3).to_string())

    # 2. the tail ---------------------------------------------------------
    say("\n2. WHERE THE STRATEGY BETS  (log loss, lower is better; mean over "
        "seed x fold)")
    for scope in ("all", "band"):
        row = {n: fold_df[f"ll_{scope}_{n}"].mean()
               for n in ("market", "stage1", "corr", "band")}
        say(f"  {'every match' if scope == 'all' else 'betting band':13s}"
            + "  ".join(f"{n} {v:.4f}" for n, v in row.items()))
    d = fold_df["ll_band_stage1"] - fold_df["ll_band_market"]
    db = fold_df["ll_band_band"] - fold_df["ll_band_market"]
    say(f"  stage 1 minus market, band : {q(d)}   stage 1 better in "
        f"{(d < -1e-4).mean():.0%}")
    say(f"  w_band arm minus market    : {q(db)}   better in {(db < -1e-4).mean():.0%}")
    say("  A negative number would be an edge the all-match log loss hides.")

    # 3. luck ---------------------------------------------------------------
    say("\n3. LUCK CHECKS  (operating point: top 10% by edge, tau from validation)")
    arms = seed_df["op_arm"].value_counts().to_dict()
    say(f"  arm the operating point ran on : {arms}")
    r = seed_df["op_roi"]
    say(f"  seed    ROI across seeds       : {q(r)}  (min {r.min():+.2f}, "
        f"max {r.max():+.2f})")
    say(f"          seeds with ROI > 0     : {(r > 0).mean():.0%}")
    if 42 in set(seed_df["seed"]):
        r42 = float(seed_df.loc[seed_df["seed"] == 42, "op_roi"].iloc[0])
        say(f"          seed 42 (the default)  : {r42:+.2f}%, percentile "
            f"{(r < r42).mean():.0%} of the sweep")
    say(f"  sample  week-bootstrap P(ROI<=0): {q(seed_df['wk_p_le_zero'])}")
    say(f"          seeds whose 95% CI > 0  : {(seed_df['wk_lo'] > 0).mean():.0%}")
    say(f"  select  share of random beaten : {q(seed_df['beats_random'])}")
    say(f"          seeds beating > 95%    : {(seed_df['beats_random'] > 0.95).mean():.0%}")
    say(f"  folds   best fold's share of profit (profitable seeds only): "
        f"{q(seed_df['best_fold_share'])}")
    say("          above 1.0 means one fold earned more than the whole run.")

    # 4. every arm ------------------------------------------------------------
    say("\n4. EVERY ARM AT 10% COVERAGE BY EDGE  (pooled test threshold; for "
        "comparison, not a strategy)")
    for prefix in ARMS:
        s = seed_df.get(f"roi10_{prefix}")
        if s is None or s.isna().all():
            say(f"  {prefix:8s} collapsed onto the market in every seed")
            continue
        say(f"  {prefix:8s} {q(s)}   ROI > 0 in {(s > 0).mean():.0%}   "
            f"(rankable in {s.notna().mean():.0%} of seeds)")

    say(f"\n  {seed_df['seconds'].median():.0f}s per seed (median)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", default="export", choices=["export", "multiseason"])
    ap.add_argument("--source", default=None)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--learner", default="torch", choices=list(ts.LEARNERS))
    ap.add_argument("--ensemble-k", type=int, default=1)
    ap.add_argument("--learned-w", action="store_true",
                    help="also fit w(x); slower, and not needed for w or w_band")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None, help="output folder")
    args = ap.parse_args()

    defaults = {"export": "raw/all-euro-data-2025-2026.csv",
                "multiseason": "matches_all.csv"}
    source = args.source or defaults[args.mode]
    out = Path(args.out) if args.out else (
        ts.DATA / "sweeps" /
        f"{args.mode}_{args.learner}_k{args.ensemble_k}_s{args.seeds}")
    out.mkdir(parents=True, exist_ok=True)

    df = (ts.load_and_prepare(source) if args.mode == "export"
          else ts.load_multiseason(source))
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"{len(seeds)} seeds on {args.jobs} workers -> {out}")

    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
        delayed(one_seed)(df, s, args.learner, args.ensemble_k, args.blocks,
                          args.learned_w)
        for s in seeds)

    fold_df = pd.DataFrame([row for rows, _, _ in results for row in rows])
    seed_df = pd.DataFrame([summary for _, summary, _ in results])
    bets_df = pd.concat([b for _, _, b in results], ignore_index=True)
    fold_df.to_csv(out / "folds.csv", index=False)
    seed_df.to_csv(out / "seeds.csv", index=False)
    bets_df.to_csv(out / "op_bets.csv", index=False)

    text = report(fold_df, seed_df)
    text += f"\n  {time.time() - t0:.0f}s wall clock\n"
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"wrote folds.csv, seeds.csv, op_bets.csv, summary.txt to {out}")


if __name__ == "__main__":
    main()
