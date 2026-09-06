# Selective Betting Project — Group 11

## What this is

A quantitative football betting pipeline. The research claim is **not** that we beat
the market. It is that the value-betting selection rule suffers a **winner's curse**:
thresholding on estimated edge selects for estimation error as much as for real edge,
so realised return degrades as the threshold tightens. We correct it with James–Stein
shrinkage of the model's deviation toward the market price, applied *before* the
threshold.

Contribution = shrinkage at the selection step. Everything else is standard.

## Non-negotiable conventions

- **Order matters:** de-vig → model → calibrate → deviation → shrink → threshold.
  Never shrink an uncalibrated probability.
- **Every feature must be computable at kickoff.** Any feature using post-match or
  end-of-dataset information is a bug, not a design choice.
- **Every reported ROI needs a bootstrapped CI.** No point estimates alone, ever.
  Resample matches, not bets — bets on the same match are correlated.
- **Accuracy is not a selection criterion.** Rank models by log loss, RPS, Brier and
  classwise ECE. Report accuracy once, with a note explaining why it is excluded.
- Tune on validation only. The test set is unlocked once, at the end.

## Measured facts — do not re-derive

From EDA on the 2025/26 data (all 22 divisions, 7,466 matches):

- Bookmaker overround averages **7.7%**. Raw `1/odds` is not a probability.
- Favourite–longshot bias is present and monotone: outcomes priced 0.113 win 0.076
  (ratio 0.68). The margin is **not** spread evenly, so basic normalisation is wrong
  at long odds. Shin is the default; compare against power and basic.
- De-vigged market log loss ≈ **1.00**. This is the number to beat and the harness
  sanity check — a dummy model returning the market price must score this.
- Rolling form correlates **+0.0001** (p = 0.995) with the market residual. Form
  features are fully priced. Do not expect them to add predictive value.
- Per-bet P&L standard deviation ≈ **1.18** at flat stakes. Roughly **2,900 bets**
  are needed to resolve a 3.5% edge at two standard errors.
- Reported backtest of +3.48% on 927 bets bootstraps to a 95% CI of
  **[−2.67%, +9.55%]**, t = 1.12. Not significant. Treat it as provisional.

## Data state — read before writing any loader

**We currently hold one season only: 2025/26, 7,466 matches, 22 divisions.**
The `train_seasons = ["1516" ... "2122"]` lists in `newFrame.py` refer to data we do
not have. Any run against the current files will fail or silently train on almost
nothing. Downloading more seasons is the highest-value hour available — see below.

### The supplied export is malformed in four ways

`all-euro-data-2025-2026.csv` is a naive concatenation of 22 sheets. Reading it with
`pd.read_csv` defaults silently drops 573 rows and NaNs every odds column. Use
`load_export.py`, which handles all four:

1. **Semicolon-delimited**, not comma.
2. **Comma decimal separator** — values are `1,44` not `1.44`. This is the one that
   matters most: 90% of numeric cells are affected, and `pd.to_numeric` turns them all
   into NaN without raising. Symptom is B365H at 6% coverage instead of 99.8%.
3. **Embedded header rows** — each of the 22 sheets kept its own `Div,Date,...` row,
   and the sheets have 124 / 131 / 132 / 133 columns (EC lacks match statistics,
   13 divisions lack `Referee`, E3 has a trailing empty field).
4. **Dates are m/d/yy**, US order — `8/15/25` is 15 August. `dayfirst=True` or
   `format="mixed"` will misparse silently. Football-data's own CSVs use dd/mm/yyyy,
   so this is an artefact of the export, not the source.

A correct load yields exactly: 7,466 rows, 22 divisions, 2025-07-25 to 2026-05-14,
zero unparsed dates, PSH coverage 39.3%, B365H and MaxH 99.8%. Treat those as the
regression test for any loader change.

### Requiring Pinnacle discards 60% of the data

Only **2,931 of 7,466** matches have both Pinnacle and Max prices. `p_shin` is
currently derived from `PSH`, so the pipeline drops most of the season before it
starts. Source the benchmark from `AvgCH` (99.8% coverage) and use Pinnacle only
where present.

### What one season can support

A chronological 70/30 split gives 5,216 training and 2,236 test matches, which yields
roughly 250–950 bets depending on how many clear the gates — a standard error of
3.8% to 7.4%. That cannot distinguish a real edge from zero, which is the same
position the reported 927-bet backtest is in.

### Fix: download from source

Use `fetch_data.py`. Start at **2005/06** — that is when `Max`/`Avg` prices begin, and
`MaxH` is the execution price we need. All 22 divisions through 2025/26 is roughly
150,000 matches.

```python
SEASONS = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2005, 2026)]
```

Files from `football-data.co.uk/mmz4281/{season}/{div}.csv` are clean: comma
delimited, dd/mm/yyyy, dot decimals. None of the four problems above apply to them.
**Until this download happens, walk-forward is impossible and every interval stays at
±4% or worse.**

## Known traps

**Pinnacle is unusable from 23/07/2025.** Coverage 39.3%. Its price exceeds the
recorded market maximum in 26.2% of matches; no other bookmaker does so once. Drop
`PS*` / `P*` columns for 25/26, or source the benchmark from `AvgCH` instead.

**`Max*` and `Avg*` have structural breaks** at 2019/20 (Betbrain → Oddsportal) and
at 23/07/2025 (Pinnacle removed from the calculation). Do not pool across these
silently.

**Rest-days bug in `calculate_rolling_features`.** `df.groupby('HomeTeam')['Date'].diff()`
gives days since that team's last *home* match, not its last match. Fix by computing
rest on the long-format table, the same way goals are already handled.

**Referee is 44.6% complete**, entirely absent in Spain and Italy. Cannot be used
uniformly across a pooled model.

**National League (EC) has no match statistics** — results and odds only.

**Closing odds exist only from 2019/20.** `Max`/`Avg` only from 2005/06. This bounds
the modelling window.

## Decisions already made

- **Bet at `MaxH`, benchmark against Pinnacle.** Betting at `PSH` while benchmarking
  against Pinnacle-derived probabilities is circular — we would be trying to beat the
  sharpest book using its own price as the reference. `MaxH` margin is 3.47% against
  Pinnacle's; the switch is worth 3–4 ROI points mechanically.
- **Two-stage model.** Stage 1 trains with **no odds features**, so it forms an
  independent view. Stage 2 combines Stage 1's output with market data. Without this,
  `p_shin_*` sits in `FEATURE_COLS`, the model largely reproduces the market, and
  shrinking toward the market double-counts.
- **Walk-forward, not a fixed three-season test window.** More training seasons do not
  increase statistical power; more *test* seasons do. Walk-forward across 20 seasons
  takes us from ~927 bets to ~4,000+, and from t = 1.12 to roughly t = 2.4.
- **Refit `w` per fold.** Fitting once on 22/23 assumes the shrinkage weight is stable
  across two decades. It probably is not, and testing that is a result in itself.
- **Gradient boosting over deep learning.** Yeung, Sit & Fujii (2023) found GBT beats
  LSTM/GRU on football prediction — tabular structure, short training windows.

## Do not

- Do not add TabPFN, FT-Transformer, TabNet or an RL staking agent. Evidence says they
  lose to LightGBM here, and an end-to-end network destroys the modularity the
  shrinkage claim depends on.
- Do not tune the threshold on test data. The original prototype did this and turned
  −4.39% into a reported +9.51%.
- Do not report a pooled ROI across 20 seasons without an era breakdown. The market
  got sharper; a single number hides whether the edge is dead.
- Do not treat the +3.48% as an established edge in any writing.

## Open questions the code should answer

1. What is `w`? If it exceeds 0.5, something structural is wrong — investigate rather
   than proceed.
2. Does `w` decline across eras? Predicted yes, as the market matured. This would be
   the strongest finding available.
3. **Is the correction non-vacuous?** Shrinking by `w` then thresholding at `τ` could
   be arithmetically identical to thresholding at `τ/w` without shrinking. If so we
   have relabelled an axis, not corrected anything. The corrected and uncorrected
   curves must differ in **shape**, not merely position. This is a required check.

## The headline figure

ROI against threshold `τ`, two lines: uncorrected (expected to decline) and corrected
(expected to flatten). With bootstrapped bands. Everything else in the repo exists to
make that figure trustworthy.
