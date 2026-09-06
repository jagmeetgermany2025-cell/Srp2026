# TASKS

Sequenced. Each task is one Claude Code session. Do not skip ahead — task 2 gates
everything after it.

---

## 0. Load the data correctly — do this first

**Prompt:**
> Add `load_export.py` to the repo and wire it into `newFrame.py` as the entry point,
> replacing the current `pd.read_csv` calls.
>
> The supplied `all-euro-data-2025-2026.csv` is malformed four ways: semicolon
> delimited, comma decimal separator (`1,44` not `1.44`), 22 embedded header rows with
> differing column counts, and m/d/yy dates. Reading it with pandas defaults silently
> drops 573 rows and NaNs every odds column.
>
> Assert on load: exactly 7,466 rows, 22 divisions, dates 2025-07-25 to 2026-05-14,
> zero unparsed dates, PSH coverage 39.3%, B365H and MaxH coverage 99.8%. Fail loudly
> if any of these do not hold — they are the regression test.
>
> Then change `p_shin` to derive from `AvgCH/AvgCD/AvgCA` rather than `PSH/PSD/PSA`.
> Requiring Pinnacle drops us from 7,452 usable matches to 2,931.

---

## 0b. Download the real dataset

**Prompt:**
> Run `fetch_data.py` for seasons 2005/06 through 2025/26 across all 22 divisions,
> caching to parquet. 2005/06 is the start because that is when Max/Avg prices begin.
>
> Then print a coverage table per season: rows, and the non-null fraction for B365H,
> B365CH, PSH, MaxH, AvgH, HST. Use it to fix the modelling window — closing odds only
> exist from 2019/20.
>
> The current `train_seasons`/`val_seasons`/`test_seasons` lists in `newFrame.py`
> reference seasons we do not have. Nothing downstream works until this runs.

**This gates tasks 3 and 7.** With one season, walk-forward is impossible and every
confidence interval stays at ±4% or worse.

---

## 1. Fix the three known bugs

**Prompt:**
> Fix three bugs in `newFrame.py`, one commit each.
>
> (a) Rest days. `calculate_rolling_features` computes `home_rest` by grouping on
> `HomeTeam`, which gives days since that team's last *home* match. It should be days
> since its last match of any kind. The long-format `matches` table built later in the
> same function is the right structure — compute rest there and join back.
>
> (b) Execution price. The backtest bets at `PSH`. Change it to bet at `MaxH`/`MaxD`/
> `MaxA` while keeping `p_shin` from Pinnacle as the benchmark. Add the `Max*` columns
> to `output_cols` in `run_model_training` so they reach the backtest. Report both
> prices side by side so the mechanical gain from line shopping is visible separately
> from any model edge.
>
> (c) Contaminated season. Add a flag to exclude 25/26 from the test window, and print
> the bet count per season before and after so the cost is measurable.
>
> After each fix, re-run the backtest and report how the numbers moved.

---

## 2. GATE — bootstrap the existing result

**Prompt:**
> Add a bootstrap function that takes a bets dataframe and returns ROI with a 95% CI
> and P(ROI ≤ 0). Resample **matches**, not individual bets — two bets on the same
> match are correlated. Apply it to every ROI printed anywhere in the codebase,
> including `run_threshold_sweep`.
>
> Then re-run everything and give me a table: threshold, bets, ROI, CI, for raw and
> shrunk.

**Decision point.** If the corrected curve does not differ from the uncorrected one
once intervals are attached, say so plainly and stop. Reframe before building further.

---

## 3. Walk-forward

**Prompt:**
> Restructure `run_model_training` into walk-forward. Currently `train_seasons`,
> `val_seasons` and `test_seasons` are hardcoded. Replace with a loop: for each test
> season k, train on all seasons up to k−2, validate on k−1, test on k. Refit the
> shrinkage weight `w` on each fold's validation set rather than once globally.
> Accumulate out-of-sample predictions across all folds into a single dataframe.
>
> Store the per-fold `w` so we can plot it over time.
>
> Extend the data back to 2005/06 — that is when `Max`/`Avg` prices begin.
>
> Report: total out-of-sample matches, total bets, and `w` per fold.

This is the task that matters most. It is the only change that can make the result
statistically significant.

---

## 4. Two-stage model

**Prompt:**
> Split the model in two.
>
> Stage 1: train on `FEATURE_COLS` with all `p_shin_*` columns removed, so it never
> sees the market. Output three probabilities.
>
> Stage 2: train on Stage 1's three outputs plus `p_shin_H/D/A` plus the variance
> proxies. Output the final three probabilities.
>
> Use out-of-fold predictions from Stage 1 as Stage 2's inputs — do not let Stage 2
> train on Stage 1's in-sample predictions.
>
> Then compare, on the same test folds: Stage 1 alone, Stage 2, and the current
> single-stage model. Report log loss, RPS, ECE, and the estimated `w` for each.
>
> Expectation: `w` should be *lower* for the two-stage version, because Stage 1's
> disagreement with the market is genuine rather than an echo of it.

---

## 5. Non-vacuousness check

**Prompt:**
> Verify the shrinkage correction is not a reparametrisation of the threshold.
>
> For each threshold τ under shrinkage, find the τ′ that selects the same number of
> bets without shrinkage. Compare the two bet sets: how much do they overlap, and do
> the ROIs differ?
>
> If shrinking by `w` and thresholding at τ selects essentially the same matches as
> thresholding at τ′ with no shrinkage, the correction adds nothing and we need to
> know that. Report the overlap fraction per threshold.

Do not skip this. It is the check an examiner will go for.

---

## 6. Model pool

**Prompt:**
> Add XGBoost, CatBoost, multinomial logistic regression, and a Dixon–Coles Poisson
> baseline alongside LightGBM. Wrap them so the calibration, shrinkage and backtest
> code is unchanged.
>
> Produce two rankings on validation data: one by accuracy, one by classwise ECE.
> Show whether they disagree — that disagreement is the Walsh & Joshi replication.
>
> Then run the shrinkage correction on each. If the threshold–ROI pattern appears
> across all architectures, it is a property of the selection rule rather than of
> LightGBM.

---

## 7. Era analysis

**Prompt:**
> Using the per-fold `w` from task 3, plot the shrinkage weight over time. Also break
> ROI down by era rather than pooling across 20 seasons.
>
> Add an era marker at 2019/20 (Betbrain → Oddsportal changed how `Max`/`Avg` are
> computed) and check whether results shift discontinuously there.
>
> Hypothesis: `w` declines over time as the market sharpens. If it does, that is a
> direct measurement of the market becoming harder to beat.

---

## 8. Leakage test suite

**Prompt:**
> Write tests asserting:
> - every feature is computable at kickoff (no post-match information)
> - shifting all features forward one match degrades log loss
> - removing any single pipeline component changes the output — the original prototype
>   shipped a rating engine that was never called and nobody noticed
> - `p_shin` sums to 1 and each component is in (0,1)
> - no test-set row appears in any training fold

---

## 9. Figures and write-up

**Prompt:**
> Produce the headline figure: ROI against threshold, corrected and uncorrected, with
> bootstrapped bands. Plus reliability diagrams per model, `w` over time, and ROI by
> era.

---

## Cut order if time runs short

7 → 6 → 4. Never cut 0, 0b, 2, 3, 5, or 8.

Tasks 2, 3 and 5 are the dissertation. Everything else is supporting evidence.
