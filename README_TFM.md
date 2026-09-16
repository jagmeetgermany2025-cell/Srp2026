# Tabular models for the SRP baseline

This implements proposal 3 in `n.pdf` against branch `srp0609`, commit
`9c0f7b1bba1e710fd89164542a2f81472da776fe`. It compares the original
LightGBM hyperparameters with **TabPFN**, **TabNet**, and **FT-Transformer**.
The original scripts remain available. The new experiment is a controlled
single-stage backbone comparison, not a reproduction of their reported scores.
It retains the sequence: predict -> calibrate -> shrink toward the market ->
select bets.

TabPFN is a pretrained foundation model doing in-context supervised prediction.
TabNet and FT-Transformer are deep tabular architectures trained from scratch;
they are not described here as pretrained foundation models. The default TabPFN
checkpoint is explicitly **v2**, rather than whatever a future package chooses
by default. `--tabpfn-version v2.5` or `--checkpoint /path/to/model.ckpt` selects
another checkpoint. No CARDS implementation is claimed: the supplied proposal
does not identify an implementation or checkpoint. No RL or end-to-end betting
policy is included in this model comparison.

## Installation

Use Python 3.12 and a fresh environment. These top-level dependency versions
were tested together; each run also records installed versions in its manifest.
`requirements-tfm-lock.txt` additionally records the full verification
environment on macOS ARM64; it includes platform-specific MLX packages and is
an audit snapshot, not a portable Linux/CUDA lock file.

```bash
cd /path/to/Srp2026
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tfm.txt
```

On macOS, LightGBM requires the OpenMP runtime (`brew install libomp`). If
PyTorch is already installed, this session also successfully used its bundled
runtime without a system installation:

```bash
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

On Linux, install the platform's OpenMP runtime if LightGBM reports a missing
shared library. Requested models fail explicitly if unavailable; the experiment
never silently substitutes a different estimator.

TabPFN downloads weights on first use and then predicts locally. To keep caches
inside the project:

```bash
export TABPFN_MODEL_CACHE_DIR="$PWD/.cache/tabpfn"
export HF_HOME="$PWD/.cache/huggingface"
```

Newer checkpoints may require access/license acceptance through their provider.
The code does not bypass that requirement. A downloaded compatible checkpoint
can be passed with `--checkpoint` for offline use.

## Run

Quick pipeline check without a pretrained-weight download:

```bash
python run_tfm.py --models lightgbm tabnet ft_transformer \
  --epochs 3 --patience 2 --test-seasons 2425 --bootstrap 100 \
  --output results/smoke
```

Compare all four models with a common CPU-friendly training budget:

```bash
python run_tfm.py --models lightgbm tabpfn tabnet ft_transformer \
  --seeds 42 43 44 --max-train-rows 1000 --epochs 100 \
  --output results/tfm_cpu
```

The default 1,000-row cap is a resource setting, not a finding about the best
training size. It retains the latest complete training dates up to the cap,
identically for every model. All eligible calibration, shrinkage and test rows
are retained. Full-history training is available with `--max-train-rows 0`;
respect the selected TabPFN checkpoint's documented context limits and memory
requirements. For a larger controlled GPU experiment:

```bash
python run_tfm.py --device cuda --max-train-rows 10000 \
  --seeds 42 43 44 --epochs 100 --output results/tfm_gpu
```

`--device mps` is available when supported by the installed backend. CPU was
used for verification here. `--batch-size` controls training/query batches,
`--threads` CPU Torch threads, and `--tabpfn-estimators` its ensemble size.
Outputs must go to a fresh directory to prevent mixing different experiments.

For repeated TabPFN prediction calls, `--tabpfn-fit-mode fit_with_cache` caches
attention state and can reduce CPU runtime at the cost of additional memory.
The completed two-season run is in `results/comparison_20260916`. Its exact
command is recorded in `run_command.txt`. Generate its English report (`RESULTS.md`), pooled
scores, plot and exploratory paired bootstrap comparisons with:

```bash
python summarize_tfm.py results/comparison_20260916
```

This summarizer requires a single seed, pools each test match once, and reports
the fixed 2% EV threshold rather than choosing a threshold from test profits.

## Experimental protocol

Every eligible test season uses all earlier seasons except its immediate
predecessor for training, and that predecessor for validation. Actual dates,
not season strings, establish order. Any overlapping season ranges cause an
error. With the supplied CSV, the default eligible tests are **2425 and 2526**;
2627 contains only 228 rows and fails the default 1,000-match minimum. The
minimum is an eligibility rule, not a guarantee that a season is complete.

Within each fold, ordered calendar-day blocks are:

1. First 80% of training dates: model fitting (then apply the common row cap).
2. Last 20% of training dates: neural early stopping only.
3. First 50% of validation dates: scalar temperature calibration.
4. Last 50% of validation dates: static geometric shrinkage weight.
5. Entire test season: final evaluation only.

The base estimator is not refitted after early stopping. LightGBM and TabPFN
use exactly the same fitting rows as the neural models; they do not use the
early-stopping labels. Temperature and shrinkage parameters are frozen before
test prediction. Thresholds are specified in advance, never selected by test
ROI. No randomized cross-validation is used. The prediction CSV stores raw,
calibrated, corrected and market-only probabilities, so calibration and
shrinkage can be evaluated separately.

### Features and information timing

The default uses the baseline's 18 numeric features: 15 rolling goals/rest
features plus 3 de-vigged market probabilities. `--feature-set market_blind`
removes the market inputs from every estimator, retaining the market only for
subsequent shrinkage and comparison. No team-ID embeddings, xG, injuries, or
other extra features are introduced, so differences can be attributed more
cleanly to the backbone. FT-Transformer uses its continuous feature tokenizer;
there are no categorical input columns in this experiment.

Rolling features are shifted before exponentially weighted averaging. Rest is
time since the last home **or** away appearance. Club histories persist across
promotion/relegation within a country. Duplicate fixtures and a club appearing
twice on the same date cause errors because the CSV lacks a reliable ordering
for those cases. Train-only median imputation and scaling handle missing early
history; all-empty training features are kept and imputed to zero. Current
match goals, results, shots, and closing odds are never features in the default
pre-match experiment.

Pre-match mode uses `AvgH/AvgD/AvgA` as the benchmark and `MaxH/MaxD/MaxA` for
execution. Closing mode (`--market closing`) uses `AvgCH/CD/CA` and `MaxCH/CD/CA`
together. The legacy selector's column names are normalized to the selected
snapshot. Missing benchmark rows are excluded from evaluation after their
results have contributed to later histories. Missing execution odds only
exclude the affected betting candidate; they do not remove prediction rows.
No provider/timestamp fallback is performed.

The new loader also corrects a numerical error in both legacy scripts' Shin
solver: for three outcomes the square-root sum must equal `2 + z`, whereas
the baseline bisects against `2 - z`. The new vectorized solver enforces that
the unnormalized Shin probabilities sum to one and is tested against the
[published shin package example](https://github.com/mberk/shin#usage).
Legacy functions remain unchanged for reproducibility. This shared correction
applies to every arm, including LightGBM, so historical scores from the legacy
scripts must not be directly compared with the new experiment's scores.

The CSV's snapshot labels do not prove simultaneous executable prices. Best
bookmaker odds, availability, limits, fees, and time of collection are not
verified here. Results are retrospective model/selection experiments. Test
features update with earlier observed results within that test season; the
model weights and calibration remain frozen. This is sequential pre-match
prediction, not forecasting every fixture at the start of the season.

### Metrics and artifacts

- `metrics.csv`: fold/seed/model scores for each probability arm. Log loss,
  multiclass Brier (sum across classes), ranked probability score (H-D-A order,
  divided by 2), 15-bin top-label ECE, and accuracy.
- `risk_coverage.csv`: fixed EV thresholds, H/A selections, flat one-unit
  stakes, candidate coverage, ROI and percentile 95% intervals from a
  match-cluster bootstrap. Multiple selections on one match stay together.
  Empty betting sets have blank ROI values, not zero ROI.
- `predictions_<model>_<seed>_<season>.csv`: test fixture identifiers, target,
  benchmark/execution odds, and H/D/A probabilities (`p_raw`, `p_model`,
  `p_corr`, `p_ref`). Compatible with the legacy `select_bets` function.
- `folds.csv`: actual row counts, temperature, shrinkage weight, best epoch,
  elapsed time, and resolved TabPFN checkpoint/hash when available.
- `split_assignments.csv`: match-level membership for every period in every
  fold. Match reuse across later folds is intentional walk-forward training.
- `manifest.json`: source SHA-256, code hashes, baseline commit, package
  versions, features and run settings, exclusions, and complete/failed status.

Repeated market scores under each model/seed are the same reference arm,
not independent observations. Seed runs also share test matches. Report folds
and seeds separately; do not inflate the sample size by treating predictions
from different seeds as additional matches. The ROI bootstrap assumes match
clusters are exchangeable and does not account for temporal/team dependence;
use date-block or season-level inference before making stronger claims. The
runner does not select a winning model or assert statistically significant
improvement. Model tuning would need another nested chronological protocol.

## Tests

```bash
python -m pytest tests -q
# Explicitly enable the test that needs public TabPFN weights:
RUN_TABPFN_INTEGRATION=1 python -m pytest tests -k real_tabpfn -q
```

Tests cover temporal leakage, timestamp consistency, split disjointness,
training-only preprocessing, class order, calibration/shrinkage edge cases,
real neural-model fits, adapter behavior, and end-to-end output files. A smoke
run is execution verification, not a trained research result.

## Primary implementation references

- [TabPFN and supported checkpoint interfaces](https://github.com/PriorLabs/TabPFN)
- [Official PyTorch TabNet implementation](https://github.com/dreamquark-ai/tabnet)
- [Official FT-Transformer package and architecture](https://github.com/yandex-research/rtdl-revisiting-models/tree/main/package)
- [Original project baseline](https://github.com/jagmeetgermany2025-cell/Srp2026/tree/srp0609)
