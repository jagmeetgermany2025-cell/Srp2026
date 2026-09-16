# Verification record

Environment: Python 3.12, macOS ARM64, CPU; exact installed distributions are
recorded in `requirements-tfm-lock.txt`.

The automated suite passed **13 tests**, including the explicit TabPFN v2
checkpoint integration test. TabNet emits an upstream SciPy deprecation warning
and its expected message about restoring the best epoch. No tests were skipped
in this run. Compilation and `git diff --check` also passed.

The research runner additionally completed all four model arms against the supplied multi-season
CSV, with its **2024/25 season (7,678 matches)** held out. The command below is a
smoke test with three epochs and one TabPFN ensemble member, not a tuned model
comparison:

```bash
python run_tfm.py --models lightgbm tabnet ft_transformer tabpfn \
  --epochs 3 --patience 2 --test-seasons 2425 \
  --tabpfn-estimators 1 --max-train-rows 300 --bootstrap 100 \
  --output results/verified_smoke
```

The whole-date resource cap retains 268 fitting matches for every model. The
other partitions contain 1,542 early-stopping, 3,953 calibration, 3,838
shrinkage-fitting and 7,678 test matches. Generated metrics, probabilities and
run status live in `results/verified_smoke/`, which is deliberately Git-ignored.
The manifest's `status` is the authority on completion; intermediate files alone
are not evidence of a completed comparison.
This verification run finished with `status: complete` and 16 metric rows
(four probability variants for each of four models).

An earlier `results/smoke_real_data/` run used the legacy Shin solver and is
superseded by `verified_smoke`; do not use those earlier metrics for research.
The tests and data checks demonstrate execution and the tested information
boundaries, not a profitable strategy or statistically superior model.
