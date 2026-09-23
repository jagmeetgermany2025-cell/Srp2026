# SRP_2109 — selective betting with shrinkage

Does shrinking a model's estimate toward the market price *before* choosing
which bets to place remove the winner's curse? Selecting the matches where a
model most disagrees with the market selects, disproportionately, the matches
where the model is most wrong — so the act of choosing biases the estimate.

## Layout

    pipeline/   the method
      two_stage.py         the pipeline. Stage 1 is gradient-boosted trees
      two_stage_torch.py   the same pipeline, stage 1 as a neural net
      elo.py               Elo ratings, every constant fitted rather than chosen
    build/      turn raw data into features (run in this order)
      build_matches.py     merge every season into data/matches_all.csv
      build_elo.py         walk-forward Elo, refit per season on its past
      build_history.py     last season's table, promoted/relegated flags
      build_xg.py          Understat xG, rolling and shifted
      build_xgproxy.py     an xG stand-in from shot counts, for the other leagues
      build_squad_value.py Transfermarkt squad values
      build_lineups.py     who is missing from today's squad
      build_dataset.py     join everything, one row per match
    fetch/      go and get the raw data
      fetch_fd.py          football-data.co.uk seasons, via the archive
      fetch_understat.py   Understat seasons
      fetch_venues.py      a coordinate per club
      fetch_weather.py     kickoff-day weather
      fetch_injuries.py    API-Football injury reports
    analysis/   one-off checks that are worth keeping
      feature_check.py     what each feature block is worth
      check_w_curve.py     sweep the shrinkage weight by hand
    data/       everything read or written, including raw/ and preds/
    legacy/     2-stage.py, the original script this grew out of

Paths are anchored to the project root, so a script runs the same from
anywhere. A bare filename means a file in `data/`.

## Running it

    python pipeline/two_stage.py --mode multiseason --source matches_all.csv
    python pipeline/two_stage_torch.py --mode multiseason --source matches_all.csv

Add `--export preds.csv` to write per-match predictions, so a season can be
sliced out afterwards without running the whole thing again.

## Where it stands

Seven seasons, 22 divisions, 53,255 matches; five test seasons, each predicted
by a model trained only on earlier ones.

* Stage 1 closes about three quarters of the gap between knowing nothing
  (1.0770) and the closing price (0.9950), at 1.0145.
* It never beats the price, so the fitted shrinkage weight is **zero in every
  fold** and the corrected arm is the market exactly.
* No arm is profitable. Selecting the top 10% by edge returns −6% over five
  seasons against −2% for betting every match at the best available price.
* The winner's curse is real and measured: expected returns of 127% against a
  realised 2% at no shrinkage, collapsing to 10.8% against 6.2% at full
  shrinkage.
* Ranking candidates by expected value manufactures a winner's-curse pattern
  out of the favourite–longshot bias. Under EV ranking the uncorrected model
  decays from −5% to −9.9% as it gets pickier; ranked by edge it is flat at
  −3.8%. Same model, same matches.
