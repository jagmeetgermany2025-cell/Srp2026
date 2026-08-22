"""
Selective betting under the winner's curse — pipeline scaffold.

Each stub carries, where they apply:
    Source:     the paper the method comes from
    Equation:   the maths, in the notation used by the write-up
    Pseudocode: the steps to implement

Verify volume and page numbers against the originals before submission — the
paper identifications are reliable, the bibliographic details are from memory.

Notation:
    o_k     decimal odds on outcome k in {H, D, A}
    y_k     realised outcome indicator, sums to 1 over k
    Gamma   booksum, sum_k 1/o_k, exceeds 1 by the bookmaker's margin
    q_k     de-vigged market probability
    p_k     model probability estimate
    delta   model deviation from market, in log-odds
    w       shrinkage weight
    tau     selection threshold

Build order — finish each block before starting the next:

    A  data            1-2
    D  harness         3-7      <- build before any modelling
    E  leakage tests   8-11
    J  simulation      12-16    <- the contribution's insurance; no football data
    B  de-vigging      17-20
    C  features        21-25
    F  models          26-27
    G  calibration     28-30
    H  shrinkage       31-35    <- the contribution
    I  betting sim     36-39    <- produces the figure
    K  plots           40-42

Sections B-C and J are independent and can run in parallel.
"""
from __future__ import annotations

import pandas as pd

import fetch_data

# =============================================================================
# A. DATA
# =============================================================================


def load_matches(seasons=None, divisions=None):
    """Load match results and odds into one frame, one row per match.

    Why: everything downstream depends on this. Takes a `seasons` list rather
    than hardcoding a season so that adding seasons later is a config change,
    not a rewrite.

    Note: drop Pinnacle (PS*/P*) only from 2025-07-23, when the feed broke.
    Before that date it is the sharpest book in the file and the standard
    reference price in this literature — a blanket drop discards it needlessly.

    Pseudocode:
        for each (season, division):
            read CSV, coercing decimal separator and encoding
            tag rows with season and division
        concatenate
        parse Date with an explicit format, never inferred
        drop rows with no result or no opening price
        mask PS*/P* columns where Date >= 2025-07-23
        return frame sorted by Date
    """
    df = fetch_data.load_matches(seasons=seasons, divisions=divisions)
    df = df.dropna(subset=["FTR"])
    odds_cols = [c for c in ("B365H", "B365D", "B365A") if c in df.columns]
    if odds_cols:
        df = df.dropna(subset=odds_cols)
    return df.sort_values("Date").reset_index(drop=True)


def fetch_seasons(seasons, divisions, cache_dir):
    """Download football-data.co.uk CSVs and cache them locally.

    Why: deferred until the pipeline works on one season. Once it does, adding
    seasons is this function plus a config line.

    Note: URL pattern is mmz4281/{season}/{div}.csv. Closing odds exist only
    from 2019/20 — earlier seasons carry opening prices alone, which constrains
    any feature or benchmark built on line movement.

    Pseudocode:
        for each (season, division):
            if cached: skip
            GET mmz4281/{season}/{div}.csv
            skip on non-200 or implausibly small payload
            write to cache
            sleep briefly between requests
    """
    return fetch_data.fetch_seasons(seasons=seasons, divisions=divisions, cache_dir=cache_dir)


def coverage_table(df):
    """Report per-season, per-column completeness.

    Why: decides the modelling window. A feature present in 40% of seasons is
    not a feature, it is a source of silent sample-selection.

    Pseudocode:
        group by season
        for each odds column: fraction non-null
        return seasons x columns matrix
        flag columns crossing a usability threshold mid-sample
    """
    groups = {
        "Result": ["FTHG", "FTAG", "FTR"],
        "Officials": ["Referee"],
        "Shots": ["HS", "AS", "HST", "AST"],
        "Discipline": ["HF", "AF", "HY", "AY", "HR", "AR"],
        "Corners": ["HC", "AC"],
        "1X2 opening": ["B365H", "B365D", "B365A"],
        "1X2 closing": ["B365CH", "B365CD", "B365CA"],
        "Pinnacle opening": ["PSH", "PSD", "PSA"],
        # kept apart: MaxH runs from 0506 but AvgH only from 1920, so grouping
        # them reports the whole family absent for fifteen seasons
        "Market best": ["MaxH", "MaxD", "MaxA"],
        "Market avg": ["AvgH", "AvgD", "AvgA"],
        "Asian handicap": ["AHh", "B365AHH", "B365AHA"],
    }
    seasons = sorted(df["Season"].unique())

    rows = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        for season in seasons:
            g = df.loc[df["Season"] == season, cols]
            cov = g.notna().all(axis=1).mean() if len(g) else float("nan")
            rows.append({"Season": season, "Group": name, "Coverage": cov})

    table = (pd.DataFrame(rows)
             .pivot(index="Season", columns="Group", values="Coverage")
             .reindex(seasons))

    # A group present in most of one season and nearly absent in the next is a
    # silent sample-selection trap, not a usable feature across the window.
    # Report the usable SPAN rather than a start date: a column can die
    # (Pinnacle, 1213-2425) as readily as it can arrive.
    flags = {}
    for name in table.columns:
        s = table[name].dropna()
        usable = s[s >= 0.8]
        if len(s) < 2 or usable.empty or not (s < 0.2).any():
            continue
        lo, hi = usable.index.min(), usable.index.max()
        notes = []
        if lo > s.index.min():
            notes.append(f"absent before {lo}")
        if hi < s.index.max():
            notes.append(f"absent after {hi}")
        flags[name] = f"usable {lo}-{hi}" + (f" ({'; '.join(notes)})" if notes else "")

    return table, flags


# =============================================================================
# B. DE-VIGGING  —  odds to fair probabilities
# =============================================================================


def devig_basic(odds):
    """Proportional normalisation: divide each 1/odds by their sum.

    Why: the baseline everyone uses, and the one this project's own EDA shows
    is wrong. It assumes the margin sits evenly across outcomes.

    Source: standard practice; the reference method in Strumbelj (2014),
        International Journal of Forecasting 30(4).

    Equation:
        Gamma = sum_j (1 / o_j)
        q_k   = (1 / o_k) / Gamma

    Pseudocode:
        pi    <- 1 / odds
        Gamma <- sum(pi)
        return pi / Gamma
    """
    raise NotImplementedError


def devig_power(odds):
    """Power method: raise inverse odds to an exponent that makes them sum to 1.

    Why: allows the margin to fall unevenly across outcomes, which the measured
    favourite-longshot bias says it does. Because the exponent acts in log
    space, it compresses long odds more than short ones.

    Source: Strumbelj (2014), International Journal of Forecasting 30(4).

    Equation:
        find k > 0 such that sum_j (1 / o_j)^k = 1
        q_k = (1 / o_k)^k

    Pseudocode:
        pi <- 1 / odds
        define f(k) = sum(pi ** k) - 1
        solve f(k) = 0 by bisection or Brent on k in (0, 5]
        return pi ** k
    """
    raise NotImplementedError


def devig_shin(odds):
    """Shin method: back out the implied share of insider trading.

    Why: derives the uneven margin from a model of *why* it is uneven — the
    bookmaker widening prices against better-informed bettors — rather than
    fitting a free exponent. The standard alternative to the power method, and
    the one with an economic story attached.

    Source: Shin (1993), Economic Journal 103(420). Compared against the power
        method in Strumbelj (2014).

    Equation:
        pi_k   = 1 / o_k
        Gamma  = sum_j pi_j
        q_k(z) = [ sqrt( z^2 + 4(1 - z) * pi_k^2 / Gamma ) - z ] / ( 2(1 - z) )
        solve for z in [0, 1) such that sum_k q_k(z) = 1
        z is the estimated proportion of insider money

    Pseudocode:
        pi    <- 1 / odds
        Gamma <- sum(pi)
        define g(z) = sum_k q_k(z) - 1
        solve g(z) = 0 by bisection on z in [0, 1)
        return q(z), and keep z — it is interpretable on its own
    """
    raise NotImplementedError


def compare_devig_methods(df, methods):
    """Score each de-vigging method against observed outcome frequencies.

    Why: this choice must be settled *before* the threshold sweep. The EV filter
    concentrates on long odds as the threshold rises (measured: mean odds 4.04
    -> 7.20, longshot share 4% -> 27%), and long odds are exactly where basic
    normalisation is worst. Left unresolved, de-vig error and the winner's curse
    predict the same declining ROI curve and cannot be told apart.

    Source: replicates the comparison design of Strumbelj (2014).

    Acceptance test:
        longshots (q < 0.15) currently price 0.113 and win 0.076, ratio 0.68.
        A better method narrows that gap without harming the favourite end.

    Pseudocode:
        for each method:
            q <- method(odds)
            score log loss, RPS, Brier over all matches
            bin by q, compare mean predicted against observed frequency
            record longshot and favourite ratios separately
        return table; pick on calibration at the extremes, not on average
    """
    raise NotImplementedError


# =============================================================================
# C. FEATURES
# =============================================================================


def build_elo(df, k=20, hfa=60, start=1500.0):
    """Chronological Elo with a margin-of-victory multiplier.

    Why: ratings weight results by opponent quality, which a raw goal average
    cannot. Rating features beat recency features under identical models in the
    published comparison.

    Source: Elo (1978), The Rating of Chessplayers. Football adaptation and the
        margin-of-victory multiplier: Hvattum & Arntzen (2010), International
        Journal of Forecasting 26(3). Rating-beats-recency: Berrar, Lopes &
        Dubitzky (2019), Machine Learning 108.

    Equation:
        E_H = 1 / ( 1 + 10^( -(R_H + h - R_A) / 400 ) )
        S   = 1 if home win, 0.5 if draw, 0 if away win
        m   = ln( max(|goal difference|, 1) + 1 )
        R_H <- R_H + K * m * (S - E_H)
        R_A <- R_A - K * m * (S - E_H)          # zero sum

    Note: record ratings *before* updating, or a match informs its own feature.
    Keying on (league, team) resets promoted and relegated clubs to `start` —
    defensible within a division, but it discards history and leaves roughly ten
    matches of meaningless rating after every move.

    Pseudocode:
        ratings <- dict defaulting to start
        for each match in date order:
            record R_home, R_away as this match's features   # BEFORE update
            compute E_H, S, m
            update both ratings
        return elo_diff = (R_home + h) - R_away
    """
    raise NotImplementedError


def attach_clubelo(df, clubelo):
    """Join externally computed ClubElo ratings onto each match.

    Why: two purposes. First, an independent check — if our Elo diverges wildly
    from ClubElo we have a bug. Second and more useful, the *gap* between the
    two ratings is a per-match proxy for how uncertain a team's rating is, which
    feeds the heteroscedastic shrinkage weight in section H.

    Source: data source, not a method — api.clubelo.com. Pre-merged with
        football-data and team names reconciled in the GitHub project
        xgabora/Club-Football-Match-Data-2000-2025.

    Note: ClubElo is continuous across divisions, so it also sidesteps the
    promotion reset above. Snapshots are twice-monthly.

    Pseudocode:
        load ClubElo snapshots
        for each match: interpolate each team's rating to the match date
        join on (date, team) via the reconciled name mapping
        assert join rate above a threshold; sample-check rows by hand
        emit clubelo_diff and |our_elo_diff - clubelo_diff| as the proxy
    """
    raise NotImplementedError


def build_form(df, window=9):
    """Rolling form over all of a team's matches, home and away.

    Why: a season average misses the good and bad spells that matter for the
    next match. Nine matches across attack, defence, home advantage and
    opposition strength is the best-performing published configuration.

    Source: Berrar, Lopes & Dubitzky (2019), Machine Learning 108 — the
        nine-match window and four-group split are theirs.

    Equation:
        for team t at match i, over the previous n = window matches:
            f_t(i) = (1 / n) * sum_{j in previous n matches of t} x_t(j)
        feature = f_home(i) - f_away(i)

    Note: two traps. `shift(1)` must precede `rolling()`, or a match contributes
    to its own feature. And recover home/away rows by filtering on a `side`
    column, never by positional slicing (`iloc[0::2]`) — pandas' default sort is
    not stable, and positional recovery silently scrambled half the matches in
    the prototype.

    Pseudocode:
        explode to long form: one row per team per match, side in {H, A}
        sort by match index with a STABLE sort
        group by (league, team)
        for each stat: shift(1) then rolling(window).mean()
        split back with long[long.side == "H"] and long[long.side == "A"]
        join on match index, emit home-minus-away differentials
    """
    raise NotImplementedError


def build_rest(df):
    """Days since each team's previous match, and matches in the last fortnight.

    Why: computed from the date column alone, so it costs nothing. Rest-days
    measured near zero against the market residual (-0.003, p = 0.81);
    congestion is untested and is the part worth checking.

    Equation:
        rest_t(i) = date(i) - date(previous match of t)
        cong_t(i) = count of t's matches in [ date(i) - 14 days, date(i) )

    Pseudocode:
        explode to long form as in build_form
        sort by (team, date)
        rest <- groupby(team).date.diff().days
        cong <- rolling 14-day count, shifted to exclude the current match
        split back by side, emit differentials
    """
    raise NotImplementedError


def build_dispersion(df, books):
    """Spread of implied probabilities across bookmakers, plus count and best price.

    Why: when books agree the price is probably right; when they disagree at
    least one is wrong. Measured +0.029 against the market residual — one of the
    two strongest non-leaking features, and it comes from columns already held.

    Equation:
        disagree_i = stdev_b ( 1 / o_{i,b} )   over bookmakers b quoting match i
        count_i    = number of b quoting match i
        best_i     = max_b o_{i,b}

    Note: exclude Pinnacle for the affected season. Best price matters
    independently: it halves the margin (3.5% vs 7.7%) with no loss of forecast
    quality, so returns should be reported at both prices.

    Pseudocode:
        cols    <- available bookmaker columns, minus Pinnacle where masked
        implied <- 1 / df[cols]
        emit implied.std(axis=1), implied.notna().sum(axis=1), df[cols].max(axis=1)
    """
    raise NotImplementedError


def assemble_features(df):
    """Run every feature builder and return the modelling frame.

    Why: one entry point means one place to audit. Odds-derived features
    (handicap line, bookmaker disagreement) carried more signal than football
    features in the EDA, so build and verify those first.

    Note: line movement (closing minus opening) is the strongest signal measured
    (+0.070, p = 1.3e-9) but is known only at kickoff. As a feature for bets
    placed at opening prices it is leakage. Keep it as an evaluation benchmark
    (closing line value), never as a model input.

    Pseudocode:
        df <- build_elo(df)
        df <- attach_clubelo(df, clubelo)
        df <- build_form(df)
        df <- build_rest(df)
        df <- build_dispersion(df)
        df <- de-vigged market probabilities and their logits
        run section E assertions before returning
    """
    raise NotImplementedError


# =============================================================================
# D. EVALUATION HARNESS  —  build before any modelling
# =============================================================================


def log_loss_score(probs, outcomes):
    """Multiclass logarithmic loss.

    Why: the primary scoring rule. Strictly proper, so it is minimised only by
    the true probabilities — which is what makes it safe to optimise against.

    Source: Good (1952), Journal of the Royal Statistical Society B 14(1).

    Equation:
        L = -(1/N) * sum_i sum_k y_ik * log( p_ik )

    Pseudocode:
        clip p away from 0 and 1
        return -mean( log( p at the realised outcome ) )
    """
    raise NotImplementedError


def rps(probs, outcomes):
    """Ranked probability score.

    Why: unlike log loss it respects the H < D < A ordering, so predicting a
    draw when the home side wins is penalised less than predicting an away win.
    The standard metric for football forecasts.

    Source: Epstein (1969), Journal of Applied Meteorology 8(6). Argued as the
        correct rule for football specifically in Constantinou & Fenton (2012),
        Journal of Quantitative Analysis in Sports 8(1).

    Equation:
        RPS = 1/(r-1) * sum_{i=1}^{r-1} ( sum_{j=1}^{i} (p_j - y_j) )^2
        with r = 3, so the outer sum runs over the first two cumulatives and
        the divisor is 2

    Pseudocode:
        cum_p <- cumsum(probs, axis=1)
        cum_y <- cumsum(onehot(outcomes), axis=1)
        return mean( sum( (cum_p - cum_y)[:, :2] ** 2, axis=1 ) / 2 )
    """
    raise NotImplementedError


def brier(probs, outcomes):
    """Multiclass Brier score.

    Why: a third proper scoring rule, and the one that decomposes cleanly into
    calibration and refinement — which matters because this project argues those
    two come apart.

    Source: Brier (1950), Monthly Weather Review 78(1). Decomposition: Murphy
        (1973), Journal of Applied Meteorology 12(4).

    Equation:
        BS = (1/N) * sum_i sum_k ( p_ik - y_ik )^2

    Pseudocode:
        return mean( sum( (probs - onehot(outcomes)) ** 2, axis=1 ) )
    """
    raise NotImplementedError


def classwise_ece(probs, outcomes, bins=10):
    """Expected calibration error, computed per outcome class then averaged.

    Why: the threshold rule is meaningless if probabilities do not mean what
    they say. Classwise rather than pooled, because a model can look calibrated
    on average while being badly wrong on draws specifically.

    Source: Naeini, Cooper & Hauskrecht (2015), AAAI — ECE. Classwise variant:
        Kull et al. (2019), NeurIPS. Used as the selection criterion in
        Walsh & Joshi (2024).

    Equation:
        for class k, partition matches into bins B by predicted p_k
        ECE_k = sum_B ( |B| / N ) * | mean_B(y_k) - mean_B(p_k) |
        ECE   = (1/K) * sum_k ECE_k

    Pseudocode:
        for each class k:
            bin matches by p_k
            per bin: |observed frequency - mean predicted|, weighted by bin size
        average across classes
    """
    raise NotImplementedError


def walk_forward_splits(df, n_splits, min_train):
    """Yield expanding-window train/validation/test index sets in date order.

    Why: a random split leaks future form into past matches. An expanding window
    also gives several evaluations instead of one, which a single 70/30 split
    cannot, and matches how the strategy would actually be run.

    Source: Bergmeir & Benitez (2012), Information Sciences 191 — on why
        standard cross-validation is invalid for time-ordered data.

    Equation:
        fold f:  train = [0, t_f),  validation = [t_f, t_f + v),
                 test  = [t_f + v, t_f + v + h)

    Note: never cut inside a date. A split mid-day puts matches from the same
    round on both sides of the boundary — the prototype had 31 such matches.

    Pseudocode:
        boundaries <- unique dates, so no fold splits a date
        for each fold:
            train <- everything before the boundary
            valid <- next v matches, used for w and calibration only
            test  <- next h matches, touched once at the very end
            yield the three index arrays
    """
    raise NotImplementedError


def bootstrap_ci(pnl, match_ids, n_boot=5000, alpha=0.05):
    """Bootstrap confidence interval for mean profit and loss.

    Why: resample *matches*, not individual bets. Bets on the same fixture are
    not independent, and resampling bets understates the interval. Small effect
    on 1X2, large once the Asian handicap market is added.

    Source: Efron (1979), Annals of Statistics 7(1). On why these intervals are
        invalid after selection: Deng (2019), arXiv:1910.03788.

    Equation:
        for b in 1..B:  draw N matches with replacement, R*_b = mean P&L
        CI = [ percentile(R*, alpha/2), percentile(R*, 1 - alpha/2) ]

    Note: not valid conditional on the selection event — the same truncation
    that biases the point estimate distorts the interval. Report with that
    caveat stated in the text, not buried in limitations.

    Pseudocode:
        group P&L by match id
        for each replicate: sample match ids with replacement, pool their bets
        take the mean, collect, return percentiles
    """
    raise NotImplementedError


def market_baseline_model(df):
    """Dummy model returning the de-vigged market price.

    Why: the harness gate. Run the full evaluation against this before any real
    model exists. It should score a log loss near 1.00 on this data — if it does
    not, the harness is wrong and every later number would be wrong with it.

    Equation:
        p_ik = q_ik    (the model is the market)

    Pseudocode:
        return the de-vigged market probabilities unchanged
    """
    raise NotImplementedError


def evaluate(probs, outcomes, odds=None):
    """Score one set of predictions on every metric at once.

    Why: a single call site means model comparisons cannot accidentally use
    different subsets or different metric settings.

    Pseudocode:
        assert probs sum to 1 and align with outcomes
        return {log loss, RPS, Brier, classwise ECE, n}
        if odds given, add flat-stake ROI at tau = 0 for reference
    """
    raise NotImplementedError


# =============================================================================
# E. LEAKAGE AND SANITY TESTS  —  required, not optional
# =============================================================================


def assert_no_dead_components(pipeline, df):
    """Assert every component changes the output when removed.

    Why: the prototype ran a rating engine that silently did nothing. A
    component that changes no output is either broken or pointless, and both
    need finding.

    Source: the general problem is catalogued in Kaufman, Rosset & Perlich
        (2012), ACM TKDD 6(4), "Leakage in data mining".

    Pseudocode:
        baseline <- pipeline(df) predictions
        for each component:
            rerun with that component disabled
            assert predictions differ materially
    """
    raise NotImplementedError


def assert_features_available_at_kickoff(df, feature_cols):
    """Assert no feature uses information published after kickoff.

    Why: shots, cards and corners are recorded post-match. They are legitimate
    inside a lagged rolling average and fatal as direct features. Closing odds
    are the subtler case — they exist before kickoff but not before the bet.

    Pseudocode:
        maintain a whitelist of columns known at opening-price time
        assert every feature derives only from whitelisted columns
        assert no feature column is a same-match post-play statistic
    """
    raise NotImplementedError


def assert_shift_degrades(pipeline, df):
    """Assert that shifting features forward one match makes performance worse.

    Why: the sharpest single leakage test available. If stale features predict
    as well as current ones, the model is reading something it should not.

    Equation:
        loss( model on features shifted +1 match ) > loss( model as built )

    Pseudocode:
        shift every feature forward one match within each team
        refit, rescore
        assert the shifted model is measurably worse
    """
    raise NotImplementedError


def assert_no_suspicious_correlation(df, feature_cols, outcome_col):
    """Flag any feature correlating implausibly highly with the outcome.

    Why: real football features correlate weakly with the market residual —
    0.03 or below in this data. Anything far above that is leakage until proven
    otherwise, not a discovery.

    Pseudocode:
        for each feature: correlate against the market residual
        flag |r| above a threshold set from the EDA distribution
        report rather than raise; a flag needs a human decision
    """
    raise NotImplementedError


# =============================================================================
# F. MODELS
# =============================================================================


def fit_multinomial_logit(X_train, y_train):
    """Multinomial logistic regression over features and market log-odds.

    Why: the transparent baseline, and the model whose coefficients can be read
    directly against the framework.

    Equation:
        P(y = k | x) = exp( beta_k . x ) / sum_j exp( beta_j . x )

    Note: the market must enter as logit(q), not raw q. The whole method is
    defined in log-odds space, so matching the feature space makes the model's
    deviation directly readable and fixes a poor linear fit at long odds. Scale
    features before fitting — Elo spans hundreds while probabilities span 0-1,
    and the L2 penalty does not bear on them evenly otherwise.

    Pseudocode:
        X <- [features, logit(q_H), logit(q_D), logit(q_A)]
        standardise X on training statistics only
        fit multinomial logistic
        return a model exposing predict_proba
    """
    raise NotImplementedError


def fit_gradient_boosting(X_train, y_train):
    """Gradient boosted trees over the same features.

    Why: two models minimum, because the calibration-versus-accuracy comparison
    needs a pool to select between. Gradient boosting is also the published best
    performer on this task, beating recurrent architectures.

    Source: Friedman (2001), Annals of Statistics 29(5). Empirical superiority
        on football prediction: arXiv:2309.14807.

    Note: tree ensembles are typically poorly calibrated out of the box, which
    is what makes section G necessary rather than cosmetic.

    Pseudocode:
        fit with early stopping on the validation fold, never the test fold
        return a model exposing predict_proba
    """
    raise NotImplementedError


# =============================================================================
# G. CALIBRATION
# =============================================================================


def fit_platt(probs_val, outcomes_val):
    """Platt scaling — logistic recalibration fitted on a validation slice.

    Why: a two-parameter monotone correction. Cheap, stable on small validation
    sets, and the right default when data is limited.

    Source: Platt (1999), Advances in Large Margin Classifiers.

    Equation:
        p_calibrated = 1 / ( 1 + exp( A * logit(p) + B ) )
        A, B fitted by maximum likelihood on held-out data

    Pseudocode:
        fit a one-variable logistic of outcome on logit(p), per class
        return a transform to apply to future predictions
    """
    raise NotImplementedError


def fit_isotonic(probs_val, outcomes_val):
    """Isotonic regression — non-parametric monotone recalibration.

    Why: more flexible than Platt but needs more data and can overfit small
    validation sets. Compare both rather than assuming.

    Source: Zadrozny & Elkan (2002), KDD.

    Equation:
        minimise sum_i ( y_i - g(p_i) )^2   subject to g non-decreasing

    Pseudocode:
        fit isotonic per class on validation predictions
        renormalise the three calibrated probabilities to sum to 1
    """
    raise NotImplementedError


def reliability_diagram(probs, outcomes, bins=10):
    """Plot predicted probability against observed frequency.

    Why: the visual form of the calibration argument, and the same construction
    that shows the market's favourite-longshot bias.

    Source: DeGroot & Fienberg (1983), The Statistician 32. Modern usage:
        Niculescu-Mizil & Caruana (2005), ICML.

    Pseudocode:
        bin by predicted probability
        plot mean predicted against observed frequency, with the diagonal
        scale marker area by bin count so thin bins read as uncertain
    """
    raise NotImplementedError


def rank_models_by_calibration_and_accuracy(results):
    """Rank the model pool separately by calibration and by accuracy.

    Why: replicates the finding that the two rankings disagree, which is the
    argument for using calibration to select a model under a threshold rule.

    Source: Walsh & Joshi (2024), Machine Learning with Applications 16:100539,
        and its 2025 corrigendum. Cite the corrigendum; do not quote the
        original ROI figures, which it withdrew.

    Pseudocode:
        rank models by classwise ECE
        rank the same models by accuracy
        report both orderings and their rank correlation
    """
    raise NotImplementedError


# =============================================================================
# H. SHRINKAGE CORRECTION  —  the contribution
# =============================================================================


def model_deviation(probs_model, probs_market):
    """Deviation from the market in log-odds: logit(p) - logit(q).

    Why: the central quantity. This is the disagreement the threshold rule acts
    on, and the thing that must be shrunk before it is acted on.

    Equation:
        delta_hat = logit(p) - logit(q)
        delta_hat = delta + epsilon,  E[epsilon] = 0,  Var(epsilon) = sigma_e^2
        delta is the genuine edge, epsilon the estimation error;
        only their sum is observable

    Note: compute this *after* calibration. Shrinking an uncalibrated deviation
    confounds two corrections and makes the estimated weight uninterpretable.

    Pseudocode:
        clip both probability sets away from 0 and 1
        return logit(p) - logit(q)
    """
    raise NotImplementedError


def build_variance_proxies(df):
    """Assemble the observable features that predict estimation noise.

    Why: heteroscedastic shrinkage needs an x. Candidates measured or available:
    odds level (noise is larger at long odds — the EV filter's longshot share
    rises from 4% to 27% across the threshold sweep), bookmaker disagreement,
    number of books quoting, division, the Elo-versus-ClubElo gap, and how many
    matches of history each team has.

    Equation:
        sigma_e^2(x) modelled as a function of these covariates, fitted by
        regressing squared residuals on x

    Pseudocode:
        assemble the proxy columns
        standardise
        return the design matrix used by estimate_w_heteroscedastic
    """
    raise NotImplementedError


def estimate_w(deviations, outcomes, probs_market):
    """Estimate the constant shrinkage weight by regressing residual on deviation.

    Why: answers H3. w near 0 means the model's disagreements are noise and the
    correct action is not to bet; w near 1 means they can be taken at face
    value.

    Source: the shrinkage form is James & Stein (1961); the empirical Bayes
        reading is Efron & Morris (1975), JASA 70(350).

    Equation:
        w = sigma_d^2 / ( sigma_d^2 + sigma_e^2 )       posterior mean slope
        estimated by OLS:   y_k - q_k = w * (p_k - q_k) + eta

    Note: fit on validation data only. Fitting on test would repeat the exact
    selection error the project is about.

    Pseudocode:
        residual  <- outcome indicator - market probability
        regressor <- model probability - market probability
        fit OLS; report slope, standard error, and t
        also report mean |regressor|, so the reader can judge how narrow the
            range of x was — the prototype's was 0.019, which is very narrow
    """
    raise NotImplementedError


def estimate_w_heteroscedastic(deviations, outcomes, probs_market, proxies):
    """Estimate a shrinkage weight that varies with the variance proxies.

    Why: constant w is the wrong model if noise is larger at long odds, and the
    measured longshot concentration says it is. This is where the correction
    should do most of its work, so it is protected from the cut list.

    Source: bias scaling jointly with sampling variability and selection
        threshold is set out in arXiv:2511.06318 (Bayesian hybrid shrinkage
        for A/B testing).

    Equation:
        sigma_e^2(x) fitted by regressing squared residuals on x
        w(x) = sigma_d^2 / ( sigma_d^2 + sigma_e^2(x) )

    Pseudocode:
        fit a variance model: squared residual ~ proxies, constrained non-negative
        estimate sigma_d^2 as total deviance variance minus mean fitted noise
        return a callable x -> w(x), clipped to [0, 1]
    """
    raise NotImplementedError


def apply_shrinkage(probs_market, deviations, w):
    """Return corrected probabilities: sigmoid( logit(q) + w * delta_hat ).

    Why: shrinks the disagreement toward the market before the threshold is
    applied, rather than discounting realised returns after the fact.

    Equation:
        p_tilde = logit^{-1} ( logit(q) + w * delta_hat )

    Pseudocode:
        m <- logit(q)
        return sigmoid( m + w * delta_hat ), renormalised across the three
            outcomes so the corrected probabilities still sum to 1
    """
    raise NotImplementedError


def tweedie_shrinkage(deviations, sigma_sq):
    """Non-parametric correction via Tweedie's formula.

    Why: drops the normal-prior assumption behind the constant-w form and
    estimates the correction from the observed density of deviations instead.
    The ranked third variant — implement after constant and heteroscedastic w.

    Source: Efron (2011), JASA 106(496), "Tweedie's Formula and Selection
        Bias" — the problem named directly. Behaviour under dependence:
        biorxiv 2023.09.22.558978.

    Equation:
        E[ delta | delta_hat ] = delta_hat
                               + sigma_e^2 * d/d(delta_hat) log f(delta_hat)
        where f is the marginal density of the observed deviations

    Pseudocode:
        estimate f by kernel density or a smooth spline on delta_hat
        differentiate log f numerically
        return delta_hat + sigma_e^2 * dlogf
    """
    raise NotImplementedError


# =============================================================================
# I. BETTING SIMULATION  —  produces the figure
# =============================================================================


def expected_value(probs, odds):
    """Expected profit per unit staked.

    Why: the quantity the threshold is applied to. Note that error scales with
    odds — a fixed probability error becomes a much larger EV error at 7.0 than
    at 1.5, which is the mechanism behind the whole problem.

    Equation:
        EV_ik = p_ik * o_ik - 1

    Pseudocode:
        return probs * odds - 1
    """
    raise NotImplementedError


def select_bets(ev, threshold, one_per_match=True):
    """Return the selected set at a given threshold.

    Why: this is the selection step the project studies. Because q is
    recoverable from o, the rule does not select where the model is confident —
    it selects where the model disagrees with the market by at least a margin.

    Equation:
        d_ik   = 1[ p_ik * o_ik - 1 >= tau ]
        S(tau) = { (i, k) : d_ik = 1 }

    Note: default to one bet per match. Without it the same fixture can be
    backed on two outcomes at once, which is not a coherent strategy and breaks
    the independence the bootstrap assumes.

    Pseudocode:
        mask <- ev >= threshold
        if one_per_match: keep only the highest-EV outcome per match
        return match and outcome indices
    """
    raise NotImplementedError


def simulate_flat_stakes(selected, odds, outcomes):
    """Flat-stake profit and loss over the selected set.

    Why: flat stakes isolate the selection effect. Kelly sizing would mix a
    second, different correction into the result — and fractional Kelly is
    itself a shrinkage, applied to stake rather than to probability.

    Equation:
        R = sum over S(tau) of ( y_ik * o_ik - 1 )
        per bet: profit = o - 1 if the outcome occurs, else -1

    Pseudocode:
        for each selected (match, outcome):
            profit <- odds - 1 if it won else -1
        return the per-bet vector and its match ids, for the bootstrap
    """
    raise NotImplementedError


def threshold_sweep(probs, odds, outcomes, thresholds, match_ids):
    """Run the full sweep and return bets, hit rate, ROI and interval per threshold.

    Why: produces the ROI-versus-threshold curve, the single output the project
    lives on. Run it once for uncorrected probabilities and once for corrected;
    the difference between the two lines is the result.

    Equation:
        for each tau:  n(tau) = |S(tau)|,  ROI(tau) = mean profit over S(tau)
        H1 predicts ROI decreasing in tau under the uncorrected rule
        H2 predicts ROI non-decreasing in tau once corrected

    Note: report the entire sweep, never the best cell — the multiple-testing
    exposure across thresholds is real, and reporting selectively would repeat
    the exact error under study.

    Pseudocode:
        for each tau:
            select, simulate, bootstrap by match
            record n, hit rate, ROI, interval, and P(ROI <= 0)
        return the table and the curve
    """
    raise NotImplementedError


# =============================================================================
# J. SIMULATION STUDY  —  needs no football data; build early
# =============================================================================


def generate_synthetic(n, sigma_eps, seed=None):
    """Generate true probabilities and noisy estimates with controlled noise.

    Why: the hypotheses concern thresholding noisy estimates, which is not a
    fact about football. With sigma known, the bias can be measured exactly
    rather than inferred.

    Equation:
        draw q from a realistic market distribution
        delta   ~ N(0, sigma_d^2)        genuine edge
        epsilon ~ N(0, sigma_e^2)        estimation error
        p_hat = logit^{-1}( logit(q) + delta + epsilon )
        outcomes drawn from p_true = logit^{-1}( logit(q) + delta )

    Pseudocode:
        sample market probabilities matching the observed odds distribution
        add edge and noise in log-odds space
        realise outcomes from the TRUE probabilities, not the estimated ones
        return q, p_true, p_hat, outcomes, and the known sigmas
    """
    raise NotImplementedError


def measure_selection_bias(synthetic, thresholds):
    """Compare predicted against realised return on the selected set.

    Why: demonstrates the mechanism cleanly. The gap between the two is the
    winner's curse, and it should widen as the threshold rises.

    Source: the auction original is Capen, Clapp & Campbell (1971), Journal of
        Petroleum Technology. Accessible treatment: Thaler (1988), Journal of
        Economic Perspectives 2(1).

    Equation:
        E[ delta | delta_hat >= c ] < E[ delta_hat | delta_hat >= c ]
        for every c in the support — the inequality the project rests on

    Pseudocode:
        for each tau:
            select on estimated EV
            record mean predicted EV and mean realised return on that set
            the difference is the bias; plot it against tau
    """
    raise NotImplementedError


def demonstrate_correction(synthetic, thresholds):
    """Show that shrinkage closes the predicted-versus-realised gap.

    Why: if the correction cannot fix the problem where noise is known and
    controlled, it will not fix it on real data. This is the load-bearing result
    if the football intervals turn out too wide to resolve anything.

    Source: precedent for a correction improving realised performance is
        Shi et al. (2016), PLoS Genetics 12(12) — improved across all 14
        diseases tested.

    Pseudocode:
        repeat measure_selection_bias with shrunk estimates
        sweep the TRUE w and the ESTIMATED w separately, so estimation error in
            w is not confounded with the correction itself
        report the residual bias after correction
    """
    raise NotImplementedError


def power_analysis(sigma_eps, effect_size, alpha=0.05):
    """Bets required to detect an effect of a given size.

    Why: decides whether the real data can show anything at all, and therefore
    how much weight the football results can carry. Per-bet profit standard
    deviation is roughly 1.18 at realistic prices.

    Equation:
        n >= ( z_{1 - alpha/2} * sd_per_bet / effect_size )^2
        a 5% edge at two standard errors needs roughly 2,200 bets;
        a 2% edge needs roughly 13,900

    Pseudocode:
        for a grid of effect sizes: required n
        overlay the bet counts actually available per season and per market
        report the smallest effect resolvable with the data in hand
    """
    raise NotImplementedError


def check_correction_not_vacuous(synthetic, thresholds):
    """Verify the corrected curve is not the uncorrected curve rescaled in tau.

    Why: constant-w shrinkage compresses the deviation, so if it merely
    reparametrised the threshold, H2 would be satisfied trivially and mean
    nothing. It should not — shrinking in log-odds is non-linear in probability
    space and odds vary per bet, so the selected sets genuinely differ. Confirm
    that rather than assume it, and confirm it early.

    Pseudocode:
        for a grid of tau, compare S_corrected(tau) against S_uncorrected(tau')
            for the tau' matching it on bet count
        if the two sets coincide, H2 is vacuous and the design needs rethinking
        expect them to differ most at long odds, where shrinkage bites hardest
    """
    raise NotImplementedError


# =============================================================================
# K. PLOTS
# =============================================================================


def plot_roi_vs_threshold(sweep_uncorrected, sweep_corrected):
    """The figure the project lives on.

    Why: two lines — uncorrected declining, corrected flat or rising. Both will
    likely sit below zero, which is expected and not a problem: the claim is
    about the slope, not the level. Every other component exists to make these
    two lines trustworthy.

    Pseudocode:
        plot ROI against tau for both curves with bootstrap bands
        annotate bet counts per point; drop points too thin to read
        mark zero, and state in the caption that the level is not the claim
    """
    raise NotImplementedError


def plot_calibration(results):
    """Reliability diagrams for each model and for the market.

    Why: shows both that the models need calibrating and that the market itself
    is miscalibrated at long odds — the favourite-longshot bias.
    """
    raise NotImplementedError


def plot_simulation_bias(results):
    """Predicted versus realised return against threshold, from synthetic data.

    Why: the clean version of the project's central claim, with noise known
    rather than inferred. The figure to show first if the real-data intervals
    are wide.
    """
    raise NotImplementedError


def plot_w_by_proxy(w_estimates, proxy_name):
    """Estimated shrinkage weight against a variance proxy.

    Why: shows directly that noise varies with odds level and division, which is
    the justification for the heteroscedastic form and the falsifiable
    prediction the theory makes about per-division bias.
    """
    raise NotImplementedError
