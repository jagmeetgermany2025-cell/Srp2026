import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from scipy.optimize import minimize, minimize_scalar

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
COLUMN_RENAMES = {
    "PinnacleH": "PSH", "PinnacleD": "PSD", "PinnacleA": "PSA",
    "PH": "PSH", "PD": "PSD", "PA": "PSA",
    "BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA",
    "BbMxH": "MaxH", "BbMxD": "MaxD", "BbMxA": "MaxA"
}

FEATURE_COLS = [
    'p_shin_H', 'p_shin_D', 'p_shin_A',
    'home_rest', 'away_rest', 'diff_rest_days',
    'home_GF_5', 'home_GA_5', 'home_GD_5',
    'away_GF_5', 'away_GA_5', 'away_GD_5',
    'home_GF_10', 'home_GA_10', 'away_GF_10', 'away_GA_10',
    'diff_roll_GF_5', 'diff_roll_GA_5'
]

VARIANCE_PROXY_COLS = [
    'p_shin_H', 'diff_rest_days', 'home_rest', 'away_rest'
]


# -------------------------------------------------------------------------
# MATHEMATICAL UTILITIES: GEOMETRIC LOG-SPACE SHRINKAGE
# -------------------------------------------------------------------------
def apply_static_shrinkage(p_shin: np.ndarray, p_model: np.ndarray, w: float) -> np.ndarray:
    r"""
    Applies geometric interpolation in log-probability space:
    p_shrunk \propto (p_shin)^(1-w) * (p_model)^w
    """
    w = np.clip(w, 0.0, 1.0)
    eps = 1e-12
    p_shin_c = np.clip(p_shin, eps, 1.0 - eps)
    p_model_c = np.clip(p_model, eps, 1.0 - eps)

    log_p = (1.0 - w) * np.log(p_shin_c) + w * np.log(p_model_c)
    p_unnorm = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
    p_shrunk = p_unnorm / np.sum(p_unnorm, axis=1, keepdims=True)
    # Strictly enforce 64-bit float summation to prevent sklearn UserWarnings
    return (p_shrunk / p_shrunk.sum(axis=1, keepdims=True)).astype(np.float64)


def apply_heteroscedastic_shrinkage(p_shin: np.ndarray, p_model: np.ndarray,
                                     X_proxy: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Applies match-specific geometric shrinkage w(x) = sigmoid(X_proxy @ theta)."""
    eps = 1e-12
    p_shin_c = np.clip(p_shin, eps, 1.0 - eps)
    p_model_c = np.clip(p_model, eps, 1.0 - eps)

    wx = 1.0 / (1.0 + np.exp(-np.clip(X_proxy @ theta, -30, 30)))
    wx = wx[:, np.newaxis]

    log_p = (1.0 - wx) * np.log(p_shin_c) + wx * np.log(p_model_c)
    p_unnorm = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
    p_shrunk = p_unnorm / np.sum(p_unnorm, axis=1, keepdims=True)
    return (p_shrunk / p_shrunk.sum(axis=1, keepdims=True)).astype(np.float64)


def fit_static_shrinkage(p_shin_val: np.ndarray, p_model_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fits scalar weight w on validation data by minimizing multi-class Log Loss."""
    def objective(w):
        p_shrunk = apply_static_shrinkage(p_shin_val, p_model_val, w)
        return log_loss(y_val, p_shrunk)

    res = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded')
    return float(res.x)


def fit_heteroscedastic_shrinkage(p_shin_val: np.ndarray, p_model_val: np.ndarray,
                                   X_proxy_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Fits heteroscedastic parameters theta on validation data."""
    n_features = X_proxy_val.shape[1]
    init_theta = np.zeros(n_features)

    def objective(theta):
        p_shrunk = apply_heteroscedastic_shrinkage(p_shin_val, p_model_val, X_proxy_val, theta)
        return log_loss(y_val, p_shrunk)

    res = minimize(objective, init_theta, method='L-BFGS-B')
    return res.x


# -------------------------------------------------------------------------
# SECTION A-E: SHIN DE-VIGGING UTILITY
# -------------------------------------------------------------------------
def shin_de_vig(odd_h: float, odd_d: float, odd_a: float) -> tuple:
    """Calculates fair probabilities using Shin's (1992, 1993) de-vigging method."""
    if pd.isna(odd_h) or pd.isna(odd_d) or pd.isna(odd_a):
        return np.nan, np.nan, np.nan
    if odd_h <= 1.0 or odd_d <= 1.0 or odd_a <= 1.0:
        return np.nan, np.nan, np.nan

    p_raw = np.array([1.0 / odd_h, 1.0 / odd_d, 1.0 / odd_a])
    beta = p_raw.sum()
    if beta <= 1.0:
        return p_raw[0] / beta, p_raw[1] / beta, p_raw[2] / beta

    z_low, z_high = 0.0, 0.4
    for _ in range(30):
        z = (z_low + z_high) / 2.0
        val = np.sum(np.sqrt(z ** 2 + 4 * (1 - z) * (p_raw ** 2) / beta))
        if val > (2.0 - z):
            z_low = z
        else:
            z_high = z

    p_shin = (np.sqrt(z ** 2 + 4 * (1 - z) * (p_raw ** 2) / beta) - z) / (2 * (1 - z))
    p_shin = p_shin / p_shin.sum()
    return float(p_shin[0]), float(p_shin[1]), float(p_shin[2])


# -------------------------------------------------------------------------
# SECTION F-G: FEATURE ENGINEERING UTILITIES
# -------------------------------------------------------------------------
def calculate_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates exponential moving averages and rest days strictly prior to kickoff."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    shin_results = [shin_de_vig(h, d, a) for h, d, a in zip(df['PSH'], df['PSD'], df['PSA'])]
    df['p_shin_H'] = [r[0] for r in shin_results]
    df['p_shin_D'] = [r[1] for r in shin_results]
    df['p_shin_A'] = [r[2] for r in shin_results]

    for prefix, col in [('home', 'HomeTeam'), ('away', 'AwayTeam')]:
        team_dates = df.groupby(col)['Date'].diff().dt.days.fillna(14)
        df[f'{prefix}_rest'] = np.minimum(team_dates, 14)

    df['diff_rest_days'] = df['home_rest'] - df['away_rest']

    home_df = df[['Date', 'HomeTeam', 'FTHG', 'FTAG']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GF', 'FTAG': 'GA'}
    )
    home_df['is_home'] = True
    home_df['match_id'] = home_df.index

    away_df = df[['Date', 'AwayTeam', 'FTAG', 'FTHG']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GF', 'FTHG': 'GA'}
    )
    away_df['is_home'] = False
    away_df['match_id'] = away_df.index

    matches = pd.concat([home_df, away_df]).sort_values(['Date', 'match_id']).reset_index(drop=True)

    for span in [5, 10]:
        matches[f'GF_{span}'] = matches.groupby('Team')['GF'].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )
        matches[f'GA_{span}'] = matches.groupby('Team')['GA'].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )
        matches[f'GD_{span}'] = matches[f'GF_{span}'] - matches[f'GA_{span}']

    home_stats = matches[matches['is_home']].set_index('match_id')
    away_stats = matches[~matches['is_home']].set_index('match_id')

    for span in [5, 10]:
        df[f'home_GF_{span}'] = home_stats[f'GF_{span}']
        df[f'home_GA_{span}'] = home_stats[f'GA_{span}']
        df[f'home_GD_{span}'] = home_stats[f'GD_{span}']

        df[f'away_GF_{span}'] = away_stats[f'GF_{span}']
        df[f'away_GA_{span}'] = away_stats[f'GA_{span}']
        df[f'away_GD_{span}'] = away_stats[f'GD_{span}']

    df['diff_roll_GF_5'] = df['home_GF_5'] - df['away_GF_5']
    df['diff_roll_GA_5'] = df['home_GA_5'] - df['away_GA_5']

    return df


# -------------------------------------------------------------------------
# SECTION H: MODEL TRAINING & CALIBRATION
# -------------------------------------------------------------------------
def run_model_training(input_csv: str = "data/master_historical_matches_featured.csv",
                       output_csv: str = "data/model_test_predictions.csv"):
    """
    Trains LightGBM model, calibrates probabilities, fits shrinkage parameters w and w(x)
    on Season 22/23 validation data, and evaluates out-of-sample predictions.
    """
    print("\n" + "=" * 50)
    print("SECTION H & G: MODEL TRAINING, CALIBRATION & SHRINKAGE")
    print("=" * 50)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Featured dataset '{input_csv}' not found.")

    df = pd.read_csv(input_csv, low_memory=False)

    target_map = {'H': 0, 'D': 1, 'A': 2}
    df['target'] = df['FTR'].map(target_map)

    required_cols = FEATURE_COLS + VARIANCE_PROXY_COLS + ['target', 'PSH', 'PSD', 'PSA']
    valid_mask = df[required_cols].notna().all(axis=1)
    df_clean = df[valid_mask].copy()

    train_seasons = ["1516", "1617", "1718", "1819", "1920", "2021", "2122"]
    val_seasons = ["2223"]
    test_seasons = ["2324", "2425", "2526"]

    train_df = df_clean[df_clean['Season'].astype(str).isin(train_seasons)]
    val_df = df_clean[df_clean['Season'].astype(str).isin(val_seasons)]
    test_df = df_clean[df_clean['Season'].astype(str).isin(test_seasons)].copy()

    X_train, y_train = train_df[FEATURE_COLS], train_df['target'].astype(int)
    X_val, y_val = val_df[FEATURE_COLS], val_df['target'].astype(int)
    X_test, y_test = test_df[FEATURE_COLS], test_df['target'].astype(int)

    print(f"Training Set   : {len(X_train):,} matches (Seasons 15/16 - 21/22)")
    print(f"Validation Set : {len(X_val):,} matches (Season 22/23)")
    print(f"Test Set (OOS) : {len(X_test):,} matches (Seasons 23/24 - 25/26)")

    base_lgb = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.01, num_leaves=15,
        max_depth=3, subsample=0.7, colsample_bytree=0.7,
        random_state=42, verbosity=-1
    )
    calibrated_model = CalibratedClassifierCV(estimator=base_lgb, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)

    p_shin_val = val_df[['p_shin_H', 'p_shin_D', 'p_shin_A']].values
    p_model_val = calibrated_model.predict_proba(X_val)
    X_proxy_val = val_df[VARIANCE_PROXY_COLS].values

    w_scalar = fit_static_shrinkage(p_shin_val, p_model_val, y_val.values)
    theta_vec = fit_heteroscedastic_shrinkage(p_shin_val, p_model_val, X_proxy_val, y_val.values)

    print(f"\nEstimated Global Shrinkage Weight (w) : {w_scalar:.4f}")

    p_shin_test = test_df[['p_shin_H', 'p_shin_D', 'p_shin_A']].values
    p_model_test = calibrated_model.predict_proba(X_test)
    X_proxy_test = test_df[VARIANCE_PROXY_COLS].values

    p_shrunk_static = apply_static_shrinkage(p_shin_test, p_model_test, w_scalar)
    p_shrunk_hetero = apply_heteroscedastic_shrinkage(p_shin_test, p_model_test, X_proxy_test, theta_vec)

    test_df['p_model_H'], test_df['p_model_D'], test_df['p_model_A'] = p_model_test[:, 0], p_model_test[:, 1], p_model_test[:, 2]
    test_df['p_shrunk_H'], test_df['p_shrunk_D'], test_df['p_shrunk_A'] = p_shrunk_static[:, 0], p_shrunk_static[:, 1], p_shrunk_static[:, 2]
    test_df['p_hetero_H'], test_df['p_hetero_D'], test_df['p_hetero_A'] = p_shrunk_hetero[:, 0], p_shrunk_hetero[:, 1], p_shrunk_hetero[:, 2]

    shin_loss = log_loss(y_test, p_shin_test)
    model_loss = log_loss(y_test, p_model_test)
    shrunk_loss = log_loss(y_test, p_shrunk_static)
    hetero_loss = log_loss(y_test, p_shrunk_hetero)

    print("\n" + "-" * 50)
    print("OUT-OF-SAMPLE LOG LOSS COMPARISON")
    print("-" * 50)
    print(f"Pinnacle Shin Baseline Log Loss       : {shin_loss:.4f}")
    print(f"Raw Calibrated LightGBM Log Loss      : {model_loss:.4f}")
    print(f"James-Stein Shrunk Log Loss (w)       : {shrunk_loss:.4f}")
    print(f"Heteroscedastic Shrunk Log Loss w(x)  : {hetero_loss:.4f}")

    output_cols = [
        'Date', 'Season', 'Div', 'HomeTeam', 'AwayTeam', 'FTR', 'target',
        'PSH', 'PSD', 'PSA',
        'p_shin_H', 'p_shin_D', 'p_shin_A',
        'p_model_H', 'p_model_D', 'p_model_A',
        'p_shrunk_H', 'p_shrunk_D', 'p_shrunk_A',
        'p_hetero_H', 'p_hetero_D', 'p_hetero_A'
    ]

    test_df[output_cols].to_csv(output_csv, index=False)
    print(f"\nSaved out-of-sample predictions to '{output_csv}' successfully!")

    return calibrated_model, w_scalar, theta_vec


# -------------------------------------------------------------------------
# SECTION I: FRACTIONAL KELLY EV BACKTESTING & THRESHOLD SWEEP ENGINE
# -------------------------------------------------------------------------
def run_ev_backtest(input_csv: str = "data/model_test_predictions.csv",
                    output_csv: str = "data/backtest_results.csv",
                    use_shrunk: bool = True,
                    min_ev: float = 0.01,
                    max_odds: float = 3.50,
                    min_edge: float = 0.005,
                    allowed_selections: list = ['H', 'A'],
                    kelly_fraction: float = 0.25,
                    max_stake_cap: float = 3.0):
    """Executes fractional Kelly backtest on raw vs shrunk probability estimates."""
    print("\n" + "=" * 50)
    print(f"SECTION I: BACKTESTING ENGINE (Use Shrunk Probabilities = {use_shrunk})")
    print("=" * 50)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Predictions file '{input_csv}' not found.")

    df = pd.read_csv(input_csv)
    bets = []

    prefix = 'p_shrunk' if use_shrunk else 'p_model'

    for idx, row in df.iterrows():
        outcomes = [
            ('H', row[f'{prefix}_H'], row['p_shin_H'], row['PSH'], 0),
            ('D', row[f'{prefix}_D'], row['p_shin_D'], row['PSD'], 1),
            ('A', row[f'{prefix}_A'], row['p_shin_A'], row['PSA'], 2)
        ]

        for outcome_code, p_est, p_shin, odds, target_code in outcomes:
            if outcome_code not in allowed_selections:
                continue

            if pd.isna(odds) or odds <= 1.0 or odds > max_odds:
                continue

            ev = (p_est * odds) - 1.0
            prob_edge = p_est - p_shin

            if ev >= min_ev and prob_edge >= min_edge:
                full_kelly = ev / (odds - 1.0)
                stake = min(full_kelly * kelly_fraction * 100, max_stake_cap)

                is_win = 1 if row['target'] == target_code else 0
                pnl = (stake * (odds - 1.0)) if is_win else -stake

                bets.append({
                    'Date': row['Date'],
                    'Season': row['Season'],
                    'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    'Selection': outcome_code,
                    'Odds': odds,
                    'p_est': p_est,
                    'p_shin': p_shin,
                    'Edge': prob_edge,
                    'EV': ev,
                    'Stake': round(stake, 2),
                    'IsWin': is_win,
                    'PnL': round(pnl, 2)
                })

    bets_df = pd.DataFrame(bets)

    if len(bets_df) == 0:
        print("No bets met the strict criteria.")
        return

    total_bets = len(bets_df)
    total_staked = bets_df['Stake'].sum()
    total_pnl = bets_df['PnL'].sum()
    roi = (total_pnl / total_staked) * 100
    win_rate = (bets_df['IsWin'].sum() / total_bets) * 100
    avg_stake = bets_df['Stake'].mean()

    print(f"Probability Type     : {'Shrinkage Corrected' if use_shrunk else 'Raw Unshrunk Model'}")
    print(f"Filter Thresholds    : EV >= +{min_ev*100:.1f}%, Edge >= +{min_edge*100:.1f}%")
    print(f"Total Bets Placed    : {total_bets:,}")
    print(f"Total Units Staked   : {total_staked:.2f} units (Avg Stake: {avg_stake:.2f} units)")
    print(f"Win Rate             : {win_rate:.2f}%")
    print(f"Total Profit/Loss    : {total_pnl:+.2f} units")
    print(f"Return on Investment : {roi:+.2f}%")

    bets_df.to_csv(output_csv, index=False)
    print(f"Saved backtest log to '{output_csv}' successfully!")


def run_threshold_sweep(input_csv: str = "data/model_test_predictions.csv"):
    """
    Sweeps thresholds (tau) across both raw and shrinkage-corrected models
    to prove performance degradation on raw noise vs stability on shrunk edges.
    """
    print("\n" + "=" * 50)
    print("SECTION I (EXT): THRESHOLD SWEEP COMPARISON")
    print("=" * 50)

    if not os.path.exists(input_csv):
        return

    df = pd.read_csv(input_csv)
    thresholds = [0.00, 0.005, 0.01, 0.015, 0.02, 0.03]

    results = []

    for tau in thresholds:
        for use_shrunk in [False, True]:
            prefix = 'p_shrunk' if use_shrunk else 'p_model'
            staked, pnl, count = 0.0, 0.0, 0

            for _, row in df.iterrows():
                for outcome_code, p_est, p_shin, odds, target_code in [
                    ('H', row[f'{prefix}_H'], row['p_shin_H'], row['PSH'], 0),
                    ('A', row[f'{prefix}_A'], row['p_shin_A'], row['PSA'], 2)
                ]:
                    if pd.isna(odds) or odds <= 1.0 or odds > 3.50:
                        continue

                    ev = (p_est * odds) - 1.0
                    edge = p_est - p_shin

                    if ev >= tau and edge >= (tau / 2.0):
                        stake = min((ev / (odds - 1.0)) * 0.25 * 100, 3.0)
                        is_win = 1 if row['target'] == target_code else 0
                        profit = (stake * (odds - 1.0)) if is_win else -stake

                        staked += stake
                        pnl += profit
                        count += 1

            roi = (pnl / staked * 100) if staked > 0 else 0.0
            results.append({
                'Tau (Min EV)': f"+{tau*100:.1f}%",
                'Type': 'Shrunk' if use_shrunk else 'Raw Model',
                'Bets': count,
                'Staked': round(staked, 2),
                'PnL': round(pnl, 2),
                'ROI (%)': round(roi, 2)
            })

    sweep_df = pd.DataFrame(results)
    print(sweep_df.to_string(index=False))


# -------------------------------------------------------------------------
# SECTION J: LIVE BET SLATE GENERATOR WITH SHRINKAGE
# -------------------------------------------------------------------------
def predict_upcoming(model, w_scalar: float,
                     historical_csv: str = "data/master_historical_matches_featured.csv",
                     upcoming_csv: str = "data/upcoming_fixtures.csv",
                     output_csv: str = "data/live_bet_slate.csv",
                     min_ev: float = 0.01, max_odds: float = 3.50, min_edge: float = 0.005,
                     kelly_fraction: float = 0.25, max_stake_cap: float = 3.0):
    """Generates shrinkage-corrected live bet recommendations."""
    print("\n" + "=" * 50)
    print("SECTION J: LIVE BET SLATE GENERATOR (WITH SHRINKAGE)")
    print("=" * 50)

    if not os.path.exists(upcoming_csv) or not os.path.exists(historical_csv):
        print("Required CSV files missing. Skipping live predictions.")
        return

    df_upcoming = pd.read_csv(upcoming_csv)
    df_hist = pd.read_csv(historical_csv, low_memory=False)

    if df_upcoming.empty:
        print("Upcoming fixtures file is empty.")
        return

    df_hist['is_upcoming'] = False
    df_upcoming['is_upcoming'] = True

    combined_df = pd.concat([df_hist, df_upcoming], ignore_index=True)
    combined_featured = calculate_rolling_features(combined_df)

    df_valid = combined_featured[combined_featured['is_upcoming'] == True].copy()
    valid_mask = df_valid[FEATURE_COLS].notna().all(axis=1)
    df_valid = df_valid[valid_mask].copy()

    if df_valid.empty:
        print("No valid upcoming fixtures with full features.")
        return

    probs_raw = model.predict_proba(df_valid[FEATURE_COLS])
    p_shin_upcoming = df_valid[['p_shin_H', 'p_shin_D', 'p_shin_A']].values

    p_shrunk_upcoming = apply_static_shrinkage(p_shin_upcoming, probs_raw, w_scalar)

    df_valid['p_shrunk_H'] = p_shrunk_upcoming[:, 0]
    df_valid['p_shrunk_D'] = p_shrunk_upcoming[:, 1]
    df_valid['p_shrunk_A'] = p_shrunk_upcoming[:, 2]

    trade_slate = []
    for idx, row in df_valid.iterrows():
        outcomes = [
            ('H', row['p_shrunk_H'], row['p_shin_H'], row['PSH']),
            ('A', row['p_shrunk_A'], row['p_shin_A'], row['PSA'])
        ]

        for outcome_code, p_shrunk, p_shin, odds in outcomes:
            if pd.isna(odds) or odds <= 1.0 or odds > max_odds:
                continue

            ev = (p_shrunk * odds) - 1.0
            prob_edge = p_shrunk - p_shin

            if ev >= min_ev and prob_edge >= min_edge:
                full_kelly = ev / (odds - 1.0)
                stake = min(full_kelly * kelly_fraction * 100, max_stake_cap)

                trade_slate.append({
                    'Date': row['Date'],
                    'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    'Selection': outcome_code,
                    'Odds': odds,
                    'p_shrunk': round(p_shrunk, 4),
                    'p_shin': round(p_shin, 4),
                    'Edge': round(prob_edge, 4),
                    'EV': round(ev, 4),
                    'Recommended_Stake_Units': round(stake, 2)
                })

    slate_df = pd.DataFrame(trade_slate)
    if not slate_df.empty:
        print("\n" + "-" * 40)
        print("SHRUNK ACTIVE VALUE BET SLATE")
        print("-" * 40)
        print(slate_df.to_string(index=False))
        slate_df.to_csv(output_csv, index=False)
        print(f"\nSaved shrunk live slate ({len(slate_df)} bets) to '{output_csv}' successfully!")
    else:
        print("No upcoming fixtures met the shrinkage-corrected EV criteria.")


# -------------------------------------------------------------------------
# MAIN EXECUTION FLOW
# -------------------------------------------------------------------------
if __name__ == "__main__":
    featured_path = "data/master_historical_matches_featured.csv"
    predictions_path = "data/model_test_predictions.csv"

    trained_model, w_scalar, theta_vec = run_model_training(
        input_csv=featured_path,
        output_csv=predictions_path
    )

    print("\n--- COMPARATIVE BACKTEST RUNS ---")
    run_ev_backtest(input_csv=predictions_path, use_shrunk=False, min_ev=0.01, min_edge=0.005, output_csv="data/backtest_unshrunk.csv")
    run_ev_backtest(input_csv=predictions_path, use_shrunk=True, min_ev=0.01, min_edge=0.005, output_csv="data/backtest_shrunk.csv")

    run_threshold_sweep(input_csv=predictions_path)

    predict_upcoming(
        model=trained_model,
        w_scalar=w_scalar,
        upcoming_csv="data/upcoming_fixtures.csv",
        output_csv="data/live_bet_slate.csv",
        min_ev=0.01,
        min_edge=0.005
    )
