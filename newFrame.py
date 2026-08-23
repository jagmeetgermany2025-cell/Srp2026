import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

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

    # Numerical solver for z (insider trading proportion)
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

    # Parse dates safely without warnings
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    # Compute Shin probabilities for every row
    shin_results = [shin_de_vig(h, d, a) for h, d, a in zip(df['PSH'], df['PSD'], df['PSA'])]
    df['p_shin_H'] = [r[0] for r in shin_results]
    df['p_shin_D'] = [r[1] for r in shin_results]
    df['p_shin_A'] = [r[2] for r in shin_results]

    # Calculate Rest Days
    for prefix, col in [('home', 'HomeTeam'), ('away', 'AwayTeam')]:
        team_dates = df.groupby(col)['Date'].diff().dt.days.fillna(14)
        df[f'{prefix}_rest'] = np.minimum(team_dates, 14)

    df['diff_rest_days'] = df['home_rest'] - df['away_rest']

    # Melt dataframe to team-match level to compute rolling features cleanly
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

    # Compute exponentially weighted stats shifted by 1 match (prevents target leakage)
    for span in [5, 10]:
        matches[f'GF_{span}'] = matches.groupby('Team')['GF'].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )
        matches[f'GA_{span}'] = matches.groupby('Team')['GA'].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )
        matches[f'GD_{span}'] = matches[f'GF_{span}'] - matches[f'GA_{span}']

    # Map features back directly by match_id to prevent duplicate rows from merges
    home_stats = matches[matches['is_home']].set_index('match_id')
    away_stats = matches[~matches['is_home']].set_index('match_id')

    for span in [5, 10]:
        df[f'home_GF_{span}'] = home_stats[f'GF_{span}']
        df[f'home_GA_{span}'] = home_stats[f'GA_{span}']
        df[f'home_GD_{span}'] = home_stats[f'GD_{span}']

        df[f'away_GF_{span}'] = away_stats[f'GF_{span}']
        df[f'away_GA_{span}'] = away_stats[f'GA_{span}']
        df[f'away_GD_{span}'] = away_stats[f'GD_{span}']

    # Matchup Differentials
    df['diff_roll_GF_5'] = df['home_GF_5'] - df['away_GF_5']
    df['diff_roll_GA_5'] = df['home_GA_5'] - df['away_GA_5']

    return df

# -------------------------------------------------------------------------
# SECTION H: MODEL TRAINING & CALIBRATION
# -------------------------------------------------------------------------
def run_model_training(input_csv: str = "data/master_historical_matches_featured.csv",
                       output_csv: str = "data/model_test_predictions.csv"):
    """
    Trains a Calibrated LightGBM model on historical seasons incorporating
    Pinnacle market implied probabilities directly. Returns trained model.
    """
    print("\n" + "=" * 50)
    print("SECTION H: MODEL TRAINING & CALIBRATION")
    print("=" * 50)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Featured dataset '{input_csv}' not found. Run pipeline feature engineering first.")

    df = pd.read_csv(input_csv, low_memory=False)

    # Map target variable (0: Home Win, 1: Draw, 2: Away Win)
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df['target'] = df['FTR'].map(target_map)

    # Clean dataset for valid entries
    required_cols = FEATURE_COLS + ['target', 'PSH', 'PSD', 'PSA']
    valid_mask = df[required_cols].notna().all(axis=1)
    df_clean = df[valid_mask].copy()

    print(f"Total Valid Matches for Modeling: {len(df_clean):,}")

    # Temporal Split: Train on historical seasons, test on out-of-sample modern seasons
    train_seasons = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223"]
    test_seasons = ["2324", "2425", "2526"]

    train_df = df_clean[df_clean['Season'].astype(str).isin(train_seasons)]
    test_df = df_clean[df_clean['Season'].astype(str).isin(test_seasons)].copy()

    X_train, y_train = train_df[FEATURE_COLS], train_df['target'].astype(int)
    X_test, y_test = test_df[FEATURE_COLS], test_df['target'].astype(int)

    print(f"Training Set  : {len(X_train):,} matches (Seasons 15/16 - 22/23)")
    print(f"Test Set (OOS): {len(X_test):,} matches (Seasons 23/24 - 25/26)")

    # Train LightGBM + Isotonic Calibration
    print("\nTraining LightGBM Classifier with 3-Fold Isotonic Calibration...")
    base_lgb = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.01,
        num_leaves=15,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        verbosity=-1
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_lgb,
        method='isotonic',
        cv=3
    )
    calibrated_model.fit(X_train, y_train)

    # Generate Out-of-Sample Predictions
    model_probs_raw = calibrated_model.predict_proba(X_test)

    # Probability normalization & clipping
    model_probs_clipped = np.clip(model_probs_raw, 1e-15, 1 - 1e-15)
    model_probs = model_probs_clipped / model_probs_clipped.sum(axis=1, keepdims=True)

    test_df['p_model_H'] = model_probs[:, 0]
    test_df['p_model_D'] = model_probs[:, 1]
    test_df['p_model_A'] = model_probs[:, 2]

    # Evaluate Log Loss against Shin Baseline
    shin_probs_raw = test_df[['p_shin_H', 'p_shin_D', 'p_shin_A']].values
    shin_probs_clipped = np.clip(shin_probs_raw, 1e-15, 1 - 1e-15)
    shin_probs = shin_probs_clipped / shin_probs_clipped.sum(axis=1, keepdims=True)

    shin_loss = log_loss(y_test, shin_probs)
    model_loss = log_loss(y_test, model_probs)

    print("\n" + "-" * 40)
    print("OUT-OF-SAMPLE EVALUATION RESULTS")
    print("-" * 40)
    print(f"Pinnacle Shin Baseline Log Loss : {shin_loss:.4f}")
    print(f"Calibrated LightGBM Log Loss    : {model_loss:.4f}")

    loss_diff = model_loss - shin_loss
    if loss_diff > 0:
        print(f"Interpretation: Model Log Loss is +{loss_diff:.4f} higher than Shin (Expected baseline).")
    else:
        print(f"Interpretation: Model outperforms Shin baseline by {abs(loss_diff):.4f}!")

    output_cols = [
        'Date', 'Season', 'Div', 'HomeTeam', 'AwayTeam', 'FTR', 'target',
        'PSH', 'PSD', 'PSA',
        'p_shin_H', 'p_shin_D', 'p_shin_A',
        'p_model_H', 'p_model_D', 'p_model_A'
    ]

    test_df[output_cols].to_csv(output_csv, index=False)
    print(f"\nSaved out-of-sample predictions to '{output_csv}' successfully!")

    return calibrated_model


# -------------------------------------------------------------------------
# SECTION I: FRACTIONAL KELLY EV BACKTESTING ENGINE
# -------------------------------------------------------------------------
def run_ev_backtest(input_csv: str = "data/model_test_predictions.csv",
                    output_csv: str = "data/backtest_results.csv",
                    min_ev: float = 0.03,
                    max_odds: float = 3.50,
                    min_edge: float = 0.02,
                    allowed_selections: list = ['H', 'A'],
                    kelly_fraction: float = 0.25,
                    max_stake_cap: float = 3.0):
    """
    Executes backtest with Fractional Kelly Staking:
    1. Focuses strictly on Home/Away markets (excluding Draws).
    2. Odds ceiling at 3.50 to remove longshot variance.
    3. Minimum probability edge vs Shin probabilities.
    4. Dynamically scales position size using Quarter-Kelly (0.25).
    """
    print("\n" + "=" * 50)
    print("SECTION I: FRACTIONAL KELLY EV BACKTESTING")
    print("=" * 50)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Predictions file '{input_csv}' not found. Run model training first.")

    df = pd.read_csv(input_csv)
    bets = []

    for idx, row in df.iterrows():
        outcomes = [
            ('H', row['p_model_H'], row['p_shin_H'], row['PSH'], 0),
            ('D', row['p_model_D'], row['p_shin_D'], row['PSD'], 1),
            ('A', row['p_model_A'], row['p_shin_A'], row['PSA'], 2)
        ]

        for outcome_code, p_model, p_shin, odds, target_code in outcomes:
            if outcome_code not in allowed_selections:
                continue

            if pd.isna(odds) or odds <= 1.0 or odds > max_odds:
                continue

            ev = (p_model * odds) - 1.0
            prob_edge = p_model - p_shin

            if ev >= min_ev and prob_edge >= min_edge:
                # Full Kelly calculation: f* = EV / (Odds - 1)
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
                    'p_model': p_model,
                    'p_shin': p_shin,
                    'Edge': prob_edge,
                    'EV': ev,
                    'Stake': round(stake, 2),
                    'IsWin': is_win,
                    'PnL': round(pnl, 2)
                })

    bets_df = pd.DataFrame(bets)

    if len(bets_df) == 0:
        print("No bets met the strict criteria. Consider tuning min_ev or min_edge.")
        return

    total_bets = len(bets_df)
    total_staked = bets_df['Stake'].sum()
    total_pnl = bets_df['PnL'].sum()
    roi = (total_pnl / total_staked) * 100
    win_rate = (bets_df['IsWin'].sum() / total_bets) * 100
    avg_stake = bets_df['Stake'].mean()

    print(
        f"Constraints          : EV >= +{min_ev * 100:.1f}%, Odds <= {max_odds}, Edge >= +{min_edge * 100:.1f}%, Markets: {allowed_selections}")
    print(f"Staking Strategy     : Quarter-Kelly ({kelly_fraction}x), Max Stake Cap: {max_stake_cap} units")
    print(f"Total Bets Placed    : {total_bets:,}")
    print(f"Total Units Staked   : {total_staked:.2f} units (Avg Stake: {avg_stake:.2f} units)")
    print(f"Win Rate             : {win_rate:.2f}%")
    print(f"Total Profit/Loss    : {total_pnl:+.2f} units")
    print(f"Return on Investment : {roi:+.2f}%")

    bets_df.to_csv(output_csv, index=False)
    print(f"\nSaved trade-level backtest log to '{output_csv}' successfully!")


# -------------------------------------------------------------------------
# SECTION J: LIVE BET SLATE GENERATOR
# -------------------------------------------------------------------------
def predict_upcoming(model,
                     historical_csv: str = "data/master_historical_matches_featured.csv",
                     upcoming_csv: str = "data/upcoming_fixtures.csv",
                     output_csv: str = "data/live_bet_slate.csv",
                     min_ev: float = 0.03, max_odds: float = 3.50, min_edge: float = 0.02,
                     kelly_fraction: float = 0.25, max_stake_cap: float = 3.0):
    """
    Evaluates unplayed upcoming fixtures by merging them with historical context
    to properly generate rolling feature statistics before running model inference.
    """
    print("\n" + "=" * 50)
    print("SECTION J: LIVE BET SLATE GENERATOR")
    print("=" * 50)

    if not os.path.exists(upcoming_csv):
        print(f"No upcoming fixtures file found at '{upcoming_csv}'. Skipping live predictions.")
        return

    df_upcoming = pd.read_csv(upcoming_csv)
    if df_upcoming.empty:
        print("Upcoming fixtures file is empty.")
        return

    # Load historical matches to provide feature context
    if not os.path.exists(historical_csv):
        print(f"Historical context file '{historical_csv}' missing. Cannot generate features.")
        return

    df_hist = pd.read_csv(historical_csv, low_memory=False)

    # Label datasets before concatenation
    df_hist['is_upcoming'] = False
    df_upcoming['is_upcoming'] = True

    # Combine historical and upcoming matches to give rolling features complete context
    combined_df = pd.concat([df_hist, df_upcoming], ignore_index=True)

    # Recalculate rolling features across full timeline
    combined_featured = calculate_rolling_features(combined_df)

    # Filter back down strictly to upcoming fixtures
    df_valid = combined_featured[combined_featured['is_upcoming'] == True].copy()

    # Drop rows with missing features
    valid_mask = df_valid[FEATURE_COLS].notna().all(axis=1)
    df_valid = df_valid[valid_mask].copy()

    if df_valid.empty:
        print("No valid upcoming fixtures with complete feature sets after historical lookup.")
        return

    # Model inference
    probs_raw = model.predict_proba(df_valid[FEATURE_COLS])
    probs_clipped = np.clip(probs_raw, 1e-15, 1 - 1e-15)
    probs = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)

    df_valid['p_model_H'] = probs[:, 0]
    df_valid['p_model_D'] = probs[:, 1]
    df_valid['p_model_A'] = probs[:, 2]

    trade_slate = []
    for idx, row in df_valid.iterrows():
        outcomes = [
            ('H', row['p_model_H'], row['p_shin_H'], row['PSH']),
            ('A', row['p_model_A'], row['p_shin_A'], row['PSA'])
        ]

        for outcome_code, p_model, p_shin, odds in outcomes:
            if pd.isna(odds) or odds <= 1.0 or odds > max_odds:
                continue

            ev = (p_model * odds) - 1.0
            prob_edge = p_model - p_shin

            if ev >= min_ev and prob_edge >= min_edge:
                full_kelly = ev / (odds - 1.0)
                stake = min(full_kelly * kelly_fraction * 100, max_stake_cap)

                trade_slate.append({
                    'Date': row['Date'],
                    'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    'Selection': outcome_code,
                    'Odds': odds,
                    'p_model': round(p_model, 4),
                    'p_shin': round(p_shin, 4),
                    'Edge': round(prob_edge, 4),
                    'EV': round(ev, 4),
                    'Recommended_Stake_Units': round(stake, 2)
                })

    slate_df = pd.DataFrame(trade_slate)
    if not slate_df.empty:
        print("\n" + "-" * 40)
        print("ACTIVE VALUE BET SLATE")
        print("-" * 40)
        print(slate_df.to_string(index=False))
        slate_df.to_csv(output_csv, index=False)
        print(f"\nSaved live slate ({len(slate_df)} bets) to '{output_csv}' successfully!")
    else:
        print("No upcoming fixtures met the required EV/Edge criteria.")


# -------------------------------------------------------------------------
# MAIN EXECUTION FLOW
# -------------------------------------------------------------------------
if __name__ == "__main__":
    featured_path = "data/master_historical_matches_featured.csv"
    predictions_path = "data/model_test_predictions.csv"

    # 1. Execute Model Training
    trained_model = run_model_training(
        input_csv=featured_path,
        output_csv=predictions_path
    )

    # 2. Execute Quarter-Kelly Backtest
    run_ev_backtest(
        input_csv=predictions_path,
        min_ev=0.03,  # Minimum +3% EV
        max_odds=3.50,  # Cap at 3.50 odds
        min_edge=0.02,  # Minimum +2% probability edge over Shin
        allowed_selections=['H', 'A'],  # Exclude Draws
        kelly_fraction=0.25,  # Quarter-Kelly staking
        max_stake_cap=3.0  # Maximum 3.0 units per trade
    )

    # 3. Predict Upcoming Fixtures
    predict_upcoming(
        model=trained_model,
        upcoming_csv="data/upcoming_fixtures.csv",
        output_csv="data/live_bet_slate.csv"
    )
