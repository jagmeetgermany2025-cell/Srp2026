import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy.optimize import brentq
import pandas as pd
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("--- Ultimate Walk-Forward Pipeline with Rolling Window & Strict Controls (CORRECTED) ---\n")

MARKET_WEIGHT_DEFAULT = 0.60
MARKET_WEIGHT_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
INITIAL_BANKROLL = 100.0
ALL_SEASONS = [2223, 2324, 2425, 2526]
N_BOOTSTRAP = 5000

# ROLLING WINDOW CONFIGURATION
ROLLING_WINDOW_SIZE = 2

# STRICT GLOBAL THRESHOLD FLOOR
GLOBAL_LO_ODDS = 1.60
GLOBAL_HI_ODDS = 2.05
GLOBAL_MIN_PWIN = 0.58
GLOBAL_MIN_EV = 0.08


def devig_shin(odds_vec):
    if any(np.isnan(odds_vec)) or any(o <= 1.0 for o in odds_vec):
        return np.array([1 / 3, 1 / 3, 1 / 3])
    beta = 1.0 / np.array(odds_vec)
    S = np.sum(beta)
    if S <= 1.0:
        return beta / S

    def shin_obj(z):
        p = (np.sqrt(z ** 2 + 4 * (1 - z) * (beta ** 2) / S) - z) / (2 * (1 - z))
        return np.sum(p) - 1.0

    try:
        z_opt = brentq(shin_obj, 1e-6, 0.9999)
        p_true = (np.sqrt(z_opt ** 2 + 4 * (1 - z_opt) * (beta ** 2) / S) - z_opt) / (2 * (1 - z_opt))
        return p_true / np.sum(p_true)
    except Exception:
        return beta / S


# Load dataset with semicolon separator
df = pd.read_csv("matches_multiseason.csv", sep=';', low_memory=False, on_bad_lines='skip')

if 'Div' not in df.columns:
    possible_div_cols = ['league', 'League', 'Competition', 'competition', 'Division', 'division', 'Tournament']
    found = False
    for col in possible_div_cols:
        if col in df.columns:
            df = df.rename(columns={col: 'Div'})
            found = True
            break
    if not found:
        raise ValueError("Expected 'Div' column, but could not find a matching league column.")

# ============================================================
# CHANGE #3: merge in real per-match weather (from merge_weather_v2.py /
# Open-Meteo), replacing the constant-default weather block further down.
# Two-tier merge:
#   1) exact (HomeTeam, Date, kickoff hour) match -- most precise
#   2) for rows where (1) fails (missing/unparseable Time, or the exact
#      hour wasn't in the fetched range), fall back to that team+date's
#      daily-mean weather instead of silently reverting to the global
#      constant. This avoids the NaN-merge-key artifact where a missing
#      Time value would otherwise cause a real, fetched weather record
#      to be discarded purely because NaN != NaN in a merge key.
# Rows with NO weather data at all (e.g. geocoding failed for that team)
# still fall through to the constant defaults in the block below, same
# as before this change -- this only recovers cases where real data
# exists but the naive merge would have missed it.
# ============================================================
import os as _os  # already imported as `os` above; aliasing only for local clarity in this block
_weather_path = "weather_by_match.csv"
if _os.path.exists(_weather_path):
    weather_df = pd.read_csv(_weather_path)

    def _parse_hour(t):
        try:
            return int(str(t).split(':')[0])
        except Exception:
            return np.nan
    df['match_hour'] = df['Time'].apply(_parse_hour) if 'Time' in df.columns else np.nan

    # Tier 1: exact hour match
    exact = weather_df.rename(columns={'wind_speed_mph': 'wind_speed_mph_exact',
                                        'precipitation_mm': 'precipitation_mm_exact',
                                        'apparent_temp': 'apparent_temp_exact'})
    df = df.merge(exact, left_on=['HomeTeam', 'Date', 'match_hour'],
                  right_on=['HomeTeam', 'Date', 'Hour'], how='left')
    df = df.drop(columns=['Hour'], errors='ignore')

    # Tier 2: daily mean fallback per (HomeTeam, Date), for rows tier 1 missed
    daily_fallback = weather_df.groupby(['HomeTeam', 'Date'], as_index=False).agg(
        wind_speed_mph_daily=('wind_speed_mph', 'mean'),
        precipitation_mm_daily=('precipitation_mm', 'mean'),
        apparent_temp_daily=('apparent_temp', 'mean'))
    df = df.merge(daily_fallback, on=['HomeTeam', 'Date'], how='left')

    df['wind_speed_mph'] = df['wind_speed_mph_exact'].fillna(df['wind_speed_mph_daily'])
    df['precipitation_mm'] = df['precipitation_mm_exact'].fillna(df['precipitation_mm_daily'])
    df['apparent_temp'] = df['apparent_temp_exact'].fillna(df['apparent_temp_daily'])
    df = df.drop(columns=['wind_speed_mph_exact', 'precipitation_mm_exact', 'apparent_temp_exact',
                           'wind_speed_mph_daily', 'precipitation_mm_daily', 'apparent_temp_daily',
                           'match_hour'], errors='ignore')

    n_real = df['wind_speed_mph'].notna().sum()
    print(f"Real weather merged: {n_real}/{len(df)} rows ({n_real/len(df)*100:.1f}%) "
          f"have genuine weather data; remaining rows fall back to constant defaults below.")
else:
    print(f"NOTE: '{_weather_path}' not found -- using constant weather defaults for all rows "
          f"(run merge_weather_v2.py first to use real weather data).")

cols_to_coerce = [
    'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
    'AvgH', 'AvgD', 'AvgA', 'MaxH', 'MaxD', 'MaxA', 'PSCH', 'PSCD', 'PSCA'
]
for col in cols_to_coerce:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

required = ['AvgH', 'AvgD', 'AvgA', 'FTR', 'Season', 'FTHG', 'FTAG', 'HS', 'AS', 'Div']
df = df.dropna(subset=required).copy()

inefficient_leagues = [div for div in df['Div'].unique() if div not in ['E0', 'SP1', 'I1', 'D1', 'F1']]
df = df[df['Div'].isin(inefficient_leagues)].copy()
print(f"Filtered dataset to {len(df)} matches across inefficient/lower divisions.")

div_dummies = pd.get_dummies(df['Div'], prefix='div', dtype=np.float64)
df = pd.concat([df, div_dummies], axis=1)
div_feature_cols = list(div_dummies.columns)

df['BestH'] = df['MaxH'].fillna(df['AvgH']) if 'MaxH' in df.columns else df['AvgH']
df['BestD'] = df['MaxD'].fillna(df['AvgD']) if 'MaxD' in df.columns else df['AvgD']
df['BestA'] = df['MaxA'].fillna(df['AvgA']) if 'MaxA' in df.columns else df['AvgA']

shin_probs = np.array([devig_shin([r.AvgH, r.AvgD, r.AvgA]) for r in df.itertuples()])
df['p_market_H'], df['p_market_D'], df['p_market_A'] = shin_probs[:, 0], shin_probs[:, 1], shin_probs[:, 2]

df['edge_H'] = df['BestH'] / df['AvgH']
df['edge_D'] = df['BestD'] / df['AvgD']
df['edge_A'] = df['BestA'] / df['AvgA']

# ============================================================
# FIX #2: Closing-line / CLV handling.
# PSCH/PSCD/PSCA (Pinnacle closing odds) are only known once the
# match has kicked off / betting has closed. They must NEVER be
# used as a predictive feature (that is look-ahead leakage), only
# for a post-hoc "did we beat the closing line" (CLV) diagnostic.
# We also stop silently faking close_H = AvgH when PSC* is missing
# for a row -- that produced a fake CLV of exactly 0.00%.
# ============================================================
if 'PSCH' in df.columns and 'PSCD' in df.columns and 'PSCA' in df.columns:
    has_close = df[['PSCH', 'PSCD', 'PSCA']].notna().all(axis=1)
    df['close_H'] = np.where(has_close, df['PSCH'], np.nan)
    df['close_D'] = np.where(has_close, df['PSCD'], np.nan)
    df['close_A'] = np.where(has_close, df['PSCA'], np.nan)
    df['has_closing_line'] = has_close
else:
    df['close_H'] = df['close_D'] = df['close_A'] = np.nan
    df['has_closing_line'] = False

print("Closing-line (CLV) data coverage by season:")
print(df.groupby('Season')['has_closing_line'].mean().apply(lambda x: f"{x * 100:.1f}%"))
print()

# Referee Disciplinary Metrics
if 'Referee' in df.columns and 'HF' in df.columns:
    df['total_fouls'] = df['HF'].fillna(0) + df['AF'].fillna(0)
    df['total_cards'] = (df['HY'].fillna(0) + df['AY'].fillna(0)) + 2 * (df['HR'].fillna(0) + df['AR'].fillna(0))
    df['ref_fouls_mean'] = df.groupby('Referee')['total_fouls'].transform(lambda x: x.shift(1).expanding().mean()).fillna(25.0)
    df['ref_cards_mean'] = df.groupby('Referee')['total_cards'].transform(lambda x: x.shift(1).expanding().mean()).fillna(4.0)
else:
    df['ref_fouls_mean'] = 25.0
    df['ref_cards_mean'] = 4.0

# Weather Disruption Thresholds -- fallback constants for any row that still
# has no weather value after the real-weather merge above (e.g. a team whose
# geocoding failed, or a date outside the API's coverage). This block is
# unchanged from before; it now only fires for genuinely unmatched rows.
if 'wind_speed_mph' not in df.columns: df['wind_speed_mph'] = 8.0
if 'precipitation_mm' not in df.columns: df['precipitation_mm'] = 0.0
if 'apparent_temp' not in df.columns: df['apparent_temp'] = 60.0
df['wind_speed_mph'] = df['wind_speed_mph'].fillna(8.0)
df['precipitation_mm'] = df['precipitation_mm'].fillna(0.0)
df['apparent_temp'] = df['apparent_temp'].fillna(60.0)

df['high_wind_flag'] = np.where(df['wind_speed_mph'] > 15.0, 1, 0)
df['heavy_precip_flag'] = np.where(df['precipitation_mm'] > 2.5, 1, 0)
df['extreme_temp_flag'] = np.where((df['apparent_temp'] < 20.0) | (df['apparent_temp'] > 95.0), 1, 0)

raw_passing_decay = np.where(df['wind_speed_mph'] > 15.0, 1.0 - 0.015 * (df['wind_speed_mph'] - 15.0), 1.0)
df['passing_decay_coef'] = np.clip(raw_passing_decay, 0.75, 1.0)
df['scoring_weather_multiplier'] = 1.0 - (0.05 * df['heavy_precip_flag']) - (0.03 * df['high_wind_flag'])

target_map = {'H': 0, 'D': 1, 'A': 2}
df['target_cls'] = df['FTR'].map(target_map)
df = df.dropna(subset=['target_cls']).copy()
df['target_cls'] = df['target_cls'].astype(int)
df = df.sort_values(['Season', 'Date']).reset_index(drop=True)

df['h_pts'] = np.where(df['FTR'] == 'H', 3, np.where(df['FTR'] == 'D', 1, 0))
df['a_pts'] = np.where(df['FTR'] == 'A', 3, np.where(df['FTR'] == 'D', 1, 0))
df['h_xg'] = (df['FTHG'] * 0.4 + df['HS'] * 0.08) * df['scoring_weather_multiplier']
df['a_xg'] = (df['FTAG'] * 0.4 + df['AS'] * 0.08) * df['scoring_weather_multiplier']

df['h_gd_10'] = (df.groupby(['Season', 'Div', 'HomeTeam'])['FTHG'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean())
                 - df.groupby(['Season', 'Div', 'HomeTeam'])['FTAG'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()))

# ============================================================
# FIX #1: a_gd_10 previously subtracted a HomeTeam/FTHG term
# (an exact duplicate of part of h_gd_10's own calculation),
# which has nothing to do with the away team's own record.
# It should be the away team's own goals-for minus goals-against
# in its own away matches.
# ============================================================
df['a_gd_10'] = (df.groupby(['Season', 'Div', 'AwayTeam'])['FTAG'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean())
                 - df.groupby(['Season', 'Div', 'AwayTeam'])['FTHG'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()))

df['gd_diff_10'] = df['h_gd_10'].fillna(0) - df['a_gd_10'].fillna(0)

df['h_shots'] = df.groupby(['Season', 'Div', 'HomeTeam'])['HS'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0) * df['passing_decay_coef']
df['a_shots'] = df.groupby(['Season', 'Div', 'AwayTeam'])['AS'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0) * df['passing_decay_coef']
df['shot_diff_5'] = df['h_shots'] - df['a_shots']

df['h_xg_5'] = df.groupby(['Season', 'Div', 'HomeTeam'])['h_xg'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
df['a_xg_5'] = df.groupby(['Season', 'Div', 'AwayTeam'])['a_xg'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
df['xg_diff_5'] = df['h_xg_5'] - df['a_xg_5']

df['h_pts_5'] = df.groupby(['Season', 'Div', 'HomeTeam'])['h_pts'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
df['a_pts_5'] = df.groupby(['Season', 'Div', 'AwayTeam'])['a_pts'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
df['points_diff_5'] = df['h_pts_5'] - df['a_pts_5']

df['log_p_H'] = np.log(df['p_market_H'] + 1e-6)
df['log_p_D'] = np.log(df['p_market_D'] + 1e-6)
df['log_p_A'] = np.log(df['p_market_A'] + 1e-6)


def compute_elo(data, k=20.0, home_adv=60.0, season_regress=0.25):
    elo_ratings = {}
    last_season = {}
    home_pre = np.empty(len(data))
    away_pre = np.empty(len(data))
    for i, row in enumerate(data.itertuples()):
        key_h = (row.Div, row.HomeTeam)
        key_a = (row.Div, row.AwayTeam)
        for key in (key_h, key_a):
            if key not in elo_ratings:
                elo_ratings[key] = 1500.0
                last_season[key] = row.Season
            elif last_season[key] != row.Season:
                elo_ratings[key] = (1 - season_regress) * elo_ratings[key] + season_regress * 1500.0
                last_season[key] = row.Season
        h_elo, a_elo = elo_ratings[key_h], elo_ratings[key_a]
        home_pre[i], away_pre[i] = h_elo, a_elo
        diff = h_elo + home_adv - a_elo
        exp_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        actual = 1.0 if row.FTR == 'H' else (0.5 if row.FTR == 'D' else 0.0)
        gd = abs(row.FTHG - row.FTAG)
        g_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8.0)
        delta = k * g_mult * (actual - exp_home)
        elo_ratings[key_h] = h_elo + delta
        elo_ratings[key_a] = a_elo - delta
    return home_pre, away_pre


df['home_elo_pre'], df['away_elo_pre'] = compute_elo(df)
df['elo_diff'] = df['home_elo_pre'] - df['away_elo_pre']
df['elo_exp_home'] = 1.0 / (1.0 + 10 ** (-(df['elo_diff'] + 60.0) / 400.0))

# ============================================================
# CHANGE #4: fixture congestion (rest days) and league-table
# motivation state (position, distance from mid-table). These are
# derived ENTIRELY from data already in the CSV (Date, FTR, FTHG,
# FTAG) -- no external sourcing needed, unlike weather. Both are
# well-documented effects in the football analytics literature
# (fatigue from congested fixtures; "six-pointer" motivation near
# the top/bottom of the table), and -- importantly -- unlike
# p_market_*/edge_*/elo_diff, these are NOT derived from betting
# odds, so they are a genuine test of whether non-market information
# has predictive value the market underweights.
#
# Leak safety: exactly like compute_elo, standings/rest state for a
# match are read BEFORE that match's result is folded in. A team's
# rank and rest days reflect only matches strictly before the
# current one, within the same (Div, Season).
# ============================================================
def compute_fixture_and_table_features(data):
    date_parsed = pd.to_datetime(data['Date'], errors='coerce')

    # total teams per (Div, Season), for normalizing position
    n_teams_lookup = (data.groupby(['Div', 'Season'])
                       .apply(lambda g: len(set(g['HomeTeam']) | set(g['AwayTeam'])))
                       .to_dict())

    standings = {}    # (Div, Season) -> {Team: {'pts': int, 'gd': int}}
    last_date = {}     # (Div, Team) -> last match Timestamp (home or away)

    n = len(data)
    home_rest = np.full(n, np.nan)
    away_rest = np.full(n, np.nan)
    home_pos_norm = np.empty(n)
    away_pos_norm = np.empty(n)
    home_boundary = np.empty(n)
    away_boundary = np.empty(n)

    for i, row in enumerate(data.itertuples()):
        key_ds = (row.Div, row.Season)
        table = standings.setdefault(key_ds, {})
        if row.HomeTeam not in table:
            table[row.HomeTeam] = {'pts': 0, 'gd': 0}
        if row.AwayTeam not in table:
            table[row.AwayTeam] = {'pts': 0, 'gd': 0}

        # pre-match rank (higher pts, then higher gd, is better)
        ranked = sorted(table.items(), key=lambda kv: (-kv[1]['pts'], -kv[1]['gd']))
        rank_of = {team: r + 1 for r, (team, _) in enumerate(ranked)}
        n_teams = n_teams_lookup.get(key_ds, len(table))

        h_rank = rank_of[row.HomeTeam]
        a_rank = rank_of[row.AwayTeam]
        home_pos_norm[i] = h_rank / n_teams
        away_pos_norm[i] = a_rank / n_teams
        mid = (n_teams + 1) / 2.0
        half = max(n_teams / 2.0, 1e-6)
        home_boundary[i] = abs(h_rank - mid) / half   # 0 = mid-table, ~1 = top or bottom of table
        away_boundary[i] = abs(a_rank - mid) / half

        cur_date = date_parsed.iloc[i]
        key_h = (row.Div, row.HomeTeam)
        key_a = (row.Div, row.AwayTeam)
        if key_h in last_date and pd.notna(cur_date):
            home_rest[i] = (cur_date - last_date[key_h]).days
        if key_a in last_date and pd.notna(cur_date):
            away_rest[i] = (cur_date - last_date[key_a]).days
        if pd.notna(cur_date):
            last_date[key_h] = cur_date
            last_date[key_a] = cur_date

        # update standings AFTER reading pre-match state
        pts_h = 3 if row.FTR == 'H' else (1 if row.FTR == 'D' else 0)
        pts_a = 3 if row.FTR == 'A' else (1 if row.FTR == 'D' else 0)
        table[row.HomeTeam]['pts'] += pts_h
        table[row.AwayTeam]['pts'] += pts_a
        gd_home = row.FTHG - row.FTAG
        table[row.HomeTeam]['gd'] += gd_home
        table[row.AwayTeam]['gd'] -= gd_home

    return home_rest, away_rest, home_pos_norm, away_pos_norm, home_boundary, away_boundary


(df['home_days_rest'], df['away_days_rest'],
 df['home_position_norm'], df['away_position_norm'],
 df['home_boundary_proximity'], df['away_boundary_proximity']) = compute_fixture_and_table_features(df)

# First appearance of a season for a team has no prior match -> no rest
# value. Fill with 7 (a typical weekly gap) as a neutral default, same
# pattern as the existing referee-feature fillna(25.0)/fillna(4.0) above.
df['home_days_rest'] = df['home_days_rest'].fillna(7.0).clip(upper=30)
df['away_days_rest'] = df['away_days_rest'].fillna(7.0).clip(upper=30)
df['rest_diff'] = df['home_days_rest'] - df['away_days_rest']
df['position_diff'] = df['away_position_norm'] - df['home_position_norm']  # >0 favors home (better/lower rank)
df['boundary_diff'] = df['home_boundary_proximity'] - df['away_boundary_proximity']

# ============================================================
# FIX #3: steam_H/D/A (ratio of closing odds to opening/avg odds)
# removed from stage1_features. Those are derived from PSCH/PSCD/PSCA,
# which are not known at bet-placement time -- including them as
# model inputs is look-ahead leakage. close_H/D/A are retained only
# for the post-hoc CLV diagnostic computed in create_bet_candidates.
# ============================================================
stage1_features = ['p_market_H', 'p_market_D', 'p_market_A',
                   'log_p_H', 'log_p_D', 'log_p_A',
                   'shot_diff_5', 'gd_diff_10', 'xg_diff_5', 'points_diff_5',
                   'edge_H', 'edge_D', 'edge_A',
                   'elo_diff', 'elo_exp_home',
                   'ref_fouls_mean', 'ref_cards_mean',
                   'high_wind_flag', 'heavy_precip_flag', 'extreme_temp_flag',
                   'passing_decay_coef', 'scoring_weather_multiplier',
                   'home_days_rest', 'away_days_rest', 'rest_diff',
                   'home_position_norm', 'away_position_norm', 'position_diff',
                   'home_boundary_proximity', 'away_boundary_proximity', 'boundary_diff'] + div_feature_cols

stage2_features = ['odds', 'p_model', 'p_market', 'prob_residual', 'ev', 'edge', 'shot_diff_5', 'gd_diff_10']


class PyTorchStage1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 3), nn.Softmax(dim=-1)
        )

    def forward(self, x): return self.net(x)


def train_nn_fold(X_tr, y_tr, X_val, epochs=80):
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-8
    X_tr_t = torch.tensor((X_tr - mean) / std, dtype=torch.float32)
    X_val_t = torch.tensor((X_val - mean) / std, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    model = PyTorchStage1(X_tr.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.NLLLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tr_t)
        loss = criterion(torch.log(preds + 1e-8), y_tr_t)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        preds_val = model(X_val_t).numpy()
    return model, mean, std, preds_val


def _normalize_probs(p): return p / p.sum(axis=1, keepdims=True)


def run_stage1_raw(train_data, apply_data):
    X1_tr = train_data[stage1_features].values.astype(np.float64)
    y_tr = train_data['target_cls'].values
    X1_ap = apply_data[stage1_features].values.astype(np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    p_tr_nn_oof = np.zeros((len(train_data), 3))
    p_tr_xgb_oof = np.zeros((len(train_data), 3))
    p_ap_nn_list, p_ap_xgb_list = [], []
    for tr_idx, val_idx in kf.split(X1_tr):
        X_tr_f, y_tr_f = X1_tr[tr_idx], y_tr[tr_idx]
        X_val_f = X1_tr[val_idx]
        nn_f, m_f, s_f, preds_val_nn = train_nn_fold(X_tr_f, y_tr_f, X_val_f)
        p_tr_nn_oof[val_idx] = preds_val_nn
        X_ap_t = torch.tensor((X1_ap - m_f) / s_f, dtype=torch.float32)
        nn_f.eval()
        with torch.no_grad():
            p_ap_nn_list.append(nn_f(X_ap_t).numpy())
        xgb_f = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
                                  num_class=3, n_jobs=1, tree_method='hist', random_state=42)
        xgb_f.fit(X_tr_f, y_tr_f)
        p_tr_xgb_oof[val_idx] = xgb_f.predict_proba(X_val_f)
        p_ap_xgb_list.append(xgb_f.predict_proba(X1_ap))
    p_tr_raw = _normalize_probs(0.5 * p_tr_nn_oof + 0.5 * p_tr_xgb_oof)
    p_ap_raw = _normalize_probs(0.5 * np.mean(p_ap_nn_list, axis=0) + 0.5 * np.mean(p_ap_xgb_list, axis=0))
    return p_tr_raw, p_ap_raw


def blend_with_market(dataset, p_raw, weight):
    p_mkt = dataset[['p_market_H', 'p_market_D', 'p_market_A']].values
    return _normalize_probs(weight * p_mkt + (1 - weight) * p_raw)


def select_market_weight(inner_val, p_val_raw, grid=MARKET_WEIGHT_GRID):
    y_val = inner_val['target_cls'].values
    best_w, best_ll = grid[0], np.inf
    for w in grid:
        p_blend = _normalize_probs(
            w * inner_val[['p_market_H', 'p_market_D', 'p_market_A']].values + (1 - w) * p_val_raw)
        ll = log_loss(y_val, p_blend, labels=[0, 1, 2])
        if ll < best_ll:
            best_ll = ll
            best_w = w
    return best_w


def run_stage1(train_data, apply_data, weight):
    p_tr_raw, p_ap_raw = run_stage1_raw(train_data, apply_data)
    p_tr_s1 = blend_with_market(train_data, p_tr_raw, weight)
    p_ap_s1 = blend_with_market(apply_data, p_ap_raw, weight)
    return p_tr_s1, p_ap_s1


def attach_stage1(dataset, p_s1, odds_prefix):
    dataset = dataset.copy()
    dataset['p_s1_H'], dataset['p_s1_D'], dataset['p_s1_A'] = p_s1[:, 0], p_s1[:, 1], p_s1[:, 2]
    dataset['ev_s1_H'] = dataset['p_s1_H'] * dataset[f'{odds_prefix}H'] - 1.0
    dataset['ev_s1_D'] = dataset['p_s1_D'] * dataset[f'{odds_prefix}D'] - 1.0
    dataset['ev_s1_A'] = dataset['p_s1_A'] * dataset[f'{odds_prefix}A'] - 1.0
    return dataset


def create_bet_candidates(d, odds_prefix):
    rows = []
    outcomes = [('H', f'{odds_prefix}H', 'p_s1_H', 'ev_s1_H', 'p_market_H', 'edge_H', 'close_H', 0),
                ('D', f'{odds_prefix}D', 'p_s1_D', 'ev_s1_D', 'p_market_D', 'edge_D', 'close_D', 1),
                ('A', f'{odds_prefix}A', 'p_s1_A', 'ev_s1_A', 'p_market_A', 'edge_A', 'close_A', 2)]
    for r in d.itertuples():
        for code, odds_col, p_col, ev_col, mkt_col, edge_col, close_col, target_val in outcomes:
            p_model = getattr(r, p_col)
            p_mkt = getattr(r, mkt_col)
            bet_odds = getattr(r, odds_col)
            close_odds = getattr(r, close_col)
            # FIX #2 (cont.): clv is NaN, not 0, when there is no genuine closing line
            if close_odds is not None and not np.isnan(close_odds) and close_odds > 0:
                clv = (bet_odds / close_odds) - 1.0
            else:
                clv = np.nan
            rows.append({
                'season': r.Season, 'date': r.Date, 'home_team': r.HomeTeam, 'away_team': r.AwayTeam,
                'bet_type': code, 'odds': bet_odds, 'close_odds': close_odds, 'clv': clv,
                'p_model': p_model, 'p_market': p_mkt,
                'prob_residual': p_model - p_mkt, 'ev': getattr(r, ev_col), 'edge': getattr(r, edge_col),
                'shot_diff_5': r.shot_diff_5, 'gd_diff_10': r.gd_diff_10,
                'target_win': int(r.target_cls == target_val)
            })
    return pd.DataFrame(rows)


def kelly_simulate(bets):
    bankroll = INITIAL_BANKROLL
    history = [bankroll]
    staked = 0.0
    for row in bets.itertuples():
        b = row.odds - 1.0
        q = 1.0 - row.stage2_p_win
        f_kelly = (b * row.stage2_p_win - q) / b if b > 0 else 0.0
        frac = min(f_kelly * 0.05, 0.0075) if f_kelly > 0 else 0.0
        stake = bankroll * frac
        if stake > 0:
            staked += stake
            pnl = stake * b if row.target_win == 1 else -stake
            bankroll += pnl
            history.append(bankroll)
    arr = np.array(history)
    peaks = np.maximum.accumulate(arr)
    dd = np.abs(((arr - peaks) / peaks).min()) * 100.0 if len(arr) else 0.0
    net = bankroll - INITIAL_BANKROLL
    roi = (net / staked * 100.0) if staked > 0 else 0.0
    return net, roi, dd


# ============================================================
# NEW: block bootstrap for a confidence interval on pooled ROI.
# Resampling is done AT THE FOLD (season) level, not at the
# individual-bet level. Bets within a season are correlated
# (they share the same trained model, the same market regime,
# and the same short trade sequence for Kelly path-dependence),
# so resampling individual bets would understate the true
# uncertainty. Resampling whole folds with replacement is the
# standard block-bootstrap fix for that within-fold dependence,
# and it also naturally handles the very small per-fold trade
# counts (21, 39, ...) here instead of pretending they're 60
# i.i.d. observations.
# ============================================================
def bootstrap_pooled_flat_roi(fold_bet_list, n_bootstrap=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n_folds = len(fold_bet_list)
    if n_folds == 0:
        return np.array([])
    roi_samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.integers(0, n_folds, size=n_folds)
        resampled = pd.concat([fold_bet_list[i] for i in chosen], ignore_index=True)
        n_b = len(resampled)
        if n_b == 0:
            roi_samples[b] = np.nan
            continue
        profit_b = (resampled['target_win'] * (resampled['odds'] - 1.0) - (1 - resampled['target_win'])).sum()
        roi_samples[b] = profit_b / n_b * 100.0
    return roi_samples


# ============================================================
# NEW: stage2 sensitivity sweep. Holds stage1 outputs, GLOBAL_*
# thresholds, and the odds pre-filter fixed; only varies the
# stage2 calibrated classifier's random_state and CV fold count.
# This isolates whether the pooled ROI sign/magnitude is a
# robust property of the pipeline, or an artifact of one
# particular stage2 fit -- exactly the ambiguity raised by the
# sign flip after including draws in stage2 training data.
# ============================================================
def run_stage2_sweep_config(fold_candidate_cache, seed, cv):
    fold_bets_list = []
    for fc in fold_candidate_cache:
        cand_tr_filter = fc['cand_tr_filter']
        cand_te_filter = fc['cand_te_filter'].copy()
        X2_tr = cand_tr_filter[stage2_features].values.astype(np.float64)
        y2_tr = cand_tr_filter['target_win'].values
        base_s2 = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=seed)
        cal_s2 = CalibratedClassifierCV(estimator=base_s2, method='sigmoid', cv=cv)
        cal_s2.fit(X2_tr, y2_tr)

        if len(cand_te_filter) > 0:
            cand_te_filter['stage2_p_win'] = \
                cal_s2.predict_proba(cand_te_filter[stage2_features].values.astype(np.float64))[:, 1]
            cand_te_filter['stage2_ev'] = cand_te_filter['stage2_p_win'] * cand_te_filter['odds'] - 1.0
        else:
            cand_te_filter['stage2_p_win'] = np.array([])
            cand_te_filter['stage2_ev'] = np.array([])

        fold_bets = cand_te_filter[
            (cand_te_filter['odds'] >= GLOBAL_LO_ODDS) & (cand_te_filter['odds'] <= GLOBAL_HI_ODDS) &
            (cand_te_filter['stage2_p_win'] >= GLOBAL_MIN_PWIN) & (cand_te_filter['stage2_ev'] >= GLOBAL_MIN_EV)
            ]
        if len(fold_bets) > 0:
            fold_bets_list.append(fold_bets)

    if not fold_bets_list:
        return 0, np.nan
    pooled = pd.concat(fold_bets_list, ignore_index=True)
    n = len(pooled)
    flat_profit = (pooled['target_win'] * (pooled['odds'] - 1.0) - (1 - pooled['target_win'])).sum()
    flat_roi = flat_profit / n * 100.0
    return n, flat_roi


def run_stage2_sensitivity_sweep(fold_candidate_cache):
    seeds = [0, 1, 7, 21, 42, 99, 123, 2024]
    cv_options = [3, 5]
    results = []
    for cv in cv_options:
        for seed in seeds:
            n, roi = run_stage2_sweep_config(fold_candidate_cache, seed, cv)
            results.append({'seed': seed, 'cv': cv, 'n_trades': n, 'flat_roi': roi})
    return pd.DataFrame(results)


# CHANGE #2: stake into best-available price (BestH/D/A) instead of the
# market-average price (AvgH/D/A). AvgH/D/A represents a price no single
# bookmaker actually offers -- it is the mean across books, useful for
# de-vigging (kept as-is for p_market_H/D/A above) but not something a
# real bettor can transact at. BestH/D/A (already computed above, with
# fallback to AvgH/D/A where a best price isn't available) is the price
# an actual bettor shopping across books could get. This flows through
# every downstream step automatically since odds_prefix already
# parametrizes ev_s1_*, create_bet_candidates' 'odds' column, the
# GLOBAL_LO_ODDS/HI_ODDS filter, CLV, and the stage2 'odds' feature --
# no other logic changes.
odds_prefix = 'Best'
all_fold_bets = []
# NEW: cache each fold's post-stage1 candidate pools so the stage2
# sensitivity sweep below can retrain ONLY stage2 (fast: a single
# XGBClassifier + CalibratedClassifierCV fit) without repeating the
# expensive stage1 NN+XGB ensemble training for every seed/cv combo.
fold_candidate_cache = []

# ============================================================
# CHANGE #5: seed-ensembled Stage 2 (variance reduction).
# The sensitivity sweep above demonstrated that a SINGLE stage2 fit's
# random_state/cv choice can flip the sign of pooled ROI. Rather than
# just diagnosing that instability, this directly addresses it: train
# the calibrated Stage 2 classifier across the same 16 (seed, cv)
# configurations used in the sweep, and AVERAGE their predicted
# probabilities before any bet-selection threshold is applied. This is
# a standard variance-reduction technique (the same principle behind
# random forests averaging many trees) -- it does not manufacture an
# edge that isn't there, but if a real, small signal is being masked
# by single-fit noise, averaging is the legitimate way to recover it.
# GLOBAL_* thresholds are applied AFTER averaging, exactly as before.
# ============================================================
def train_stage2_ensemble(cand_tr_filter, cand_te_filter,
                          seeds=tuple(range(100)), cv_options=(3, 5)):
    # CHANGE #8: ensemble size increased from 8 to 100 seeds, per advisor
    # suggestion. This further reduces SEED-noise variance (averaging over
    # more independent random draws makes the average more representative
    # of "the space of reasonable seeds"). It does NOT address the
    # separate feature-set sensitivity found earlier (travel distance
    # flipping the sign) -- that is a different axis of instability and
    # is not fixed by averaging over more seeds.
    X2_tr = cand_tr_filter[stage2_features].values.astype(np.float64)
    y2_tr = cand_tr_filter['target_win'].values
    if len(cand_te_filter) == 0:
        return np.array([])
    X2_te = cand_te_filter[stage2_features].values.astype(np.float64)
    all_probs = []
    total_models = len(seeds) * len(cv_options)
    fitted = 0
    for cv in cv_options:
        for seed in seeds:
            base_s2 = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03,
                                        subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=seed)
            cal_s2 = CalibratedClassifierCV(estimator=base_s2, method='sigmoid', cv=cv)
            cal_s2.fit(X2_tr, y2_tr)
            all_probs.append(cal_s2.predict_proba(X2_te)[:, 1])
            fitted += 1
            if fitted % 25 == 0 or fitted == total_models:
                print(f"    ensemble progress: {fitted}/{total_models} models fitted")
    return np.mean(np.vstack(all_probs), axis=0)


for i in range(1, len(ALL_SEASONS)):
    test_season = ALL_SEASONS[i]
    full_train_seasons = ALL_SEASONS[:i]

    # ROLLING WINDOW ENFORCEMENT
    train_seasons = full_train_seasons[-ROLLING_WINDOW_SIZE:]

    train_data = df[df['Season'].isin(train_seasons)].copy()
    test_data = df[df['Season'] == test_season].copy()
    if len(train_data) < 200 or len(test_data) == 0: continue

    if len(train_seasons) > 1:
        inner_val_season = train_seasons[-1]
        inner_train_seasons = train_seasons[:-1]
        inner_train = train_data[train_data['Season'].isin(inner_train_seasons)].copy()
        inner_val = train_data[train_data['Season'] == inner_val_season].copy()
    else:
        inner_train = train_data
        inner_val = pd.DataFrame()

    if len(inner_train) >= 150 and len(inner_val) >= 50:
        p_inner_tr_raw, p_inner_val_raw = run_stage1_raw(inner_train, inner_val)
        best_weight = select_market_weight(inner_val, p_inner_val_raw)
    else:
        best_weight = MARKET_WEIGHT_DEFAULT

    p_tr_s1, p_te_s1 = run_stage1(train_data, test_data, best_weight)

    # DIAGNOSTIC: stage1 calibration quality (log-loss) vs actual outcomes,
    # independent of the EV/odds filter. Also computed for the raw market
    # probabilities alone, so you can see whether the market itself is
    # getting harder to beat (market log-loss flat/improving) vs the model
    # specifically degrading (stage1 log-loss worsening faster than market's).
    stage1_ll = log_loss(test_data['target_cls'], p_te_s1, labels=[0, 1, 2])
    market_probs_test = test_data[['p_market_H', 'p_market_D', 'p_market_A']].values
    market_ll = log_loss(test_data['target_cls'], market_probs_test, labels=[0, 1, 2])
    print(f"[Fold test={test_season}] Stage1 log-loss: {stage1_ll:.4f} | "
          f"Market-only log-loss: {market_ll:.4f} | "
          f"Model edge (market - stage1): {market_ll - stage1_ll:+.4f}")

    train_s1 = attach_stage1(train_data, p_tr_s1, odds_prefix)
    test_s1 = attach_stage1(test_data, p_te_s1, odds_prefix)
    cand_tr = create_bet_candidates(train_s1, odds_prefix)
    cand_te = create_bet_candidates(test_s1, odds_prefix)

    # CHANGE #1: draws are no longer excluded from the candidate pool.
    # The original code dropped bet_type == 'D' with no stated rationale,
    # discarding roughly a third of all candidates. Draws go through the
    # exact same odds pre-filter, stage2 calibration, and GLOBAL_* selection
    # thresholds as home/away bets -- no special-casing, so this is a pure
    # coverage change, not a new rule tuned to the test folds.
    cand_tr_filter = cand_tr[
        (cand_tr['odds'] >= 1.40) & (cand_tr['odds'] <= 2.20)].copy()
    cand_te_filter = cand_te[
        (cand_te['odds'] >= 1.40) & (cand_te['odds'] <= 2.20)].copy()

    # NEW: cache this fold's candidate pools (pre-stage2) for the
    # sensitivity sweep at the end of the script.
    fold_candidate_cache.append({
        'test_season': test_season,
        'cand_tr_filter': cand_tr_filter.copy(),
        'cand_te_filter': cand_te_filter.copy(),
    })

    if len(cand_te_filter) > 0:
        cand_te_filter['stage2_p_win'] = train_stage2_ensemble(cand_tr_filter, cand_te_filter)
        cand_te_filter['stage2_ev'] = cand_te_filter['stage2_p_win'] * cand_te_filter['odds'] - 1.0
    else:
        cand_te_filter['stage2_p_win'] = np.array([])
        cand_te_filter['stage2_ev'] = np.array([])

    # DIAGNOSTIC: inspect the distribution of stage2 outputs for this fold,
    # so a fold that produces zero qualifying bets can be explained rather
    # than silently skipped. Compares against the GLOBAL_MIN_PWIN/GLOBAL_MIN_EV
    # thresholds actually used to select bets below.
    print(f"\n--- Diagnostic: test={test_season} stage2 output distribution "
          f"(n_candidates={len(cand_te_filter)}) ---")
    if len(cand_te_filter) > 0:
        print(cand_te_filter[['stage2_p_win', 'stage2_ev']].describe())
        n_pass_pwin = (cand_te_filter['stage2_p_win'] >= GLOBAL_MIN_PWIN).sum()
        n_pass_ev = (cand_te_filter['stage2_ev'] >= GLOBAL_MIN_EV).sum()
        n_pass_odds = ((cand_te_filter['odds'] >= GLOBAL_LO_ODDS) & (cand_te_filter['odds'] <= GLOBAL_HI_ODDS)).sum()
        print(f"Candidates passing p_win>={GLOBAL_MIN_PWIN}: {n_pass_pwin}/{len(cand_te_filter)}")
        print(f"Candidates passing ev>={GLOBAL_MIN_EV}: {n_pass_ev}/{len(cand_te_filter)}")
        print(f"Candidates passing odds in [{GLOBAL_LO_ODDS},{GLOBAL_HI_ODDS}]: {n_pass_odds}/{len(cand_te_filter)}")
    else:
        print("No candidates survived the odds pre-filter (1.40-2.20, non-draw) for this fold.")
    print("---\n")

    fold_bets = cand_te_filter[
        (cand_te_filter['odds'] >= GLOBAL_LO_ODDS) & (cand_te_filter['odds'] <= GLOBAL_HI_ODDS) &
        (cand_te_filter['stage2_p_win'] >= GLOBAL_MIN_PWIN) & (cand_te_filter['stage2_ev'] >= GLOBAL_MIN_EV)
        ].sort_values(['season', 'date']).reset_index(drop=True)

    n = len(fold_bets)
    if n > 0:
        wins = fold_bets['target_win'].sum()
        wr = wins / n * 100
        bt_counts = fold_bets['bet_type'].value_counts().to_dict()
        print(f"[Fold test={test_season}] Bet type breakdown: {bt_counts}")
        flat_profit = (fold_bets['target_win'] * (fold_bets['odds'] - 1.0) - (1 - fold_bets['target_win'])).sum()
        flat_roi = flat_profit / n * 100
        net_k, roi_k, dd_k = kelly_simulate(fold_bets)
        clv_valid = fold_bets['clv'].dropna()
        mean_clv = clv_valid.mean() * 100 if len(clv_valid) > 0 else float('nan')
        print(
            f"[Fold test={test_season}] TrainSeasons={train_seasons} Trades={n} WinRate={wr:.1f}% "
            f"FlatROI={flat_roi:+.2f}% KellyROI={roi_k:+.2f}% MaxDD={dd_k:.2f}% "
            f"MeanCLV={mean_clv:+.2f}% (n_with_close={len(clv_valid)}/{n})")
        all_fold_bets.append(fold_bets)

print("=" * 80)
PRODUCTION_N = 0
PRODUCTION_FLAT_ROI = np.nan
if all_fold_bets:
    pooled = pd.concat(all_fold_bets, ignore_index=True)
    n = len(pooled)
    flat_profit = (pooled['target_win'] * (pooled['odds'] - 1.0) - (1 - pooled['target_win'])).sum()
    flat_roi = flat_profit / n * 100
    net_k, roi_k, dd_k = kelly_simulate(pooled)
    clv_valid = pooled['clv'].dropna()
    mean_clv = clv_valid.mean() * 100 if len(clv_valid) > 0 else float('nan')
    print(
        f"POOLED ROLLING WINDOW OOS TRADES (PRODUCTION -- 100-seed x cv[3,5] = 200-model ensemble): "
        f"{n} | Flat ROI: {flat_roi:+.2f}% | Kelly ROI: {roi_k:+.2f}% | "
        f"MaxDD: {dd_k:.2f}% | Mean CLV: {mean_clv:+.2f}% (n_with_close={len(clv_valid)}/{n})")
    # Captured under distinct names so later blocks (which reuse n/flat_roi
    # as loop-local variable names) can't accidentally overwrite these.
    PRODUCTION_N = n
    PRODUCTION_FLAT_ROI = flat_roi

    # ============================================================
    # NEW: fold-level block bootstrap CI on pooled flat ROI, and a
    # one-sided empirical p-value for "true ROI <= 0". This is the
    # test that actually answers "is this distinguishable from
    # noise/zero", which point-estimate ROI alone cannot answer,
    # especially with only 2-3 folds and 20-40 bets/fold.
    # ============================================================
    if len(all_fold_bets) >= 2:
        roi_boot = bootstrap_pooled_flat_roi(all_fold_bets, n_bootstrap=N_BOOTSTRAP)
        roi_boot = roi_boot[~np.isnan(roi_boot)]
        ci_lo, ci_hi = np.percentile(roi_boot, [2.5, 97.5])
        p_le_zero = np.mean(roi_boot <= 0.0)
        print(f"\nFold-level block bootstrap (n_bootstrap={N_BOOTSTRAP}, resampling whole folds):")
        print(f"  Pooled Flat ROI 95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")
        print(f"  P(true ROI <= 0) from bootstrap distribution: {p_le_zero:.3f}")
        if ci_lo <= 0.0 <= ci_hi:
            print("  --> CI includes 0: pooled ROI is NOT statistically distinguishable from zero.")
        else:
            print("  --> CI excludes 0.")
    else:
        print("\n(Fewer than 2 qualifying folds -- bootstrap CI skipped, not meaningful with 1 fold.)")

    # Per-fold breakdown so the pooled number is never read in isolation
    print("\nPer-season breakdown (do not report POOLED alone):")
    for season, grp in pooled.groupby('season'):
        n_s = len(grp)
        wr_s = grp['target_win'].mean() * 100
        flat_s = ((grp['target_win'] * (grp['odds'] - 1.0) - (1 - grp['target_win'])).sum()) / n_s * 100
        print(f"  Season {season}: n={n_s} WinRate={wr_s:.1f}% FlatROI={flat_s:+.2f}%")

    # ============================================================
    # CHANGE #11: bet-level "confidence score" diagnostics for the placed
    # bets of the production ensemble. Read-only: nothing here feeds back
    # into model training or bet selection.
    #  (a) saves every placed bet with its stage2_p_win (the confidence score)
    #  (b) compares stated confidence with realized win rate (calibration)
    #  (c) compares the model's own expected ROI with realized ROI
    #  (d) bootstrap CI that resamples INDIVIDUAL BETS (the season-level
    #      bootstrap above only reshuffles 3 seasons, so it looks tighter)
    # ============================================================
    print("\n" + "=" * 80)
    print("BET-LEVEL CONFIDENCE DIAGNOSTICS (production ensemble, placed bets only)")
    print("=" * 80)
    profit = pooled['target_win'] * (pooled['odds'] - 1.0) - (1 - pooled['target_win'])
    pooled = pooled.assign(_profit=profit)
    keep = [c for c in ['season', 'date', 'home_team', 'away_team', 'bet_type', 'odds',
                        'stage2_p_win', 'stage2_ev', 'target_win', 'clv'] if c in pooled.columns]
    pooled[keep].to_csv("placed_bets_production.csv", index=False)
    print(f"Saved {len(pooled)} placed bets to placed_bets_production.csv")

    print(f"\n(b) Stated confidence vs realized win rate")
    print(f"  Mean stated confidence (stage2_p_win): {pooled['stage2_p_win'].mean()*100:.1f}%")
    print(f"  Realized win rate:                     {pooled['target_win'].mean()*100:.1f}%")
    print(f"  Gap (realized - stated):               {(pooled['target_win'].mean()-pooled['stage2_p_win'].mean())*100:+.1f} pts")
    print("  By confidence bin:")
    bins = [0.58, 0.62, 0.66, 0.70, 1.01]
    labels = ["0.58-0.62", "0.62-0.66", "0.66-0.70", "0.70+"]
    pooled['_bin'] = pd.cut(pooled['stage2_p_win'], bins=bins, labels=labels, right=False)
    for lab in labels:
        g = pooled[pooled['_bin'] == lab]
        if len(g) == 0:
            print(f"    {lab}: n=0")
            continue
        print(f"    {lab}: n={len(g):3d}  stated={g['stage2_p_win'].mean()*100:5.1f}%  "
              f"realized={g['target_win'].mean()*100:5.1f}%  avg_odds={g['odds'].mean():.2f}  "
              f"FlatROI={g['_profit'].mean()*100:+.1f}%")
    print("  (bins with small n are noisy -- read the overall gap first)")

    print(f"\n(c) Model's own expectation vs reality")
    print(f"  Mean stage2_ev (model's expected ROI per bet): {pooled['stage2_ev'].mean()*100:+.2f}%")
    print(f"  Realized flat ROI:                             {pooled['_profit'].mean()*100:+.2f}%")

    print(f"\n(d) Bet-level bootstrap on pooled Flat ROI ({N_BOOTSTRAP} resamples of individual bets)")
    rng_b = np.random.default_rng(42)
    prof = pooled['_profit'].values
    boots = np.array([prof[rng_b.integers(0, len(prof), len(prof))].mean() * 100 for _ in range(N_BOOTSTRAP)])
    lo_b, hi_b = np.percentile(boots, [2.5, 97.5])
    se_b = prof.std(ddof=1) / np.sqrt(len(prof)) * 100
    print(f"  Flat ROI point estimate: {prof.mean()*100:+.2f}%   (n={len(prof)})")
    print(f"  Bootstrap 95% CI:        [{lo_b:+.2f}%, {hi_b:+.2f}%]")
    print(f"  Standard error:          {se_b:.2f} pts")
    print(f"  P(true ROI <= 0):        {(boots <= 0).mean():.3f}")
    print("  Caveat: bets in the same season are correlated, so even this interval is optimistic;")
    print("  it is still much wider than the 3-season bootstrap because it uses every bet.")
print("=" * 80)

# ============================================================
# NEW: stage2 sensitivity sweep report. Stage1 (NN+XGB ensemble,
# market weight selection) and all GLOBAL_* thresholds are fixed
# exactly as in the main run above -- only the stage2 calibrated
# classifier's random_state and CV fold count vary, using the
# cached post-stage1 candidate pools. If pooled ROI stays positive
# and roughly stable across this sweep, that is real evidence the
# main result is not an artifact of one lucky stage2 fit. If it
# swings in sign or magnitude, that instability is itself a result
# worth reporting, not something to suppress.
# ============================================================
print("\n" + "=" * 80)
print("STAGE2 SENSITIVITY SWEEP (DIAGNOSTIC ONLY -- single-seed models, NOT the ensemble used above)")
print("=" * 80)
sweep_df = run_stage2_sensitivity_sweep(fold_candidate_cache)
print(sweep_df.to_string(index=False))

valid_roi = sweep_df['flat_roi'].dropna()
if len(valid_roi) > 0:
    n_positive = (valid_roi > 0).sum()
    n_negative = (valid_roi < 0).sum()
    print(f"\nAcross {len(sweep_df)} (seed, cv) configs:")
    print(f"  Flat ROI: mean={valid_roi.mean():+.2f}% std={valid_roi.std():.2f}% "
          f"min={valid_roi.min():+.2f}% max={valid_roi.max():+.2f}%")
    print(f"  Positive-ROI configs: {n_positive}/{len(valid_roi)} | "
          f"Negative-ROI configs: {n_negative}/{len(valid_roi)}")
    print(f"  Trade count range across configs: {sweep_df['n_trades'].min()}-{sweep_df['n_trades'].max()}")
    if n_negative == 0:
        print("  --> Sign is stable positive across all stage2 refits tested.")
    elif n_positive == 0:
        print("  --> Sign is stable negative across all stage2 refits tested.")
    else:
        print("  --> SIGN IS NOT STABLE: pooled ROI flips between positive and negative "
              "depending on stage2 random_state/cv alone. Report this as a limitation.")
    print(f"\n  Comparison: the PRODUCTION ensemble above (which averages exactly these 16 "
          f"models' probabilities before selecting bets) reports its own pooled Flat ROI "
          f"further up this output -- compare that single number to this sweep's "
          f"mean/std/range to see whether averaging actually reduced the instability, "
          f"or merely relocated it.")
else:
    print("No config in the sweep produced any qualifying trades.")
print("=" * 80)


# ============================================================
# CHANGE #6: meta-level check on the ensemble itself.
# The 16-model ensemble reduced SINGLE-SEED noise, but the 16 seeds
# (0,1,7,21,42,99,123,2024) x cv(3,5) that make it up are themselves
# one arbitrary choice. This checks whether the ensemble's positive
# result survives being built from DIFFERENT arbitrary seed sets --
# a different set of 8 seeds, a smaller ensemble, a larger combined
# ensemble -- using the exact same GLOBAL_* thresholds and the same
# cached post-stage1 candidate pools throughout. No thresholds are
# retuned; only the composition of the Stage 2 ensemble varies.
# ============================================================
def run_ensemble_composition(fold_candidate_cache, seeds, cv_options):
    fold_bets_list = []
    for fc in fold_candidate_cache:
        cand_tr_filter = fc['cand_tr_filter']
        cand_te_filter = fc['cand_te_filter'].copy()
        if len(cand_te_filter) == 0:
            continue
        cand_te_filter['stage2_p_win'] = train_stage2_ensemble(
            cand_tr_filter, cand_te_filter, seeds=seeds, cv_options=cv_options)
        cand_te_filter['stage2_ev'] = cand_te_filter['stage2_p_win'] * cand_te_filter['odds'] - 1.0
        fold_bets = cand_te_filter[
            (cand_te_filter['odds'] >= GLOBAL_LO_ODDS) & (cand_te_filter['odds'] <= GLOBAL_HI_ODDS) &
            (cand_te_filter['stage2_p_win'] >= GLOBAL_MIN_PWIN) & (cand_te_filter['stage2_ev'] >= GLOBAL_MIN_EV)
            ]
        if len(fold_bets) > 0:
            fold_bets_list.append(fold_bets)
    if not fold_bets_list:
        return 0, np.nan
    pooled = pd.concat(fold_bets_list, ignore_index=True)
    n = len(pooled)
    flat_profit = (pooled['target_win'] * (pooled['odds'] - 1.0) - (1 - pooled['target_win'])).sum()
    flat_roi = flat_profit / n * 100.0
    return n, flat_roi


SEEDS_A = (0, 1, 7, 21, 42, 99, 123, 2024)          # the seeds used in production above
SEEDS_B = (5, 13, 29, 55, 77, 111, 256, 999)        # a completely different arbitrary set
CV_BOTH = (3, 5)

compositions = [
    ("A: production (8 seeds x cv[3,5] = 16 models)", SEEDS_A, CV_BOTH),
    ("B: alternate seed set (8 seeds x cv[3,5] = 16 models)", SEEDS_B, CV_BOTH),
    ("A, cv=3 only (8 models)", SEEDS_A, (3,)),
    ("A, cv=5 only (8 models)", SEEDS_A, (5,)),
    ("A+B combined (16 seeds x cv[3,5] = 32 models)", SEEDS_A + SEEDS_B, CV_BOTH),
    # CHANGE #9: the proper LARGE-SCALE version of the composition test.
    # PRODUCTION above already IS one 200-model block (seeds 0-99 x
    # cv[3,5]) -- not recomputed here to avoid duplicating ~600 model
    # fits. This adds one INDEPENDENT second block of the same size
    # (seeds 1000-1099, non-overlapping with production's 0-99) so the
    # two can be compared directly: does averaging over 200 seeds give
    # the same answer regardless of WHICH 200 seeds, or does meaningful
    # composition-sensitivity persist even at this larger scale?
    ("D: independent large block (seeds 1000-1099 x cv[3,5] = 200 models)",
     tuple(range(1000, 1100)), CV_BOTH),
]

print("\n" + "=" * 80)
print("META-CHECK: does the ensemble's result survive DIFFERENT arbitrary seed sets?")
print("(same GLOBAL_* thresholds, same cached candidates -- only ensemble composition varies)")
print("=" * 80)
meta_results = []
for label, seeds, cv_opts in compositions:
    n, roi = run_ensemble_composition(fold_candidate_cache, seeds, cv_opts)
    meta_results.append({'composition': label, 'n_trades': n, 'flat_roi': roi})
    print(f"  {label:55s} n={n:4d}  FlatROI={roi:+.2f}%" if not np.isnan(roi)
          else f"  {label:55s} n={n:4d}  FlatROI=nan")

meta_df = pd.DataFrame(meta_results)
valid_meta = meta_df['flat_roi'].dropna()
if len(valid_meta) > 0:
    n_pos = (valid_meta > 0).sum()
    n_neg = (valid_meta < 0).sum()
    print(f"\nAcross {len(meta_df)} ensemble compositions:")
    print(f"  Flat ROI: mean={valid_meta.mean():+.2f}% std={valid_meta.std():.2f}% "
          f"min={valid_meta.min():+.2f}% max={valid_meta.max():+.2f}%")
    print(f"  Positive: {n_pos}/{len(valid_meta)} | Negative: {n_neg}/{len(valid_meta)}")
    if n_neg == 0:
        print("  --> Result is POSITIVE across every ensemble composition tested. "
              "This is real evidence the earlier positive result is not just an artifact "
              "of which 16 seeds happened to be averaged.")
    elif n_pos == 0:
        print("  --> Result is NEGATIVE across every ensemble composition tested.")
    else:
        print("  --> STILL NOT STABLE: even after averaging within each ensemble, the sign "
              "changes depending on WHICH arbitrary seed set was used to build the ensemble. "
              "This means the earlier positive production result was likely still an artifact "
              "of that particular set of 16 seeds, not a recovered real signal.")
print("=" * 80)

# ============================================================
# Explicit large-scale comparison: PRODUCTION (seeds 0-99) vs the new
# independent 200-model block D (seeds 1000-1099). This is the properly
# scaled-up version of the composition test -- both blocks are as large
# as the actual production ensemble, not just 8-or-16-model toy versions.
# ============================================================
print("\n" + "=" * 80)
print("LARGE-SCALE COMPARISON: production (seeds 0-99) vs. independent block D (seeds 1000-1099)")
print("=" * 80)
d_row = meta_df[meta_df['composition'].str.startswith('D:')]
if len(d_row) > 0 and PRODUCTION_N > 0 and not np.isnan(PRODUCTION_FLAT_ROI):
    d_roi = d_row.iloc[0]['flat_roi']
    d_n = d_row.iloc[0]['n_trades']
    print(f"  Production (seeds 0-99):      n={PRODUCTION_N:4d}  FlatROI={PRODUCTION_FLAT_ROI:+.2f}%")
    print(f"  Block D    (seeds 1000-1099): n={d_n:4d}  FlatROI={d_roi:+.2f}%")
    same_sign = (PRODUCTION_FLAT_ROI > 0) == (d_roi > 0)
    diff = abs(PRODUCTION_FLAT_ROI - d_roi)
    if same_sign:
        print(f"  --> SAME SIGN across two independent 200-model blocks (difference: {diff:.2f} pts). "
              f"This is real evidence the sign, at least, is not sensitive to which large block of "
              f"seeds was used -- though the magnitude still varies by {diff:.2f} percentage points.")
    else:
        print(f"  --> SIGN FLIPS even between two independent 200-model blocks (production "
              f"{PRODUCTION_FLAT_ROI:+.2f}% vs. block D {d_roi:+.2f}%). This means composition "
              f"sensitivity persists even at large ensemble scale -- averaging over more seeds "
              f"reduces but does not eliminate this source of instability.")
else:
    print("  Could not compare -- production or block D produced no qualifying trades.")
print("=" * 80)
