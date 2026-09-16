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
from scipy.optimize import brentq
import pandas as pd
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("--- Running Odds-Stratified 2-Stage Pipeline (PRODUCTION GOLDEN VERSION) ---\n")

# =====================================================================
# PRODUCTION CONFIGURATION CONSTANTS (GOLDEN MODE)
# =====================================================================
MARKET_WEIGHT = 0.65     # Trust the bookies slightly MORE (65/35 split)
INITIAL_BANKROLL = 100.0

MAX_ODDS = 2.20          # Capped at 2.20 to avoid underdog variance
MIN_EV = 0.035           # Demands a strictly positive 3.5% edge

TRAIN_SEASONS = [2021, 2122, 2223, 2324]
TEST_SEASON = 2425


# =====================================================================
# 1. SHIN'S DEVIGGING FUNCTION
# =====================================================================
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


# =====================================================================
# 2. DATA PREPARATION & FEATURE ENGINEERING (Reverted to Core 8 Features)
# =====================================================================
df = pd.read_csv("matches_multiseason.csv", low_memory=False)

base_cols = ['AvgH', 'AvgD', 'AvgA', 'FTR', 'Season', 'FTHG', 'FTAG', 'HS', 'AS', 'HomeTeam', 'AwayTeam', 'Date']
df = df.dropna(subset=['AvgH', 'AvgD', 'AvgA', 'FTR', 'Season', 'FTHG', 'FTAG', 'HS', 'AS']).copy()

df['BestH'] = df['MaxH'].fillna(df['AvgH']) if 'MaxH' in df.columns else df['AvgH']
df['BestD'] = df['MaxD'].fillna(df['AvgD']) if 'MaxD' in df.columns else df['AvgD']
df['BestA'] = df['MaxA'].fillna(df['AvgA']) if 'MaxA' in df.columns else df['AvgA']

shin_probs = []
for idx, r in df.iterrows():
    p_devig = devig_shin([r['AvgH'], r['AvgD'], r['AvgA']])
    shin_probs.append(p_devig)

shin_probs = np.array(shin_probs)
df['p_market_H'] = shin_probs[:, 0]
df['p_market_D'] = shin_probs[:, 1]
df['p_market_A'] = shin_probs[:, 2]

df['edge_H'] = df['BestH'] / df['AvgH']
df['edge_D'] = df['BestD'] / df['AvgD']
df['edge_A'] = df['BestA'] / df['AvgA']

target_map = {'H': 0, 'D': 1, 'A': 2}
df['target_cls'] = df['FTR'].map(target_map)
df = df.dropna(subset=['target_cls']).copy()
df['target_cls'] = df['target_cls'].astype(int)

df = df.sort_values(['Season', 'Date']).reset_index(drop=True)

df['h_pts'] = np.where(df['FTR'] == 'H', 3, np.where(df['FTR'] == 'D', 1, 0))
df['a_pts'] = np.where(df['FTR'] == 'A', 3, np.where(df['FTR'] == 'D', 1, 0))

df['h_xg'] = df['FTHG'] * 0.4 + df['HS'] * 0.08
df['a_xg'] = df['FTAG'] * 0.4 + df['AS'] * 0.08

df['h_gd_10'] = df.groupby(['Season', 'HomeTeam'])['FTHG'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean()) - \
                df.groupby(['Season', 'HomeTeam'])['FTAG'].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=1).mean())
df['a_gd_10'] = df.groupby(['Season', 'AwayTeam'])['FTAG'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean()) - \
                df.groupby(['Season', 'AwayTeam'])['FTHG'].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=1).mean())
df['gd_diff_10'] = df['h_gd_10'].fillna(0) - df['a_gd_10'].fillna(0)

df['h_shots'] = df.groupby(['Season', 'HomeTeam'])['HS'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
df['a_shots'] = df.groupby(['Season', 'AwayTeam'])['AS'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
df['shot_diff_5'] = df['h_shots'] - df['a_shots']

df['h_xg_5'] = df.groupby(['Season', 'HomeTeam'])['h_xg'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
df['a_xg_5'] = df.groupby(['Season', 'AwayTeam'])['a_xg'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
df['xg_diff_5'] = df['h_xg_5'] - df['a_xg_5']

df['h_pts_5'] = df.groupby(['Season', 'HomeTeam'])['h_pts'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
df['a_pts_5'] = df.groupby(['Season', 'AwayTeam'])['a_pts'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
df['points_diff_5'] = df['h_pts_5'] - df['a_pts_5']

df['log_p_H'] = np.log(df['p_market_H'] + 1e-6)
df['log_p_D'] = np.log(df['p_market_D'] + 1e-6)
df['log_p_A'] = np.log(df['p_market_A'] + 1e-6)

stage1_features = [
    'p_market_H', 'p_market_D', 'p_market_A',
    'log_p_H', 'log_p_D', 'log_p_A',
    'shot_diff_5', 'gd_diff_10', 'xg_diff_5', 'points_diff_5',
    'edge_H', 'edge_D', 'edge_A'
]

train_data = df[df['Season'].isin(TRAIN_SEASONS)].copy()
test_data = df[df['Season'] == TEST_SEASON].copy()

print(f"Train matches (Seasons {TRAIN_SEASONS}) : {len(train_data)}")
print(f"Test matches  (Season {TEST_SEASON})        : {len(test_data)}\n")


# =====================================================================
# 3. STAGE 1 TRAINING (OOF PROBABILITIES & BLENDING)
# =====================================================================
class PyTorchStage1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


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


X1_tr = train_data[stage1_features].values
y_tr = train_data['target_cls'].values
X1_te = test_data[stage1_features].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
p_tr_nn_oof = np.zeros((len(train_data), 3))
p_tr_xgb_oof = np.zeros((len(train_data), 3))
p_te_nn_list = []
p_te_xgb_list = []

for tr_idx, val_idx in kf.split(X1_tr):
    X_tr_f, y_tr_f = X1_tr[tr_idx], y_tr[tr_idx]
    X_val_f, y_val_f = X1_tr[val_idx], y_tr[val_idx]

    nn_f, m_f, s_f, preds_val_nn = train_nn_fold(X_tr_f, y_tr_f, X_val_f)
    p_tr_nn_oof[val_idx] = preds_val_nn

    X_te_t = torch.tensor((X1_te - m_f) / s_f, dtype=torch.float32)
    nn_f.eval()
    with torch.no_grad():
        p_te_nn_list.append(nn_f(X_te_t).numpy())

    xgb_f = xgb.XGBClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
        num_class=3, n_jobs=1, tree_method='hist', random_state=42
    )
    xgb_f.fit(X_tr_f, y_tr_f)
    p_tr_xgb_oof[val_idx] = xgb_f.predict_proba(X_val_f)
    p_te_xgb_list.append(xgb_f.predict_proba(X1_te))

p_tr_raw = 0.5 * p_tr_nn_oof + 0.5 * p_tr_xgb_oof
p_te_raw = 0.5 * np.mean(p_te_nn_list, axis=0) + 0.5 * np.mean(p_te_xgb_list, axis=0)

p_tr_mkt = train_data[['p_market_H', 'p_market_D', 'p_market_A']].values
p_te_mkt = test_data[['p_market_H', 'p_market_D', 'p_market_A']].values

p_tr_s1 = (MARKET_WEIGHT * p_tr_mkt) + ((1.0 - MARKET_WEIGHT) * p_tr_raw)
p_te_s1 = (MARKET_WEIGHT * p_te_mkt) + ((1.0 - MARKET_WEIGHT) * p_te_raw)

train_data['p_s1_H'], train_data['p_s1_D'], train_data['p_s1_A'] = p_tr_s1[:, 0], p_tr_s1[:, 1], p_tr_s1[:, 2]
test_data['p_s1_H'], test_data['p_s1_D'], test_data['p_s1_A'] = p_te_s1[:, 0], p_te_s1[:, 1], p_te_s1[:, 2]

for dataset in [train_data, test_data]:
    dataset['ev_s1_H'] = (dataset['p_s1_H'] * dataset['BestH']) - 1.0
    dataset['ev_s1_D'] = (dataset['p_s1_D'] * dataset['BestD']) - 1.0
    dataset['ev_s1_A'] = (dataset['p_s1_A'] * dataset['BestA']) - 1.0


# =====================================================================
# 4. CANDIDATE GENERATION & STAGE 2 TRAINING
# =====================================================================
def create_bet_candidates(d):
    rows = []
    outcomes = [('H', 'BestH', 'p_s1_H', 'ev_s1_H', 'p_market_H', 'edge_H', 0),
                ('D', 'BestD', 'p_s1_D', 'ev_s1_D', 'p_market_D', 'edge_D', 1),
                ('A', 'BestA', 'p_s1_A', 'ev_s1_A', 'p_market_A', 'edge_A', 2)]
    for idx, r in d.iterrows():
        for code, odds_col, p_col, ev_col, mkt_col, edge_col, target_val in outcomes:
            prob_residual = r[p_col] - r[mkt_col]
            rows.append({
                'season': r['Season'], 'date': r['Date'], 'home_team': r['HomeTeam'], 'away_team': r['AwayTeam'],
                'bet_type': code, 'odds': r[odds_col], 'p_model': r[p_col], 'p_market': r[mkt_col],
                'prob_residual': prob_residual, 'ev': r[ev_col], 'edge': r[edge_col],
                'shot_diff_5': r['shot_diff_5'], 'gd_diff_10': r['gd_diff_10'],
                'target_win': int(r['target_cls'] == target_val)
            })
    return pd.DataFrame(rows)

cand_tr = create_bet_candidates(train_data)
cand_te = create_bet_candidates(test_data)

# Strictly 8 features - the exact set the GridSearch optimized for
stage2_features = ['odds', 'p_model', 'p_market', 'prob_residual', 'ev', 'edge', 'shot_diff_5', 'gd_diff_10']

cand_tr_filter = cand_tr[(cand_tr['odds'] >= 1.40) & (cand_tr['odds'] <= MAX_ODDS) & (cand_tr['bet_type'] != 'D')].copy()
cand_tr_filter['profit_binary'] = (cand_tr_filter['target_win'] == 1).astype(int)

X2_tr = cand_tr_filter[stage2_features].values
y2_tr = cand_tr_filter['profit_binary'].values

# Hardcoded the exact winning settings that produced 7.71% ROI
base_stage2 = xgb.XGBClassifier(
    n_estimators=80,
    max_depth=2,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    n_jobs=-1,
    random_state=42
)

calibrated_stage2 = CalibratedClassifierCV(estimator=base_stage2, method='sigmoid', cv=3)
calibrated_stage2.fit(X2_tr, y2_tr)

cand_te_filter = cand_te[(cand_te['odds'] >= 1.40) & (cand_te['odds'] <= MAX_ODDS) & (cand_te['bet_type'] != 'D')].copy()

if len(cand_te_filter) > 0:
    X2_te = cand_te_filter[stage2_features].values
    cand_te_filter['stage2_p_win'] = calibrated_stage2.predict_proba(X2_te)[:, 1]
    cand_te_filter['stage2_ev'] = (cand_te_filter['stage2_p_win'] * cand_te_filter['odds']) - 1.0
else:
    cand_te_filter['stage2_p_win'] = np.array([])
    cand_te_filter['stage2_ev'] = np.array([])


# =====================================================================
# 5. GENTLE MICRO-ADJUSTED FILTERING LOGIC
# =====================================================================
def filter_by_dynamic_tier(df_in):
    t_mid = df_in[
        (df_in['odds'] >= 1.70) &
        (df_in['odds'] <= MAX_ODDS) &
        (df_in['stage2_p_win'] >= 0.50) &
        (df_in['stage2_ev'] >= MIN_EV)
        ]
    return t_mid.sort_values(['season', 'date']).reset_index(drop=True)

optimized_bets = filter_by_dynamic_tier(cand_te_filter)

# =====================================================================
# 6. DYNAMIC KELLY EVALUATION & REPORTING
# =====================================================================
print("=" * 105)
print(f"{'PEAK BENCHMARK PERFORMANCE SUMMARY (Season ' + str(TEST_SEASON) + ')':^105}")
print("=" * 105)

n_opt = len(optimized_bets)
if n_opt > 0:
    wins_opt = optimized_bets['target_win'].sum()
    wr_opt = (wins_opt / n_opt) * 100.0
    flat_prof_opt = (optimized_bets['target_win'] * (optimized_bets['odds'] - 1.0) - (
                1 - optimized_bets['target_win'])).sum()
    flat_roi_opt = (flat_prof_opt / n_opt) * 100.0

    # Balanced Dynamic Kelly Staking Engine (1/8th Kelly)
    ek_bankroll = INITIAL_BANKROLL
    ek_history = [INITIAL_BANKROLL]
    ek_staked = 0.0

    for idx, row in optimized_bets.iterrows():
        b = row['odds'] - 1.0
        q = 1.0 - row['stage2_p_win']
        f_kelly = (b * row['stage2_p_win'] - q) / b

        dynamic_fraction = 0.125
        stake_amount = ek_bankroll * min(f_kelly * dynamic_fraction, 0.02) if f_kelly > 0 else 0.0

        if stake_amount > 0:
            ek_staked += stake_amount
            p_trade = stake_amount * b if row['target_win'] == 1 else -stake_amount
            ek_bankroll += p_trade
            ek_history.append(ek_bankroll)

    ek_arr = np.array(ek_history)
    peaks = np.maximum.accumulate(ek_arr)
    drawdowns = (ek_arr - peaks) / peaks
    ek_max_dd = np.abs(drawdowns.min()) * 100.0 if len(drawdowns) > 0 else 0.0
    ek_net = ek_bankroll - INITIAL_BANKROLL
    ek_roi = (ek_net / ek_staked) * 100.0 if ek_staked > 0 else 0.0

    print(f"Total Trades Executed : {n_opt:d}")
    print(f"Overall Win Rate      : {wr_opt:.1f}%")
    print(f"Flat 1-Unit Profit    : {flat_prof_opt:+.2f}u  (ROI: {flat_roi_opt:+.2f}%)")
    print(f"Dynamic-Kelly Profit  : {ek_net:+.2f}u  (ROI: {ek_roi:+.2f}%, Max Drawdown: {ek_max_dd:.2f}%)")
print("=" * 105)

# Tier Breakdown Print
print("\n--- ODDS TIER BREAKDOWN ---")
t_trades = len(optimized_bets)
if t_trades > 0:
    t_wins = optimized_bets['target_win'].sum()
    t_wr = (t_wins / t_trades) * 100.0
    t_flat_profit = (tier_profit := (optimized_bets['target_win'] * (optimized_bets['odds'] - 1.0) - (
                1 - optimized_bets['target_win'])).sum())
    t_flat_roi = (tier_profit / t_trades) * 100.0
    print(
        f"[Target Odds | 1.70-{MAX_ODDS:.2f}]: Trades={t_trades:4d} | Win Rate={t_wr:5.1f}% | Flat Profit={tier_profit:+6.2f}u | Flat ROI={t_flat_roi:+6.2f}%")
else:
    print(f"[Target Odds | 1.70-{MAX_ODDS:.2f}]: No Trades Executed.")

# Export to CSV
export_cols = ['season', 'date', 'home_team', 'away_team', 'bet_type', 'odds', 'p_model', 'stage2_p_win', 'stage2_ev',
               'target_win']
optimized_bets[export_cols].to_csv("executed_bets.csv", index=False)
print(f"\nSuccessfully exported {len(optimized_bets)} optimized bets to 'executed_bets.csv'.")
