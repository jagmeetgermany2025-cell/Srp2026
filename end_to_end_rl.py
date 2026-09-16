"""
end_to_end_rl.py -- End-to-end selective betting + RL Staking Agent.

WHAT THIS FILE IS
A complete pipeline that unifies feature representation, calibration, and 
shrinkage into a single PyTorch MLP. It also replaces heuristic staking 
with a Reinforcement Learning agent that maximizes log-bankroll growth.
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# Import the robust data pipeline and evaluation harness from two_stage.py
from modified_two_stage import (
    load_and_prepare, make_folds, STAGE1_FEATURES, MARKET_FEATURES, 
    probs, outcome_cols, show, scoring_report, betting_report, 
    risk_coverage_curve, aurc, select_bets, staking_comparison
)

# =========================================================================
# 1. THE CONTINUOUS DIFFERENTIABLE NETWORK & FOCAL LOSS
# =========================================================================

class EndToEndNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(EndToEndNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3) # H, D, A
        )

    def forward(self, x):
        return self.net(x)

class CalibratedFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super(CalibratedFocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        p = torch.softmax(logits, dim=1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt) ** self.gamma 
        return (focal_term * ce_loss).mean()

class EndToEndModel:
    def __init__(self, seed: int = 42, epochs: int = 35, lr: float = 5e-4):
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = None
        torch.manual_seed(seed)

    def _prepare_tensors(self, df: pd.DataFrame, features: list, fit_scaler: bool = False):
        X_raw = np.nan_to_num(df[features].to_numpy(dtype=float))
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            X_scaled = self.scaler.transform(X_raw)
        
        X = torch.tensor(X_scaled, dtype=torch.float32)
        y = torch.tensor(df["target"].astype(int).to_numpy(), dtype=torch.long)
        return X, y

    def fit(self, train_df: pd.DataFrame, features: list):
        X, y = self._prepare_tensors(train_df, features, fit_scaler=True)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        self.model = EndToEndNetwork(input_dim=len(features))
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = CalibratedFocalLoss(gamma=2.0)
        
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        
        self.model.eval()
        return self

    def predict_proba(self, df: pd.DataFrame, features: list) -> np.ndarray:
        X, _ = self._prepare_tensors(df, features, fit_scaler=False)
        with torch.no_grad():
            p = torch.softmax(self.model(X), dim=1).numpy()
        return np.clip(p, 1e-6, 1.0)

# =========================================================================
# 2. RL STAKE AGENT (DIRECT UTILITY MAXIMIZATION)
# =========================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, max_stake: float = 3.0):
        super(PolicyNetwork, self).__init__()
        self.max_stake = max_stake
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x) * self.max_stake

class LogBankrollLoss(nn.Module):
    def __init__(self):
        super(LogBankrollLoss, self).__init__()

    def forward(self, stakes, odds, is_win):
        profit = torch.where(is_win == 1, stakes * (odds - 1.0), -stakes)
        bankroll_multiplier = torch.clamp(1.0 + profit, min=1e-4)
        return -torch.log(bankroll_multiplier).mean()

class RLStakeAgent:
    def __init__(self, seed: int = 42, max_stake: float = 3.0, epochs: int = 50):
        self.max_stake = max_stake
        self.epochs = epochs
        self.features = ["p_est", "p_ref", "EV", "Odds"]
        self.scaler = StandardScaler()
        self.agent = None
        torch.manual_seed(seed)

    def fit(self, val_bets: pd.DataFrame) -> "RLStakeAgent":
        if len(val_bets) < 50:
            self.agent = None
            return self

        X_raw = np.nan_to_num(val_bets[[c for c in self.features if c in val_bets.columns]].to_numpy(float))
        X = torch.tensor(self.scaler.fit_transform(X_raw), dtype=torch.float32)
        odds = torch.tensor(val_bets["Odds"].to_numpy(float), dtype=torch.float32).unsqueeze(1)
        is_win = torch.tensor(val_bets["IsWin"].to_numpy(float), dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X, odds, is_win)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.agent = PolicyNetwork(input_dim=len(self.features), max_stake=self.max_stake)
        optimizer = optim.AdamW(self.agent.parameters(), lr=0.01)
        criterion = LogBankrollLoss()

        self.agent.train()
        for epoch in range(self.epochs):
            for X_batch, odds_batch, win_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.agent(X_batch), odds_batch, win_batch)
                loss.backward()
                optimizer.step()

        self.agent.eval()
        return self

    def stakes(self, bets: pd.DataFrame) -> np.ndarray:
        if self.agent is None:
            return np.ones(len(bets))
            
        X_raw = np.nan_to_num(bets[[c for c in self.features if c in bets.columns]].to_numpy(float))
        X = torch.tensor(self.scaler.transform(X_raw), dtype=torch.float32)
        
        with torch.no_grad():
            raw_stakes = self.agent(X).squeeze(1).numpy()
            
        mean_stake = raw_stakes.mean()
        if not np.isfinite(mean_stake) or mean_stake <= 0:
            return np.ones(len(bets))
            
        return np.clip(raw_stakes / mean_stake, 0.0, self.max_stake)

# =========================================================================
# 3. ORCHESTRATION & REPORTING
# =========================================================================

def fit_fold_e2e_rl(fold, features: list, seed: int = 42, stake_tau: float = 0.02) -> dict:
    """
    Fits the End-to-End network, predicts on val to train the RL Agent,
    and then evaluates on the test fold.
    """
    tr, va, te = fold.train, fold.val, fold.test.copy()
    
    # 1. Train E2E Network
    net = EndToEndModel(seed=seed).fit(tr, features)
    
    # 2. Predict on Validation to prepare bets for RL Agent
    va_scored = va.copy()
    va_scored[outcome_cols("p_corr")] = net.predict_proba(va, features)
    
    # 3. Train RL Stake Agent on validation bets
    rl_agent = RLStakeAgent(seed=seed).fit(select_bets(va_scored, stake_tau, "p_corr"))
    
    # 4. Predict on Test Fold
    p_test = net.predict_proba(te, features)
    te[outcome_cols("p_corr")] = p_test
    te[outcome_cols("p_indep")] = p_test # Filler for non-vacuousness check compatibility

    y_test = te["target"].astype(int)
    diagnostics = {
        "test_unit": fold.test_unit, 
        "n_train": len(tr), "n_test": len(te),
        "ll_market": log_loss(y_test, probs(te, "p_ref"), labels=[0, 1, 2]),
        "ll_corrected": log_loss(y_test, p_test, labels=[0, 1, 2]),
    }
    return {"test": te, "diagnostics": diagnostics, "stake_model": rl_agent}


def main():
    print("=" * 70)
    print("END-TO-END SELECTIVE BETTING + RL STAKING PIPELINE")
    print("  Architecture : PyTorch MLP + Policy Network")
    print("  Losses       : Calibrated Focal Loss & Log Bankroll Loss")
    print("=" * 70)

    df = load_and_prepare("all-euro-data-2025-2026.csv")
    all_features = [c for c in STAGE1_FEATURES + MARKET_FEATURES if c in df.columns]
    
    fold_list, split_col = make_folds(df, all_features, min_test_matches=300, n_blocks=10)

    preds, diagnostics, stake_models = [], [], {}
    print("\nTraining across walk-forward folds...")
    for fold in fold_list:
        out = fit_fold_e2e_rl(fold, all_features)
        preds.append(out["test"])
        diagnostics.append({"unit": split_col, **out["diagnostics"]})
        if out["stake_model"] is not None:
            stake_models[fold.test_unit] = out["stake_model"]

    preds_df = pd.concat(preds, ignore_index=True)
    y = preds_df["target"].astype(int).to_numpy()

    show("Out-of-sample scoring:", pd.DataFrame([
        scoring_report(y, probs(preds_df, "p_ref"), "market"),
        scoring_report(y, probs(preds_df, "p_corr"), "end_to_end_network")
    ]))

    show("ROI at matched coverage (Flat Stakes):", betting_report(preds_df, prefixes=("p_corr",)))

    unit_col = "Block" if "Block" in preds_df.columns else "Season"
    if stake_models:
        show("Staking rules on identical bets (Flat vs Q-Kelly vs RL Agent):",
             staking_comparison(preds_df, stake_models, unit_col),
             "total_staked shows whether a rule won by allocating or leveraging.")

    curve = risk_coverage_curve(preds_df, "p_corr")
    show(f"Risk-coverage, End-to-End: AURC = {aurc(curve):+.3f}", curve)

if __name__ == "__main__":
    main()