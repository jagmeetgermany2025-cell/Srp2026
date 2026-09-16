"""
staking_rl_agent.py -- Reinforcement Learning agent for optimal stake sizing.

WHAT THIS FILE IS
An implementation of the "Learn Strategy" phase. Instead of applying fixed 
heuristic rules like 0.25 * Kelly, this trains a neural agent to directly 
output continuous stake amounts.

THE ARCHITECTURE
A continuous Policy Network (Actor). It observes the state of the bet 
(our estimated probability, the market probability, expected value, and odds)
and outputs a stake between 0 and max_stake.

THE REWARD
Direct Utility Maximization. The agent is trained using a custom loss function 
that optimizes log-bankroll growth. By maximizing the log return, the network 
natively learns a Kelly-like risk-adjusted sizing strategy without manual formulas.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

class PolicyNetwork(nn.Module):
    """
    The RL Agent that observes the betting state and outputs a continuous stake.
    """
    def __init__(self, input_dim: int, max_stake: float = 3.0):
        super(PolicyNetwork, self).__init__()
        self.max_stake = max_stake
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Bounds output between 0 and 1
        )

    def forward(self, x):
        # Scale sigmoid output to [0, max_stake]
        return self.net(x) * self.max_stake

class LogBankrollLoss(nn.Module):
    """
    Direct Utility Maximization.
    Optimizes for log-bankroll growth (the Kelly criterion objective) directly.
    """
    def __init__(self):
        super(LogBankrollLoss, self).__init__()

    def forward(self, stakes, odds, is_win):
        # If win:  Return = Stake * (Odds - 1)
        # If loss: Return = -Stake
        profit = torch.where(
            is_win == 1,
            stakes * (odds - 1.0),
            -stakes
        )
        
        # Bankroll multiplier (assuming initial bankroll of 1.0 per batch element for scaling)
        # We add 1.0 + profit. We clamp to a small positive number to avoid log(0) bankruptcies.
        bankroll_multiplier = torch.clamp(1.0 + profit, min=1e-4)
        
        # We want to MAXIMIZE log bankroll, so we MINIMIZE the negative log bankroll
        loss = -torch.log(bankroll_multiplier).mean()
        return loss

class RLStakeAgent:
    """
    Wrapper class compatible with the pipeline's StakeModel interface.
    Replaces the regression-based StakeModel.
    """
    def __init__(self, seed: int = 42, max_stake: float = 3.0, epochs: int = 50):
        self.seed = seed
        self.max_stake = max_stake
        self.epochs = epochs
        self.features = ["p_est", "p_ref", "EV", "Odds"]
        self.scaler = StandardScaler()
        self.agent = None
        torch.manual_seed(seed)

    def _prepare_tensors(self, bets: pd.DataFrame, fit_scaler: bool = False):
        cols = [c for c in self.features if c in bets.columns]
        X_raw = np.nan_to_num(bets[cols].to_numpy(float))
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            X_scaled = self.scaler.transform(X_raw)
            
        X = torch.tensor(X_scaled, dtype=torch.float32)
        odds = torch.tensor(bets["Odds"].to_numpy(float), dtype=torch.float32).unsqueeze(1)
        is_win = torch.tensor(bets["IsWin"].to_numpy(float), dtype=torch.float32).unsqueeze(1)
        
        return X, odds, is_win

    def fit(self, val_bets: pd.DataFrame) -> "RLStakeAgent":
        if len(val_bets) < 50:
            # Fallback if there are too few bets to train the agent safely
            self.agent = None
            return self

        X, odds, is_win = self._prepare_tensors(val_bets, fit_scaler=True)
        dataset = TensorDataset(X, odds, is_win)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.agent = PolicyNetwork(input_dim=len(self.features), max_stake=self.max_stake)
        optimizer = optim.AdamW(self.agent.parameters(), lr=0.01)
        criterion = LogBankrollLoss()

        self.agent.train()
        for epoch in range(self.epochs):
            for X_batch, odds_batch, win_batch in loader:
                optimizer.zero_grad()
                stakes = self.agent(X_batch)
                loss = criterion(stakes, odds_batch, win_batch)
                loss.backward()
                optimizer.step()

        self.agent.eval()
        return self

    def stakes(self, bets: pd.DataFrame) -> np.ndarray:
        if self.agent is None:
            return np.ones(len(bets))
            
        X, _, _ = self._prepare_tensors(bets, fit_scaler=False)
        with torch.no_grad():
            raw_stakes = self.agent(X).squeeze(1).numpy()
            
        mean_stake = raw_stakes.mean()
        if not np.isfinite(mean_stake) or mean_stake <= 0:
            return np.ones(len(bets))
            
        # Rescale so the average exposure matches flat staking for fair comparison
        scaled_stakes = np.clip(raw_stakes / mean_stake, 0.0, self.max_stake)
        return scaled_stakes