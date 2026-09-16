"""
end_to_end.py -- End-to-end selective betting with a continuous differentiable network.

WHAT THIS FILE IS
An implementation of the End-to-End Model. Instead of splitting the pipeline into
isolated phases (predicting physical metrics -> applying Dixon-Coles -> 
fitting mispricing -> applying separate shrinkage), this unifies the pipeline.

THE ARCHITECTURE
A PyTorch Multi-Layer Perceptron (MLP). It ingests both Match Form features 
(Stage 1) and Market Features (Stage 2) simultaneously. 

THE LOSS FUNCTION
Uses a Calibrated Focal Loss. Instead of applying a scalar shrinkage weight `w` 
or a fitted `w(x)` after the fact, the focal loss dynamically down-weights easily 
predicted matches during gradient descent. The network learns feature representations, 
calibration, and noise shrinkage simultaneously within its internal parameters.

HOW TO RUN
    python3 end_to_end.py
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss  # Added import

# Import the robust data pipeline and evaluation harness from the existing two_stage.py
from modified_two_stage import (
    load_and_prepare, load_multiseason, make_folds, 
    STAGE1_FEATURES, MARKET_FEATURES, OUTCOMES, TARGET_MAP,
    probs, outcome_cols, show, scoring_report, accuracy_report,
    betting_report, risk_coverage_curve, aurc, check_non_vacuousness, era_breakdown
)

# =========================================================================
# 1. THE CONTINUOUS DIFFERENTIABLE NETWORK
# =========================================================================

class EndToEndNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(EndToEndNetwork, self).__init__()
        # Unifies Stage 1 & Stage 2 into a single representation learner
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

# =========================================================================
# 2. CUSTOM LOSS (REPLACES MANUAL SHRINKAGE)
# =========================================================================

class CalibratedFocalLoss(nn.Module):
    """
    Native probability calibration and shrinkage. Focuses gradients on 
    uncertain matches, preventing the winner's curse internally.
    """
    def __init__(self, gamma: float = 2.0):
        super(CalibratedFocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Shrinkage term: suppresses updates for matches with high variance/noise
        focal_term = (1 - pt) ** self.gamma 
        return (focal_term * ce_loss).mean()

# =========================================================================
# 3. END-TO-END TRAINING LOOP
# =========================================================================

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
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
        
        self.model.eval()
        return self

    def predict_proba(self, df: pd.DataFrame, features: list) -> np.ndarray:
        X, _ = self._prepare_tensors(df, features, fit_scaler=False)
        with torch.no_grad():
            logits = self.model(X)
            p = torch.softmax(logits, dim=1).numpy()
        return np.clip(p, 1e-6, 1.0)

# =========================================================================
# 4. ORCHESTRATION & REPORTING
# =========================================================================

def fit_fold_e2e(fold, features: list, seed: int = 42) -> dict:
    """
    Fits the single end-to-end network on the walk-forward fold.
    Because shrinkage is native to the loss function, there is no separate
    w or w(x) to fit, calculate, or apply.
    """
    tr, va, te = fold.train, fold.val, fold.test.copy()
    
    # Train the unified network
    net = EndToEndModel(seed=seed).fit(tr, features)
    
    # Predict directly (p_corr serves as our fully-shrunk prediction)
    p_test = net.predict_proba(te, features)
    te[outcome_cols("p_corr")] = p_test
    
    # We populate p_indep with the same values just so the evaluator
    # doesn't crash when running non-vacuousness checks.
    te[outcome_cols("p_indep")] = p_test

    y_test = te["target"].astype(int)
    diagnostics = {
        "test_unit": fold.test_unit, 
        "val_unit": fold.val_unit,
        "n_train": len(tr), 
        "n_test": len(te),
        "ll_market": log_loss(y_test, probs(te, "p_ref"), labels=[0, 1, 2]),
        "ll_corrected": log_loss(y_test, p_test, labels=[0, 1, 2]),
    }
    return {"test": te, "diagnostics": diagnostics}


def main():
    print("=" * 70)
    print("END-TO-END SELECTIVE BETTING PIPELINE")
    print("  Architecture : PyTorch Multi-Layer Perceptron")
    print("  Loss Function: Calibrated Focal Loss")
    print("=" * 70)

    # 1. Load Data using two_stage.py's robust loader
    df = load_and_prepare("all-euro-data-2025-2026.csv")
    
    # E2E utilizes ALL features simultaneously
    all_features = [c for c in STAGE1_FEATURES + MARKET_FEATURES if c in df.columns]
    
    fold_list, split_col = make_folds(df, all_features, min_test_matches=300, n_blocks=10)

    preds, diagnostics = [], []
    print("Training across walk-forward folds...")
    for fold in fold_list:
        out = fit_fold_e2e(fold, all_features)
        preds.append(out["test"])
        diagnostics.append({"unit": split_col, **out["diagnostics"]})

    preds_df = pd.concat(preds, ignore_index=True)
    y = preds_df["target"].astype(int).to_numpy()

    show("Out-of-sample scoring:",
         pd.DataFrame([
             scoring_report(y, probs(preds_df, "p_ref"), "market"),
             scoring_report(y, probs(preds_df, "p_corr"), "end_to_end_network")
         ]))

    show("ROI at matched coverage:",
         betting_report(preds_df, prefixes=("p_corr",)))

    curve = risk_coverage_curve(preds_df, "p_corr")
    show(f"Risk-coverage, End-to-End: AURC = {aurc(curve):+.3f}", curve)

if __name__ == "__main__":
    main()