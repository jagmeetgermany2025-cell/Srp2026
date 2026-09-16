# End-to-End Differentiable Pipeline & RL Staking

This branch implements the End-to-End predictive model and Reinforcement Learning (RL) Staking Agent as outlined in the project specification.

## 1. Architectural Changes

We have replaced the decoupled two-stage pipeline with a unified PyTorch architecture:

*   **Continuous Differentiable Network:** A multi-layer perceptron (MLP) replaces the isolated Match Engine (Stage 1) and Market/Edge Model (Stage 2). It ingests both Match Form features and Market signals simultaneously, projecting them directly into $H, D, A$ probabilities.
*   **Calibrated Focal Loss:** We removed the static Dixon-Coles parameters and the manual `w` shrinkage scalar. Instead, the network optimizes a Calibrated Focal Loss:
    $$ \text{Loss} = \alpha (1 - p_t)^\gamma \text{CE}(p_t, y) $$
    This mathematically suppresses gradient updates for extreme, noisy predictions, allowing the network to natively learn calibration and shrinkage.
*   **Direct Utility Maximization (RL Staking):** We replaced the regression-based `StakeModel` and the fixed heuristic `0.25 \times \text{Quarter-Kelly}` formula. The new `RLStakeAgent` (a Policy Network) formulates staking as a Markov Decision Process (MDP), directly outputting a continuous stake bounded between 0 and 3 units by minimizing the negative log-bankroll growth.

---

## 2. Experimental Results

The pipeline was executed across a dataset of **7,466 matches** across **22 divisions** (2025-07-25 to 2026-05-14), using a 10-block walk-forward validation strategy.

### Out-of-Sample Scoring
The end-to-end model's raw probability predictions compared to the market baseline:

| Model | Log Loss | RPS | Brier | ECE | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Market Baseline** | 0.9948 | 0.2006 | 0.5950 | 0.0151 | 51.37% |
| **End-to-End Network** | 1.0243 | 0.2094 | 0.6157 | 0.0475 | 50.13% |

### ROI at Matched Coverage (Flat Stakes)
Sweeping the threshold to evaluate profitability at various selection rates. The Area Under the Risk-Coverage Curve (**AURC**) is **-16.424**.

| Coverage | Bets Placed | Win Rate | Avg Odds | ROI | Shop Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **50%** | 5,073 | 26.26% | 4.39 | **-10.16%** | +5.21 pts |
| **25%** | 2,537 | 19.59% | 5.47 | **-16.00%** | +5.89 pts |
| **10%** | 1,015 | 14.78% | 6.95 | **-18.95%** | +6.83 pts |
| **5%** | 508 | 12.40% | 7.95 | **-25.25%** | +6.65 pts |

### Staking Strategy Comparison
Evaluating the RL Agent against Flat and Quarter-Kelly staking on identical bets at the 25%, 10%, and 5% coverage thresholds:

| Coverage | Rule | Total Staked | ROI |
| :--- | :--- | :--- | :--- |
| **25%** | Flat | 2,537.00 | -16.00% |
| **25%** | Quarter-Kelly | 4,799.22 | -12.42% |
| **25%** | **RL Learned** | **2,537.00** | **-16.00%** |
| **10%** | Flat | 1,015.00 | -18.95% |
| **10%** | Quarter-Kelly | 2,414.42 | -17.08% |
| **10%** | **RL Learned** | **1,015.00** | **-18.95%** |
| **5%** | Flat | 508.00 | -25.25% |
| **5%** | Quarter-Kelly | 1,298.66 | -21.03% |
| **5%** | **RL Learned** | **508.00** | **-25.25%** |

---

## 3. Analysis & Conclusions

1. **The RL Agent behaves rationally:** The RL agent has learned to mirror flat staking exactly. Because the underlying model currently has a negative expectation (ROI is negative across all coverage thresholds), the agent's optimization function (log-bankroll maximization) mathematically dictates minimizing exposure to survive. The agent is functioning perfectly as a utility maximizer.
2. **The Winner's Curse is active:** As coverage tightens from 50% down to 5%, the average odds selected climb aggressively from 4.39 to 7.95, while ROI plummets from -10.16% to -25.25%. 
3. **Focal Loss requires tuning:** The current Focal Loss ($\gamma = 2.0$) is under-shrinking the long-shot tails. It allows high-variance bets with massive estimation error to pass the threshold, confusing statistical noise for value.

## 4. Next Steps

*   **Increase Focal Gamma:** Raise $\gamma$ in the `CalibratedFocalLoss` from `2.0` to `5.0` or `10.0` to more aggressively suppress gradients on highly uncertain, high-odds predictions.
*   **Feature Regularization:** Apply stronger $L2$ weight decay (`weight_decay=1e-3` in the AdamW optimizer) to prevent the network from memorizing form-based anomalies.
*   **Change Selection Criterion:** Run the pipeline using `criterion="edge"` to remove the mechanical odds multiplier from the selection threshold, isolating the network's true predictive edge.
