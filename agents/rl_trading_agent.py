"""
Reinforcement Learning Trading Agent (PPO)
───────────────────────────────────────────
Learns a portfolio allocation policy via Proximal Policy Optimisation.

Architecture
  ActorCritic network (shared MLP trunk, separate heads):
    • Actor  → softmax logits → portfolio weights (n_assets + 1 cash slot)
    • Critic → scalar state value

State per step (flat vector):
  For each of the top-N assets:
    [return_1d, return_5d, volatility_21d, rsi_norm,
     momentum_21d, ml_pred_return, ml_confidence, fused_score]  (8 features)
  Market state:
    [spy_return_21d, vix_level, crash_probability]              (3 features)
  Current portfolio weights:
    [w_0, w_1, ..., w_{N-1}, w_cash]                           (N+1 features)

  Total = N × 9 + 4

Action
  Continuous softmax allocation over N assets + cash.
  → portfolio_weights[N+1] summing to 1.

Reward
  r_t = portfolio_return_t
        − λ_drawdown × max_drawdown_since_start
        − λ_vol       × portfolio_daily_vol

Training
  • Walk-forward on historical featured_data / raw_data
  • Configurable n_episodes, n_epochs (fast-mode defaults are quick on CPU)
  • Model saved to rl_model_path after training

Inference
  • Load saved weights if available; else use untrained network
    (falls back to equal-weight allocation gracefully)
"""

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import setup_logger

logger = setup_logger("rl_trading_agent")

# ── Lazy imports — only fail at runtime if torch is absent ────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Dirichlet
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not found — RL agent will use equal-weight fallback")

_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Neural network
# ─────────────────────────────────────────────────────────────────────────────

class _ActorCritic(nn.Module if _TORCH_AVAILABLE else object):
    """Shared-trunk MLP: actor outputs Dirichlet concentration params, critic outputs V(s)."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden // 2, n_actions)
        self.critic_head = nn.Linear(hidden // 2, 1)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = self.trunk(x)
        # Dirichlet concentration > 0; softplus ensures positivity
        concentrations = F.softplus(self.actor_head(h)) + 0.1
        value          = self.critic_head(h).squeeze(-1)
        return concentrations, value

    def get_action_value(
        self, x: "torch.Tensor", deterministic: bool = False
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        concentrations, value = self(x)
        dist   = Dirichlet(concentrations)
        action = concentrations / concentrations.sum(-1, keepdim=True) if deterministic else dist.rsample()
        log_p  = dist.log_prob(action.clamp(_EPS, 1 - _EPS))
        entropy = dist.entropy()
        return action, log_p, entropy, value


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio simulation environment
# ─────────────────────────────────────────────────────────────────────────────

class _PortfolioEnv:
    """
    Gym-style environment that simulates portfolio performance.

    Observation space : flat feature vector (state_dim,)
    Action space      : portfolio weights for n_assets + cash (sums to 1)
    """

    def __init__(
        self,
        price_matrix:    np.ndarray,    # (T, N)
        feature_matrix:  np.ndarray,    # (T, N, F)
        market_features: np.ndarray,    # (T, 3)  [spy_ret, vix, crash_prob]
        episode_len:     int = 63,
        lambda_dd:       float = 0.10,
        lambda_vol:      float = 0.05,
        transaction_cost:float = 0.001,
    ):
        self.price_matrix    = price_matrix      # (T, N)
        self.feature_matrix  = feature_matrix    # (T, N, F)
        self.market_features = market_features   # (T, 3)
        self.episode_len     = episode_len
        self.lambda_dd       = lambda_dd
        self.lambda_vol      = lambda_vol
        self.tc              = transaction_cost

        self.T, self.N    = price_matrix.shape
        self.F            = feature_matrix.shape[2]
        n_actions         = self.N + 1                     # assets + cash
        self.state_dim    = self.N * (self.F + 1) + 3 + 1  # feats + cur_wts + mkt + cash_wt

        # Filled by reset()
        self.t      = 0
        self.t_end  = 0
        self.weights = np.ones(n_actions) / n_actions
        self.portfolio_value = 1.0
        self.peak_value      = 1.0
        self.daily_rets: List[float] = []

    def reset(self, start: Optional[int] = None) -> np.ndarray:
        max_start = max(0, self.T - self.episode_len - 1)
        self.t     = np.random.randint(0, max(1, max_start)) if start is None else start
        self.t_end = min(self.t + self.episode_len, self.T - 1)

        self.weights       = np.ones(self.N + 1) / (self.N + 1)
        self.portfolio_value = 1.0
        self.peak_value    = 1.0
        self.daily_rets    = []
        return self._obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        # action: weight allocation (N+1,), must sum to 1
        action = np.clip(action, 0.0, 1.0)
        s = action.sum()
        if s < _EPS:
            action = np.ones_like(action) / len(action)
        else:
            action = action / s

        # Transaction cost: proportional to total weight change
        turnover = float(np.abs(action - self.weights).sum()) / 2.0
        tc_penalty = self.tc * turnover

        # Apply portfolio return
        if self.t + 1 < self.T:
            price_now  = self.price_matrix[self.t]
            price_next = self.price_matrix[self.t + 1]
            valid = (price_now > 0) & (price_next > 0) & ~np.isnan(price_now) & ~np.isnan(price_next)
            asset_rets = np.where(valid, price_next / price_now - 1.0, 0.0)
        else:
            asset_rets = np.zeros(self.N)

        # Portfolio return (equity weights only, cash = 0 return)
        equity_ret  = float(np.dot(action[: self.N], asset_rets))
        net_ret     = equity_ret - tc_penalty
        self.portfolio_value *= (1.0 + net_ret)
        self.daily_rets.append(net_ret)

        # Update peak for drawdown tracking
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Portfolio volatility (rolling 10-day)
        recent = self.daily_rets[-10:] if len(self.daily_rets) >= 2 else [0.0, 0.0]
        port_vol = float(np.std(recent)) * math.sqrt(252)

        # Reward
        reward = net_ret - self.lambda_dd * drawdown - self.lambda_vol * port_vol

        self.weights = action.copy()
        self.t      += 1
        done         = (self.t >= self.t_end)
        return self._obs(), reward, done

    def _obs(self) -> np.ndarray:
        t = min(self.t, self.T - 1)
        feats   = self.feature_matrix[t].reshape(-1)    # N*F
        mkt     = self.market_features[t]               # 3
        wts     = self.weights                           # N+1
        obs     = np.concatenate([feats, mkt, wts], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# PPO rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

class _PPOBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.states   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.values   = np.zeros(capacity, dtype=np.float32)
        self.log_probs= np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.cap = capacity

    def push(self, s, a, r, v, lp, done):
        i = self.ptr % self.cap
        self.states[i]    = s
        self.actions[i]   = a
        self.rewards[i]   = r
        self.values[i]    = v
        self.log_probs[i] = lp
        self.dones[i]     = float(done)
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.cap

    def compute_returns(self, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        n = min(self.ptr, self.cap)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for i in reversed(range(n)):
            next_val = 0.0 if self.dones[i] else (self.values[i + 1] if i + 1 < n else 0.0)
            delta    = self.rewards[i] + gamma * next_val - self.values[i]
            gae      = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages[i] = gae
        returns = advantages + self.values[:n]
        return advantages, returns

    def clear(self):
        self.ptr = 0


# ─────────────────────────────────────────────────────────────────────────────
# Main RL Trading Agent
# ─────────────────────────────────────────────────────────────────────────────

_STOCK_FEATURES = [
    "return_1d", "return_5d", "volatility_21d",
    "rsi_14", "momentum_21d",
]
_MARKET_FEATURES = ["spy_return_21d", "vix_level"]
_N_ASSET_FEATURES = len(_STOCK_FEATURES)  # will be extended by ml/fused cols at runtime


class RLTradingAgent:
    """PPO-based portfolio allocation agent."""

    def __init__(self, config: dict):
        rl = config.get("rl_agent", {})
        self.n_assets        = int(rl.get("n_assets",        10))
        self.hidden          = int(rl.get("hidden",          256))
        self.n_episodes      = int(rl.get("n_episodes",       8))
        self.n_epochs        = int(rl.get("n_epochs",         4))
        self.episode_len     = int(rl.get("episode_len",     126))
        self.rollout_len     = int(rl.get("rollout_len",     252))
        self.batch_size      = int(rl.get("batch_size",       64))
        self.lr              = float(rl.get("lr",           3e-4))
        self.gamma           = float(rl.get("gamma",         0.99))
        self.gae_lambda      = float(rl.get("gae_lambda",    0.95))
        self.clip_eps        = float(rl.get("clip_eps",       0.2))
        self.entropy_coef    = float(rl.get("entropy_coef",  0.01))
        self.value_coef      = float(rl.get("value_coef",    0.5))
        self.lambda_dd       = float(rl.get("lambda_drawdown",0.10))
        self.lambda_vol      = float(rl.get("lambda_vol",    0.05))
        self.transaction_cost= float(rl.get("transaction_cost",0.001))
        self.model_path      = rl.get("model_path",          "./models/cache/rl_model.pt")
        self.training_enabled= bool(rl.get("training_enabled", True))

        self._net:    Optional[_ActorCritic] = None
        self._device: Optional[str]         = None
        self._tickers: List[str]            = []
        self._n_features: int               = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        raw_data:       Dict[str, pd.DataFrame],
        featured_data:  Dict[str, pd.DataFrame],
        ml_predictions: Dict[str, Dict[str, dict]],
        fused_scores:   Dict[str, Dict[str, float]],
        crash_probability: float = 0.0,
    ) -> bool:
        """Train PPO on historical data. Returns True if training succeeded."""
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable — skipping RL training")
            return False
        if not self.training_enabled:
            logger.info("RL training disabled in config — loading saved model if exists")
            return self._load_model()

        tickers, pm, fm, mkt = self._build_matrices(
            raw_data, featured_data, ml_predictions, fused_scores, crash_probability
        )
        if pm is None or pm.shape[0] < 30:
            logger.warning("Insufficient data for RL training")
            return False

        self._tickers   = tickers
        self._n_features = fm.shape[2]
        n_actions        = len(tickers) + 1
        state_dim        = len(tickers) * (self._n_features + 1) + 3 + 1
        self._device     = "cuda" if torch.cuda.is_available() else "cpu"
        self._net        = _ActorCritic(state_dim, n_actions, self.hidden).to(self._device)
        opt              = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        env    = _PortfolioEnv(pm, fm, mkt, self.episode_len, self.lambda_dd, self.lambda_vol, self.transaction_cost)
        buffer = _PPOBuffer(state_dim, n_actions, self.rollout_len)

        logger.info(
            "Training RL agent: %d assets, state_dim=%d, episodes=%d, device=%s",
            len(tickers), state_dim, self.n_episodes, self._device,
        )

        for episode in range(self.n_episodes):
            obs = env.reset()
            buffer.clear()
            ep_reward = 0.0

            while not buffer.full():
                s_t   = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    action, log_p, _, value = self._net.get_action_value(s_t)
                a_np   = action.squeeze(0).cpu().numpy()
                v_np   = float(value.squeeze(0).cpu().item())
                lp_np  = float(log_p.cpu().item())

                obs_next, reward, done = env.step(a_np)
                buffer.push(obs, a_np, reward, v_np, lp_np, done)
                ep_reward += reward
                obs = obs_next if not done else env.reset()

            advantages, returns = buffer.compute_returns(self.gamma, self.gae_lambda)

            # Normalise advantages
            adv_mean = advantages.mean()
            adv_std  = advantages.std() + _EPS
            advantages = (advantages - adv_mean) / adv_std

            n = buffer.ptr
            # PPO update over n_epochs mini-batches
            total_loss = 0.0
            for _ in range(self.n_epochs):
                idxs = np.random.permutation(n)
                for start in range(0, n, self.batch_size):
                    b = idxs[start: start + self.batch_size]
                    s  = torch.FloatTensor(buffer.states[b]).to(self._device)
                    a  = torch.FloatTensor(buffer.actions[b]).to(self._device)
                    lp_old = torch.FloatTensor(buffer.log_probs[b]).to(self._device)
                    adv    = torch.FloatTensor(advantages[b]).to(self._device)
                    ret    = torch.FloatTensor(returns[b]).to(self._device)

                    concentrations, value = self._net(s)
                    dist    = Dirichlet(concentrations)
                    log_p   = dist.log_prob(a.clamp(_EPS, 1 - _EPS))
                    entropy = dist.entropy().mean()

                    ratio    = torch.exp(log_p - lp_old)
                    surr1    = ratio * adv
                    surr2    = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                    actor_l  = -torch.min(surr1, surr2).mean()
                    critic_l = F.mse_loss(value, ret)
                    loss     = actor_l + self.value_coef * critic_l - self.entropy_coef * entropy

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
                    opt.step()
                    total_loss += loss.item()

            avg_loss = total_loss / max(1, self.n_epochs * max(1, n // self.batch_size))
            logger.info(
                "Episode %d/%d  ep_reward=%.4f  avg_loss=%.4f",
                episode + 1, self.n_episodes, ep_reward, avg_loss,
            )

        self._save_model()
        return True

    def predict(
        self,
        featured_data:     Dict[str, pd.DataFrame],
        ml_predictions:    Dict[str, Dict[str, dict]],
        fused_scores:      Dict[str, Dict[str, float]],
        crash_probability: float = 0.0,
        current_weights:   Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Inference: return portfolio weights for the current market state.

        Returns
        -------
        {
            "weights":             {ticker: float},
            "cash_weight":         float,
            "recommended_action":  str,
            "confidence":          float,
        }
        """
        tickers = self._tickers or sorted(
            set(featured_data) & set(ml_predictions) & set(fused_scores)
        )[: self.n_assets]

        if not _TORCH_AVAILABLE or self._net is None:
            return self._equal_weight_fallback(tickers, crash_probability)

        # Build latest state vector
        obs = self._build_latest_obs(
            tickers, featured_data, ml_predictions, fused_scores,
            crash_probability, current_weights,
        )
        s = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            action, _, _, _ = self._net.get_action_value(s, deterministic=True)
        weights_arr = action.squeeze(0).cpu().numpy()

        # Map to tickers
        n = len(tickers)
        asset_weights = {t: round(float(weights_arr[i]), 4) for i, t in enumerate(tickers)}
        cash_weight   = round(float(weights_arr[n]) if n < len(weights_arr) else 0.0, 4)

        # Normalise to sum to 1
        total = sum(asset_weights.values()) + cash_weight
        if total > _EPS:
            asset_weights = {t: w / total for t, w in asset_weights.items()}
            cash_weight   = cash_weight / total

        confidence = 1.0 - crash_probability
        action_label = _crash_to_action(crash_probability)

        logger.info(
            "RL inference — top allocations: %s  cash=%.1f%%  action=%s",
            {t: f"{w:.1%}" for t, w in sorted(asset_weights.items(), key=lambda x: -x[1])[:5]},
            cash_weight * 100, action_label,
        )
        return {
            "weights":            asset_weights,
            "cash_weight":        round(float(cash_weight), 4),
            "recommended_action": action_label,
            "confidence":         round(float(confidence), 4),
        }

    # ── Data preparation ──────────────────────────────────────────────────────

    def _build_matrices(
        self,
        raw_data:          Dict[str, pd.DataFrame],
        featured_data:     Dict[str, pd.DataFrame],
        ml_predictions:    Dict[str, Dict[str, dict]],
        fused_scores:      Dict[str, Dict[str, float]],
        crash_probability: float,
    ) -> Tuple:
        """Build (tickers, price_matrix, feature_matrix, market_features) arrays."""
        # Select tickers with sufficient data
        candidates = [
            t for t in featured_data
            if featured_data[t] is not None and len(featured_data[t]) >= 60
        ]
        if not candidates:
            return [], None, None, None

        tickers = candidates[: self.n_assets]

        # Align on common date index
        close_series = {}
        for t in tickers:
            df = raw_data.get(t)
            if df is None or "Close" not in df.columns:
                continue
            s = df["Close"].dropna()
            s.index = pd.to_datetime(s.index).normalize()
            s = s[~s.index.duplicated(keep="last")]
            close_series[t] = s

        tickers = [t for t in tickers if t in close_series]
        if not tickers:
            return [], None, None, None

        price_df = pd.DataFrame(close_series).sort_index().ffill().bfill().dropna()
        T, N     = price_df.shape
        tickers  = list(price_df.columns)

        # Feature matrix: T × N × F
        feat_cols = _STOCK_FEATURES
        F_dim     = len(feat_cols) + 2  # +ml_pred +fused_score
        fm        = np.zeros((T, N, F_dim), dtype=np.float32)

        for j, t in enumerate(tickers):
            fdf = featured_data.get(t)
            if fdf is None:
                continue
            fdf = fdf.copy()
            idx = pd.to_datetime(fdf.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            fdf.index = idx.normalize()
            fdf = fdf[~fdf.index.duplicated(keep="last")]
            fdf = fdf.reindex(price_df.index, method="ffill")

            for f_i, col in enumerate(feat_cols):
                if col in fdf.columns:
                    fm[:, j, f_i] = fdf[col].fillna(0.0).values

            # ml_pred_return (short horizon)
            ml_ret = ml_predictions.get(t, {}).get("short", {}).get("predicted_return", 0.0) or 0.0
            fm[:, j, len(feat_cols)]     = float(ml_ret)
            # fused_score
            fs = fused_scores.get(t, {}).get("short", 0.5) or 0.5
            fm[:, j, len(feat_cols) + 1] = float(fs)

        # Market features: T × 3
        mkt = np.zeros((T, 3), dtype=np.float32)
        first_t = tickers[0]
        fdf0 = featured_data.get(first_t)
        if fdf0 is not None:
            fdf0 = fdf0.copy()
            idx0 = pd.to_datetime(fdf0.index)
            if idx0.tz is not None:
                idx0 = idx0.tz_localize(None)
            fdf0.index = idx0.normalize()
            fdf0 = fdf0[~fdf0.index.duplicated(keep="last")].reindex(price_df.index, method="ffill")
            if "spy_return_21d" in fdf0.columns:
                mkt[:, 0] = fdf0["spy_return_21d"].fillna(0.0).values
            if "vix_level" in fdf0.columns:
                mkt[:, 1] = fdf0["vix_level"].fillna(0.20).values
        mkt[:, 2] = float(crash_probability)

        # Normalise feature matrix column-wise
        for j in range(N):
            for f_i in range(F_dim):
                col_data = fm[:, j, f_i]
                mu  = col_data.mean()
                std = col_data.std() + _EPS
                fm[:, j, f_i] = (col_data - mu) / std

        # Price matrix
        pm = price_df.values.astype(np.float32)

        return tickers, pm, fm, mkt

    def _build_latest_obs(
        self,
        tickers:           List[str],
        featured_data:     Dict[str, pd.DataFrame],
        ml_predictions:    Dict[str, Dict[str, dict]],
        fused_scores:      Dict[str, Dict[str, float]],
        crash_probability: float,
        current_weights:   Optional[Dict[str, float]],
    ) -> np.ndarray:
        N     = len(tickers)
        F_dim = self._n_features if self._n_features > 0 else len(_STOCK_FEATURES) + 2
        feat_cols = _STOCK_FEATURES

        feats = np.zeros(N * F_dim, dtype=np.float32)
        for j, t in enumerate(tickers):
            fdf = featured_data.get(t)
            if fdf is None or fdf.empty:
                continue
            for f_i, col in enumerate(feat_cols):
                if col in fdf.columns:
                    v = fdf[col].dropna()
                    feats[j * F_dim + f_i] = float(v.iloc[-1]) if not v.empty else 0.0
            ml_ret = ml_predictions.get(t, {}).get("short", {}).get("predicted_return", 0.0) or 0.0
            feats[j * F_dim + len(feat_cols)]     = float(ml_ret)
            fs = fused_scores.get(t, {}).get("short", 0.5) or 0.5
            feats[j * F_dim + len(feat_cols) + 1] = float(fs)

        # Market features
        spy_ret = vix_lvl = 0.0
        for df in featured_data.values():
            if df is None or df.empty:
                continue
            if "spy_return_21d" in df.columns:
                s = df["spy_return_21d"].dropna()
                if not s.empty:
                    spy_ret = float(s.iloc[-1])
            if "vix_level" in df.columns:
                s = df["vix_level"].dropna()
                if not s.empty:
                    vix_lvl = float(s.iloc[-1])
            break
        mkt = np.array([spy_ret, vix_lvl, crash_probability], dtype=np.float32)

        # Current weights
        cw = current_weights or {}
        wts = np.array(
            [float(cw.get(t, 1.0 / (N + 1))) for t in tickers]
            + [float(cw.get("cash", 1.0 / (N + 1)))],
            dtype=np.float32,
        )
        wts /= wts.sum() + _EPS

        obs = np.concatenate([feats, mkt, wts])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_model(self):
        if not _TORCH_AVAILABLE or self._net is None:
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            "state_dict":   self._net.state_dict(),
            "tickers":      self._tickers,
            "n_features":   self._n_features,
            "state_dim":    self._net.trunk[0].in_features,
            "n_actions":    self._net.actor_head.out_features,
            "hidden":       self.hidden,
        }, self.model_path)
        logger.info("RL model saved to %s", self.model_path)

    def _load_model(self) -> bool:
        if not _TORCH_AVAILABLE or not os.path.exists(self.model_path):
            return False
        try:
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
            sd   = ckpt["state_dim"]
            na   = ckpt["n_actions"]
            h    = ckpt.get("hidden", self.hidden)
            self._net = _ActorCritic(sd, na, h)
            self._net.load_state_dict(ckpt["state_dict"])
            self._net.eval()
            self._tickers   = ckpt.get("tickers", [])
            self._n_features = ckpt.get("n_features", len(_STOCK_FEATURES) + 2)
            self._device     = "cpu"
            logger.info("Loaded RL model from %s", self.model_path)
            return True
        except Exception as exc:
            logger.warning("Could not load RL model: %s", exc)
            return False

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _equal_weight_fallback(tickers: List[str], crash_prob: float) -> dict:
        n = len(tickers)
        if n == 0:
            return {"weights": {}, "cash_weight": 1.0, "recommended_action": "HOLD", "confidence": 0.5}
        # Reduce equity exposure proportionally to crash risk
        equity_frac = max(0.0, 1.0 - crash_prob)
        w_each      = equity_frac / n if n > 0 else 0.0
        return {
            "weights":            {t: round(w_each, 4) for t in tickers},
            "cash_weight":        round(1.0 - equity_frac, 4),
            "recommended_action": _crash_to_action(crash_prob),
            "confidence":         0.3,
        }


# ── Helper ────────────────────────────────────────────────────────────────────

def _crash_to_action(crash_prob: float) -> str:
    if crash_prob >= 0.75:
        return "STOP"
    if crash_prob >= 0.50:
        return "REDUCE"
    if crash_prob >= 0.25:
        return "HOLD"
    return "BUY"
