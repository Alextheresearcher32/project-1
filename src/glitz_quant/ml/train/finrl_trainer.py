"""
FinRL training scaffold. Trains an RL agent on historical OHLCV.

EXPLICIT WARNING: this produces a model file. Wiring its predictions into
live execution requires additional integration we have NOT enabled by default.
RL strategies are research artifacts here, not production traders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from glitz_quant.settings import ROOT_DIR, get_app_config
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


def train_ppo_on_candles(
    candles: pd.DataFrame,
    save_to: str | Path | None = None,
    total_timesteps: int | None = None,
) -> str:
    """
    Train a PPO agent on a single-symbol candle dataset.

    candles: DataFrame with columns [open, high, low, close, volume] and datetime index.
    Returns: filesystem path to the saved model zip.
    """
    # Imports are local so the rest of the system runs without torch/finrl installed.
    from finrl.config import INDICATORS  # type: ignore[import-not-found]
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # type: ignore[import-not-found]
    from stable_baselines3 import PPO  # type: ignore[import-not-found]
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore[import-not-found]

    cfg = get_app_config().get("ml", {}).get("training", {}).get("finrl", {})
    timesteps = int(total_timesteps or cfg.get("total_timesteps", 100_000))

    df = candles.copy()
    df["date"] = df.index
    df["tic"] = "BTC"
    df = df.reset_index(drop=True)
    df["day"] = (df["date"] - df["date"].min()).dt.days

    stock_dim = 1
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 10_000,
        "num_stock_shares": [0],
        "buy_cost_pct": [0.0015],
        "sell_cost_pct": [0.0015],
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
    }

    env = DummyVecEnv([lambda: StockTradingEnv(df=df, **env_kwargs)])
    model = PPO("MlpPolicy", env, verbose=0)
    log.info("finrl_training_started", timesteps=timesteps)
    model.learn(total_timesteps=timesteps)

    out_dir = Path(save_to or (ROOT_DIR / "data" / "models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppo_btc.zip"
    model.save(str(out_path))
    log.info("finrl_training_done", model_path=str(out_path))
    return str(out_path)


def load_and_predict(
    model_path: str | Path, observation: list[float]
) -> tuple[Any, dict[str, Any]]:
    """Load a saved PPO model and produce an action."""
    from stable_baselines3 import PPO  # type: ignore[import-not-found]

    model = PPO.load(str(model_path))
    action, _state = model.predict(observation, deterministic=True)
    return action, {"model": str(model_path)}
