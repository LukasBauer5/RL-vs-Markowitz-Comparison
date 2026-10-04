from helpers import set_seed
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from portfolio_env import PortfolioEnv
from scipy.special import softmax
from helpers import apply_drift
import pandas as pd
import numpy as np

def train_rl_model(SEED, train_returns, train_rebal_dates, window_rl, transaction_cost, include_transaction_cost, decision_frequency,
                   time_steps, model_path):
    set_seed(SEED)

    vec_env = make_vec_env(
        PortfolioEnv,
        n_envs=8,
        seed=SEED,
        env_kwargs={"returns": train_returns, "rebal_dates": train_rebal_dates, "window": window_rl, "transaction_cost": transaction_cost,
                    "include_transaction_cost": include_transaction_cost, "decision_frequency": decision_frequency}
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        clip_range=0.15,
        learning_rate=5e-5,
        n_epochs=5,
        seed=SEED,
    )

    model.learn(total_timesteps=time_steps)

    model.save(model_path)

def run_rl(
        returns,
        test_returns,
        test_rebal_dates,
        window_rl,
        transaction_cost,
        include_transaction_cost,
        decision_frequency,
        rl_paths
):
    first_decision_idx = returns.index.get_loc(test_rebal_dates[0])

    rl_results = {}

    for label, path in rl_paths.items():
        loaded = PPO.load(path)

        env = PortfolioEnv(
            returns=returns,
            window=window_rl,
            rebal_dates=test_rebal_dates,
            transaction_cost=transaction_cost,
            start_idx=first_decision_idx,
            episode_length=len(test_returns),
            include_transaction_cost=include_transaction_cost,
            decision_frequency=decision_frequency,
        )

        obs, _ = env.reset()

        last_valid = (
            returns.index.get_loc(test_rebal_dates[-2])
            if env.decision_frequency == "monthly"
            else len(returns) - 1
        )

        daily_list, done = [], False
        turnover_list = []
        weights_list = []
        decision_dates = []

        while not done and env.t <= last_valid:
            action, _ = loaded.predict(obs, deterministic=True)
            weights = softmax(np.asarray(action)).astype(np.float32)

            decision_t = env.t
            weights_list.append(weights)
            decision_dates.append(returns.index[decision_t])

            steps_taken = env.steps_taken

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            turnover_list.append(info["portfolio turnover"])

            if env.decision_frequency == "daily":
                holding_period_returns = returns.iloc[decision_t: env.t]
            else:
                holding_period_returns = returns.iloc[decision_t + 1: env.t + 1]

            daily, _ = apply_drift(holding_period_returns, weights)

            if env.include_transaction_cost and steps_taken > 0:
                daily.iloc[0] -= info["portfolio turnover"] * env.transaction_cost

            daily_list.append(daily)

        if len(daily_list) > 0:
            rl_returns = pd.concat(daily_list)

            if env.decision_frequency == "daily":
                rl_returns = rl_returns.iloc[1:]

            rl_turnover = pd.Series(turnover_list, index=decision_dates, name=f"{label} turnover")
            rl_weights = pd.DataFrame(weights_list, index=decision_dates, columns=returns.columns)

            rl_results[label] = {
                "returns": rl_returns,
                "weights": rl_weights,
                "turnover": rl_turnover,
            }
    return rl_results