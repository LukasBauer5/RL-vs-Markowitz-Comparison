import gymnasium as gym
import numpy as np
from scipy.special import softmax

class PortfolioEnv(gym.Env):
    def __init__(self,
                 returns,
                 rebal_dates,
                 transaction_cost=0.001,
                 window=60,
                 start_idx=None,
                 episode_length=252,
                 include_transaction_cost=False,
                 decision_frequency="daily"):
        self.returns = returns
        self.returns_np = returns.to_numpy(dtype=np.float32)
        self.window = window
        self.rebal_dates = set([d for d in rebal_dates if d in self.returns.index])
        self.rebal_date_positions = set(
            returns.index.get_loc(d) for d in self.rebal_dates
        )
        self.num_assets = returns.shape[1]
        self.transaction_cost = transaction_cost
        self.episode_length = episode_length
        self.start_idx = start_idx
        self.include_transaction_cost = include_transaction_cost
        self.decision_frequency = decision_frequency
        self.turnover = []
        self.observation_space = gym.spaces.Dict(
            {
                "returns": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window, self.num_assets),
                                          dtype=np.float32),
                "weights": gym.spaces.Box(0, 1, shape=(self.num_assets,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(self.num_assets,), dtype=np.float32)
        self.rebalance_log = []
        self.value = None
        self.steps_taken = None
        self.weights = None
        self.t = None

    def _get_obs(self):
        if self.decision_frequency == "monthly":
            obs_returns = self.returns_np[self.t - self.window + 1: self.t + 1]
        else:
            obs_returns = self.returns_np[self.t - self.window: self.t]

        return {"returns": obs_returns,
                "weights": self.weights.astype(np.float32), }

    def _get_info(self, turnover=0.0):
        return {"portfolio value": self.value, "portfolio turnover": float(turnover)}

    def _find_next_rebal_date(self, current_t):
        t = current_t + 1
        while t < len(self.returns_np):
            if t in self.rebal_date_positions:
                return t
            t += 1
        return len(self.returns_np)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.value = 1
        if self.start_idx is None:
            max_start = len(self.returns) - self.episode_length - 1
            if self.decision_frequency == "daily":
                self.t = int(self.np_random.integers(self.window, max_start))
            else:
                t = int(self.np_random.integers(self.window, max_start))
                while t < max_start and self.returns.index[t] not in self.rebal_dates:
                    t += 1
                self.t = t
        else:
            self.t = self.start_idx

        self.turnover = []
        self.rebalance_log = []
        self.weights = np.ones(self.num_assets) / self.num_assets
        self.steps_taken = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        action = np.asarray(action)
        action = softmax(action)
        old_weights = self.weights.copy()
        turnover = 0.5 * np.sum(np.abs(action - old_weights)) if self.include_transaction_cost else 0
        cost = turnover * self.transaction_cost if self.include_transaction_cost and self.steps_taken > 0 else 0

        if self.decision_frequency == "daily":
            start_t = self.t
            end_t = min(self.t + 1, len(self.returns))
            next_decision_t = end_t
        elif self.decision_frequency == "monthly":
            decision_t = self.t
            next_rebal_t = self._find_next_rebal_date(decision_t)

            start_t = decision_t + 1
            end_t = min(next_rebal_t + 1, len(self.returns))

            next_decision_t = next_rebal_t
        else:
            raise ValueError("decision_frequency must be 'daily' or 'monthly'")

        start_date = self.returns.index[start_t] if start_t < len(self.returns) else None
        end_date = self.returns.index[end_t - 1] if end_t - 1 < len(self.returns) else None

        self.rebalance_log.append({
            "rebalance_number": self.steps_taken + 1,
            "start_date": start_date,
            "end_date": end_date,
            "days_held": end_t - start_t,
            "turnover": float(turnover),
        })

        reward = 0.0
        first_step = True
        t = start_t

        current_weights = action.copy()
        while t < end_t:
            new_returns = self.returns_np[t]
            portfolio_return = float(np.dot(current_weights, new_returns))

            if first_step:
                portfolio_return -= cost
                first_step = False

            reward += np.log(max(1e-8, 1 + portfolio_return))
            self.value *= (1 + portfolio_return)
            current_weights = current_weights * (1 + new_returns)
            current_weights_sum = current_weights.sum()

            if current_weights_sum > 0:
                current_weights = current_weights / current_weights_sum
            else:
                current_weights = np.ones(self.num_assets) / self.num_assets
            t += 1

        self.weights = current_weights.copy()
        self.turnover.append(turnover)
        self.steps_taken += 1

        self.t = next_decision_t
        terminated = self.t >= len(self.returns)
        truncated = self.steps_taken >= self.episode_length
        info = self._get_info(turnover=turnover)
        if terminated or truncated:
            observation = {"returns": np.zeros((self.window, self.num_assets), dtype=np.float32),
                           "weights": np.zeros((self.num_assets,), dtype=np.float32)}
        else:
            observation = self._get_obs()

        return observation, reward, terminated, truncated, info