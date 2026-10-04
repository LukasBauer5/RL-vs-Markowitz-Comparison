import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from helpers import apply_drift
import pandas as pd

def run_markowitz(
        returns,
        test_returns,
        test_rebal_dates,
        decision_frequency,
        transaction_cost,
        include_transaction_cost,
        markowitz_configs,
):
    n_assets = returns.shape[1]
    if decision_frequency == "daily":
        start_date = test_returns.index[test_returns.index.get_loc(test_rebal_dates[0])]

        decision_dates = test_returns.loc[start_date:].index.tolist()
    elif decision_frequency == "monthly":
        decision_dates = test_rebal_dates
    else:
        raise ValueError("decision_frequency must be daily or monthly")

    markowitz_results = {}
    equal_weight_results = None

    for label, config in markowitz_configs.items():
        lam = config["lambda"]
        window = config["window"]

        portfolio_returns_list = []
        equal_portfolio_returns_list = []
        turnover_markowitz_list = []
        turnover_equal_list = []
        weights_list = []

        old_weights_markowitz = np.ones(n_assets) / n_assets
        old_weights_equal = np.ones(n_assets) / n_assets

        for i in range(len(decision_dates) - 1):
            date = decision_dates[i]
            next_date = decision_dates[i + 1]

            t = returns.index.get_loc(date)
            historical_window = returns.iloc[t - window + 1: t + 1]
            mu = historical_window.mean()
            sigma = LedoitWolf().fit(historical_window).covariance_

            t_test = test_returns.index.get_loc(date)
            t_next_test = test_returns.index.get_loc(next_date)
            holding_period_returns = test_returns.iloc[t_test + 1: t_next_test + 1]
            holding_days = len(holding_period_returns)

            mu_h = mu * holding_days
            sigma_h = sigma * holding_days
            n = len(mu)

            old_weights = old_weights_markowitz.copy()

            def objective(w, mu_h, sigma_h, lam, old_weights):
                expected_return = mu_h @ w
                risk_penalty = 0.5 * lam * (w @ sigma_h @ w)
                turnover_cost = transaction_cost * 0.5 * np.sum(np.abs(w - old_weights)) if include_transaction_cost and i > 0 else 0
                return -expected_return + risk_penalty + turnover_cost

            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0,1)] * n
            result = minimize(
                objective,
                np.ones(n) / n,
                args=(mu_h, sigma_h, lam, old_weights),
                constraints=constraints,
                bounds=bounds)

            if not result.success:
                raise RuntimeError(
                    f"Optimization failed on {date} for {label}: {result.message}"
                )

            w = result.x
            weights_list.append(w)
            equal_weights = np.ones(n) / n

            turnover_markowitz = 0.5 * np.sum(np.abs(w - old_weights_markowitz)) if include_transaction_cost and i > 0 else 0
            turnover_equal = 0.5 * np.sum(np.abs(equal_weights - old_weights_equal)) if include_transaction_cost and i > 0 else 0

            turnover_markowitz_list.append(turnover_markowitz)
            turnover_equal_list.append(turnover_equal)

            realized_portfolio_returns_markowitz, drifted_weights = apply_drift(holding_period_returns, w)
            realized_portfolio_returns_markowitz.iloc[0] -= turnover_markowitz * transaction_cost

            realized_portfolio_returns_equal, drifted_equal_weight = apply_drift(holding_period_returns, equal_weights)
            realized_portfolio_returns_equal.iloc[0] -= turnover_equal * transaction_cost

            portfolio_returns_list.append(realized_portfolio_returns_markowitz)
            equal_portfolio_returns_list.append(realized_portfolio_returns_equal)

            old_weights_markowitz = drifted_weights.copy()
            old_weights_equal = drifted_equal_weight.copy()

        markowitz_turnover = pd.Series(turnover_markowitz_list, index=decision_dates[:-1], name=f"{label} turnover")
        markowitz_weights = pd.DataFrame(
            weights_list,
            index=decision_dates[:-1],
            columns=returns.columns
        )
        markowitz_results[label] = {
            "returns": pd.concat(portfolio_returns_list),
            "weights": markowitz_weights,
            "turnover": markowitz_turnover}

        if equal_weight_results is None:
            equal_weight_results = {
                "returns": pd.concat(equal_portfolio_returns_list),
                "turnover": pd.Series(turnover_equal_list, index=decision_dates[:-1], name="Equal weight turnover")}

    return markowitz_results, equal_weight_results