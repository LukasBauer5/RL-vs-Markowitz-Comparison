from helpers import prepare_data
from reinforcement_learning import train_rl_model
from reinforcement_learning import run_rl
from markowitz import run_markowitz
from metrics import get_risk_free_rate
from reporting import report_results



def main():
    decision_frequency = "monthly"
    include_transaction_cost = True
    transaction_cost = 0.001
    initial_portfolio_value = 1000
    period = "recent"

    train_models = False
    time_steps = 500000

    rl_configs = {
        "RL model 1": {
            "seed": 1,
            "path": "models/ppo_portfolio_seed1_500k_monthly_tc",
        },
        "RL model 2": {
            "seed": 2,
            "path": "models/ppo_portfolio_seed2_500k_monthly_tc",
        },
        "RL model 3": {
            "seed": 3,
            "path": "models/ppo_portfolio_seed3_500k_monthly_tc",
        },
    }

    markowitz_configs = {
        "Markowitz 15%": {
            "lambda": 10000,
            "window": 60,
        },
        "Markowitz 20%": {
            "lambda": 26.1,
            "window": 60,
        },
        "Markowitz 30%": {
            "lambda": 2.1,
            "window": 252,
        },
    }

    max_markowitz_window = max(
        config["window"]
        for config in markowitz_configs.values()
    )

    (
        train_returns,
        test_returns,
        window_rl,
        returns,
        test_rebal_dates,
        train_rebal_dates,
    ) = prepare_data(period, max_markowitz_window)

    if train_models:
        for config in rl_configs.values():
            train_rl_model(
                config["seed"],
                train_returns,
                train_rebal_dates,
                window_rl,
                transaction_cost,
                include_transaction_cost,
                decision_frequency,
                time_steps,
                config["path"],
            )

    rl_paths = {
        label: config["path"]
        for label, config in rl_configs.items()
    }

    rl_results = run_rl(
        returns,
        test_returns,
        test_rebal_dates,
        window_rl,
        transaction_cost,
        include_transaction_cost,
        decision_frequency,
        rl_paths,
    )

    markowitz_results, equal_results = run_markowitz(
        returns,
        test_returns,
        test_rebal_dates,
        decision_frequency,
        transaction_cost,
        include_transaction_cost,
        markowitz_configs,
    )

    risk_free_rate = get_risk_free_rate(
        test_returns.index.min(),
        test_returns.index.max(),
    )

    metrics_df, comparison_df = report_results(
        rl_results,
        markowitz_results,
        equal_results,
        risk_free_rate,
        decision_frequency,
        include_transaction_cost,
        initial_portfolio_value=initial_portfolio_value,
        plot_name="portfolio_comparison.png",
    )

    return metrics_df, comparison_df


if __name__ == "__main__":
    main()
