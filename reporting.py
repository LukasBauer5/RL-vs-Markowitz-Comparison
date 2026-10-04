import matplotlib.pyplot as plt
from metrics import calculate_metrics
import pandas as pd

def report_results(
        rl_results,
        markowitz_results,
        equal_results,
        risk_free_rate,
        decision_frequency,
        include_transaction_cost,
        initial_portfolio_value = 1000,
        plot_name = None,
):
    metrics_results = {}

    for label, result in rl_results.items():
        metrics_results[label] = calculate_metrics(
            result["returns"],
            result["turnover"],
            risk_free_rate,
            decision_frequency,
            initial_portfolio_value=initial_portfolio_value,
        )

    for label, result in markowitz_results.items():
        metrics_results[label] = calculate_metrics(
            result["returns"],
            result["turnover"],
            risk_free_rate,
            decision_frequency,
            initial_portfolio_value=initial_portfolio_value,
        )

    metrics_results["Equal Weight"] = calculate_metrics(
        equal_results["returns"],
        equal_results["turnover"],
        risk_free_rate,
        decision_frequency,
        initial_portfolio_value=initial_portfolio_value,
    )

    metrics_df = pd.DataFrame(metrics_results).T
    if not include_transaction_cost:
        metrics_df = metrics_df.drop(columns=["turnover"])
    print(metrics_df.to_string())

    comparison_df = pd.DataFrame({
        **{
            label: initial_portfolio_value * (1 + result["returns"]).cumprod()
            for label, result in rl_results.items()
        },
        **{
            label: initial_portfolio_value * (1 + result["returns"]).cumprod()
            for label, result in markowitz_results.items()
        },
        "Equal Weight":
            initial_portfolio_value
            * (1 + equal_results["returns"]).cumprod(),
    })

    ax = comparison_df.plot(figsize=(14, 6))

    transaction_cost_text = (
        "with transaction costs"
        if include_transaction_cost
        else "without transaction costs"
    )

    ax.set_title(
        f"{decision_frequency.capitalize()} rebalancing {transaction_cost_text}",
        fontsize=20
    )
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel("Portfolio value", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=16)

    plt.tight_layout()

    if plot_name is not None:
        plt.savefig(plot_name)

    plt.show()

    return metrics_df, comparison_df