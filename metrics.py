import yfinance as yf
import numpy as np



def portfolio_values(returns, initial_portfolio_value=1000):
    return initial_portfolio_value * (1 + returns).cumprod()

def cumulative_returns(returns, initial_portfolio_value=1000):
    if len(returns) == 0:
        raise ValueError("Cumulative Return is undefined for an empty return series")
    values = portfolio_values(returns, initial_portfolio_value)
    return (values.iloc[-1] - initial_portfolio_value) / initial_portfolio_value

def deannualize(annual_rate, periods=252):
    return (1 + annual_rate) ** (1 / periods) - 1

def get_risk_free_rate(start, end):
    annualized = yf.download("^IRX", start=start, end=end, auto_adjust=False)["Close"] / 100
    daily = annualized.apply(deannualize).squeeze("columns")
    return daily

def maximum_drawdown(returns):
    wealth = (1 + returns).cumprod()
    dd = 1 - wealth / wealth.cummax()
    return dd.max()

def sharpe_ratio(returns, risk_free_rate, periods_per_year=252):
    rf_mean = risk_free_rate.mean()
    return np.sqrt(periods_per_year) * (returns.mean() - rf_mean) / returns.std()

def turnover_annualized(turnover, decision_frequency):
    if decision_frequency == "daily":
        return turnover.mean() * 252
    elif decision_frequency == "monthly":
        return turnover.mean() * 12
    else:
        raise ValueError("decision frequency must be daily or monthly")


def volatility_yearly(returns, periods_per_year=252):
    return np.sqrt(periods_per_year) * np.std(returns, ddof=1)

def compound_annual_growth_rate(returns, initial_portfolio_value=1000, periods_per_year=252):
    if len(returns) == 0:
        raise ValueError("CAGR is undefined for an empty return series")
    n = len(returns) / periods_per_year
    values = portfolio_values(returns, initial_portfolio_value)
    return (values.iloc[-1] / initial_portfolio_value) ** (1 / n) - 1

def calculate_metrics(returns, turnover, risk_free_rate, decision_frequency, initial_portfolio_value=1000, periods_per_year=252):
    return {
        "cumulative return": cumulative_returns(returns, initial_portfolio_value),
        "cagr": compound_annual_growth_rate(returns, initial_portfolio_value, periods_per_year),
        "volatility": volatility_yearly(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "max drawdown": maximum_drawdown(returns),
        "turnover": turnover_annualized(turnover, decision_frequency),
    }