import numpy as np
import pandas as pd
import random
import torch
import yfinance as yf

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def prepare_data(period, max_markowitz_window):
    assets_recent = [
        'MSFT', 'AAPL', 'AMZN', 'GOOG', 'UNH', 'V',
        'WMT', 'PG', 'KO', 'JNJ', 'PFE', 'MRK',
        'XOM', 'CVX', 'JPM', 'BAC', 'CAT', 'HON',
        'IBM', 'INTC', 'CSCO', 'MMM',
        'BA', 'DIS', 'VZ', 'T', 'CVS', 'F'
    ]

    assets_crisis = [
        "MSFT", "AAPL", "AMZN", "IBM", "INTC", "CSCO",
        "WMT", "PG", "KO", "PEP", "JNJ", "PFE", "MRK", "MCD", "CL",
        "COST", "HD", "DIS",
        "XOM", "CVX",
        "JPM", "BAC",
        "CAT", "HON", "MMM", "BA", "GE",
        "F", "VZ",
    ]

    if period == "recent":
        start="2011-01-01"
        end="2026-04-01"
        split_date="2019-01-01"
        last_train_date="2018-12-31"
        selected_assets = assets_recent
    elif period == "crisis":
        start="1998-01-01"
        end="2015-01-01"
        split_date="2007-01-01"
        last_train_date="2006-12-31"
        selected_assets = assets_crisis
    else:
        raise ValueError("Period must be 'recent' or 'crisis'")

    pf_data = yf.download(
        selected_assets,
        start=start,
        end=end,
        auto_adjust=True
    )["Close"]

    pf_data = pf_data.dropna(axis=1)
    print("Final asset count:", pf_data.shape[1])
    print("Remaining assets:", list(pf_data.columns))
    returns = pf_data.pct_change().dropna()
    train_returns = returns.loc[:last_train_date]
    test_returns = returns.loc[split_date:]
    window_rl = 60
    window_size = max(max_markowitz_window, window_rl)
    rebal_dates = returns.groupby([returns.index.year, returns.index.month]).tail(1).index
    test_rebal_dates = [d for d in rebal_dates if
                        returns.index.get_loc(d) >= window_size - 1 and d in test_returns.index]
    train_rebal_dates = [d for d in rebal_dates if
                         returns.index.get_loc(d) >= window_size - 1 and d in train_returns.index]

    return (
        train_returns,
        test_returns,
        window_rl,
        returns,
        test_rebal_dates,
        train_rebal_dates
    )

def apply_drift(holding_period_returns, initial_weights):
    weights = initial_weights.copy()
    daily_returns = []

    for _, asset_returns in holding_period_returns.iterrows():
        asset_returns = asset_returns.to_numpy()

        portfolio_return = float(weights @ asset_returns)
        daily_returns.append(portfolio_return)

        weights = (1 + asset_returns) * weights
        weights = weights / weights.sum()

    return pd.Series(daily_returns, index=holding_period_returns.index), weights