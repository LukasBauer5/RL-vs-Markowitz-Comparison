# Reinforcement Learning for Portfolio Optimization

This repository contains the Python implementation developed as part of my bachelor's thesis, *Reinforcement Learning for Portfolio Optimization: A Comparison with the Markowitz Approach*.

The project compares a reinforcement learning-based approach using Proximal Policy Optimization (PPO) with classical Markowitz mean-variance optimization. Both methods are evaluated using historical stock market data, considering different rebalancing frequencies, transaction cost settings, and market periods.

An equally weighted portfolio serves as an additional benchmark.

## Methodology

The implementation includes:

- **Reinforcement Learning:** A custom portfolio management environment built with Gymnasium and a PPO agent trained using Stable-Baselines3.
- **Markowitz Optimization:** Mean-variance portfolio optimization using numerical optimization and shrinkage covariance estimation.
- **Benchmark:** An equally weighted portfolio for comparison.

The experiments consider:

- Daily and monthly portfolio rebalancing
- Portfolio management with and without transaction costs
- Two historical market periods, including one covering the global financial crisis
- Evaluation using multiple portfolio performance and risk metrics

## Technologies

Python, NumPy, pandas, SciPy, scikit-learn, Gymnasium, Stable-Baselines3, Matplotlib, and yfinance.

## Repository Structure

- `main.py` — Experiment configuration and execution
- `helpers.py` — Data preparation and utility functions
- `portfolio_env.py` — Custom reinforcement learning environment
- `reinforcement_learning.py` — PPO model training and evaluation
- `markowitz.py` — Markowitz portfolio optimization and equal-weight benchmark
- `metrics.py` — Portfolio performance evaluation
- `reporting.py` — Result reporting and visualization

## Running the Experiments

The experiment parameters can be configured in `main.py`, including the market period, rebalancing frequency, and transaction cost settings.

Historical market data is retrieved using yfinance.

To train new PPO models, set `train_models = True` in `main.py`. Otherwise, previously trained models must be available at the configured paths.

The experiments can then be executed using `python main.py` after installing the required Python dependencies.

## Bachelor's Thesis

[Read the full thesis](thesis.pdf)
