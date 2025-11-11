import numpy as np
import pandas as pd
import bt

# Create some test data
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")

n_days = len(dates)
returns_a = np.random.normal(0.0005, 0.015, n_days)
returns_b = np.random.normal(0.0003, 0.012, n_days)

prices_a = 100 * np.cumprod(1 + returns_a)
prices_b = 100 * np.cumprod(1 + returns_b)

data = pd.DataFrame({
    "asset_a": prices_a,
    "asset_b": prices_b,
}, index=dates)

# Strategy 1
strategy1 = bt.Strategy(
    "equal_weight",
    [
        bt.algos.SelectAll(),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance()
    ]
)

# Strategy 2
strategy2 = bt.Strategy(
    "momentum",
    [
        bt.algos.SelectAll(),
        bt.algos.RunMonthly(run_on_first_date=True),
        bt.algos.SelectWhere(data > data.rolling(20).mean()),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance()
    ]
)

# Run backtest
bt1 = bt.Backtest(strategy1, data, progress_bar=False)
bt2 = bt.Backtest(strategy2, data, progress_bar=False)
results = bt.run(bt1, bt2)

# Calculate calmar ratio for strategy 1
prices1 = results.backtests["equal_weight"].strategy.prices
returns1 = prices1.pct_change().dropna()
calmar1 = bt.backtest.calmar_ratio(returns1, periods=252)

# Calculate calmar ratio for strategy 2
prices2 = results.backtests["momentum"].strategy.prices
returns2 = prices2.pct_change().dropna()
calmar2 = bt.backtest.calmar_ratio(returns2, periods=252)

print("Strategy Comparison:")
print(f"Equal Weight - Calmar Ratio: {calmar1:.2f}, Max Drawdown: {results.stats['equal_weight'].max_drawdown * 100:.2f}%")
print(f"Momentum - Calmar Ratio: {calmar2:.2f}, Max Drawdown: {results.stats['momentum'].max_drawdown * 100:.2f}%")
