import numpy as np
import pandas as pd
import pytest

import bt


@pytest.fixture(
    scope="module",
    params=[(252, 10), (1000, 50)],
    ids=["1y-10-assets", "4y-50-assets"],
)
def prices(request):
    periods, assets = request.param
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.01, size=(periods, assets))
    return pd.DataFrame(
        100.0 * np.exp(returns.cumsum(axis=0)),
        index=pd.date_range("2010-01-01", periods=periods, freq="B"),
        columns=[f"asset_{index}" for index in range(assets)],
    )


@pytest.fixture(scope="module")
def fixed_income_data(prices):
    return {
        "bidoffer": prices * 0.001,
        "coupons": prices * 0.0001,
    }


def make_strategy(name):
    return bt.Strategy(
        name,
        [
            bt.algos.RunMonthly(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )


def run_equity_backtest(prices):
    backtest = bt.Backtest(make_strategy("equity"), prices)
    return bt.run(backtest, progress_bar=False)


def run_fixed_income_backtest(prices, additional_data):
    strategy = bt.FixedIncomeStrategy(
        "fixed-income",
        [
            bt.algos.RunMonthly(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
        children=[bt.CouponPayingSecurity(column) for column in prices.columns],
    )
    backtest = bt.Backtest(strategy, prices, additional_data=additional_data)
    return bt.run(backtest, progress_bar=False)


@pytest.fixture(scope="module")
def completed_strategy(prices):
    backtest = bt.Backtest(make_strategy("history"), prices)
    backtest.run()
    return backtest.strategy


@pytest.mark.benchmark(group="backtest")
def test_equity_backtest(benchmark, prices):
    result = benchmark(run_equity_backtest, prices)

    assert result.prices.shape[0] == prices.shape[0] + 1


@pytest.mark.benchmark(group="backtest")
def test_fixed_income_backtest(benchmark, prices, fixed_income_data):
    result = benchmark(run_fixed_income_backtest, prices, fixed_income_data)

    assert result.prices.shape[0] == prices.shape[0] + 1


@pytest.mark.benchmark(group="history")
def test_strategy_prices(benchmark, prices, completed_strategy):
    result = benchmark(getattr, completed_strategy, "prices")

    assert result.index[-1] == prices.index[-1]
