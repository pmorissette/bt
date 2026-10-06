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


def run_close_dead_backtest(prices):
    strategy = bt.Strategy(
        "close-dead",
        [bt.algos.SelectAll(include_no_data=True), bt.algos.WeighEqually(), bt.algos.CloseDead(), bt.algos.Rebalance()],
        children=list(prices.columns),
    )
    backtest = bt.Backtest(strategy, prices, integer_positions=False)
    return bt.run(backtest, progress_bar=False)


def run_select_types_backtest(prices, child_count):
    strategy = bt.Strategy(
        "select-types",
        [bt.algos.SelectAll(), bt.algos.SelectTypes(), bt.algos.RunMonthly(), bt.algos.WeighEqually(), bt.algos.Rebalance()],
        children=[bt.Security(name) for name in prices.columns[:child_count]],
    )
    backtest = bt.Backtest(strategy, prices, integer_positions=False, progress_bar=False)
    backtest.run()
    return backtest


def run_fixed_income_backtest(prices, additional_data, funded=False):
    strategy = bt.FixedIncomeStrategy(
        "fixed-income",
        [
            bt.algos.RunMonthly(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            *([bt.algos.SetNotional("notional")] if funded else []),
            bt.algos.Rebalance(),
        ],
        children=[bt.CouponPayingSecurity(column) for column in prices.columns],
    )
    backtest = bt.Backtest(strategy, prices, additional_data=additional_data)
    return bt.run(backtest, progress_bar=False)


class ExistingLifecycleRegistries(bt.Algo):
    def __call__(self, target):
        # Model a seasoned universe with overlapping rolled/matured instruments.
        if "closed" not in target.perm:
            labels = list(target.universe.columns)
            target.perm["closed"] = set(labels[:len(labels) // 2])
            target.perm["rolled"] = set(labels[len(labels) // 4:3 * len(labels) // 4])
        return True


def run_select_active_backtest(data, ranks, selected_count=None):
    first_active = 3 * len(data.columns) // 4
    selector = bt.algos.SelectAll() if selected_count is None else bt.algos.SelectThese(list(data.columns[first_active:first_active + selected_count]))
    strategy = bt.Strategy("active", [
        ExistingLifecycleRegistries(), selector, bt.algos.SelectActive(),
        bt.algos.SetStat("ranks"), bt.algos.SelectN(10, sort_descending=False, filter_selected=True),
        bt.algos.WeighEqually(), bt.algos.Rebalance(),
    ])
    backtest = bt.Backtest(strategy, data, additional_data={"ranks": ranks}, integer_positions=False, progress_bar=False)
    backtest.run()
    return backtest


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
def test_close_dead_backtest(benchmark):
    # A wide, constant-price universe exercises lookups with an independent holdings oracle.
    data = pd.DataFrame(100.0, index=pd.bdate_range("2010-01-01", periods=252), columns=[f"asset_{i}" for i in range(500)])
    result = benchmark(run_close_dead_backtest, data)
    strategy = result.backtests["close-dead"].strategy

    assert strategy.values.iloc[1:].eq(1_000_000.0).all()
    assert all(child.position == 20.0 for child in strategy.children.values())


@pytest.mark.benchmark(group="backtest")
@pytest.mark.parametrize("child_count", [1, 10, 32, 1000])
def test_select_types_backtest(benchmark, child_count):
    # Daily type filtering followed by monthly allocation exercises a wide typed universe.
    data = pd.DataFrame(100.0, index=pd.bdate_range("2010-01-01", periods=252), columns=[f"asset_{i}" for i in range(1000)])
    strategy = benchmark(run_select_types_backtest, data, child_count).strategy

    assert strategy.values.iloc[1:].eq(1_000_000.0).all()
    assert len(strategy.children) == child_count
    assert all(np.isclose(child.position, 10000.0 / child_count) for child in strategy.children.values())
    assert strategy.temp["selected"] == list(data.columns[:child_count])


@pytest.mark.benchmark(group="backtest")
@pytest.mark.parametrize("selected_count", [1, 10, 31, 32, None], ids=["one", "ten", "below-cutoff", "at-cutoff", "whole-universe"])
def test_select_active_backtest(benchmark, selected_count):
    # Rank daily within the eligible universe, then fund its ten lowest-ranked assets.
    labels = [f"asset_{i}" for i in range(2000)]
    data = pd.DataFrame(100.0, index=pd.bdate_range("2010-01-01", periods=252), columns=labels)
    ranks = pd.DataFrame(np.broadcast_to(np.arange(len(labels)), data.shape), index=data.index, columns=labels)
    # Keep registry preparation fixed while varying the number of membership checks.
    backtest = benchmark(run_select_active_backtest, data, ranks, selected_count)
    strategy = backtest.strategy

    # Constant prices and equal weights reconstruct holdings independently of selection.
    held_count = min(10, selected_count) if selected_count is not None else 10
    assert strategy.temp["selected"] == labels[1500:1500 + held_count]
    assert set(strategy.children) == set(labels[1500:1500 + held_count])
    assert all(np.isclose(child.position, 10000.0 / held_count) for child in strategy.children.values())
    assert strategy.values.iloc[1:].eq(1_000_000.0).all()
    assert strategy.perm == {"closed": set(labels[:1000]), "rolled": set(labels[500:1500])}


@pytest.mark.benchmark(group="backtest")
def test_fixed_income_backtest(benchmark, prices, fixed_income_data):
    result = benchmark(run_fixed_income_backtest, prices, fixed_income_data)

    assert result.prices.shape[0] == prices.shape[0] + 1


@pytest.mark.benchmark(group="backtest")
def test_funded_fixed_income_backtest(benchmark, prices, fixed_income_data):
    additional_data = {**fixed_income_data, "notional": pd.Series(10000.0, index=prices.index)}
    result = benchmark(run_fixed_income_backtest, prices, additional_data, funded=True)

    strategy = result.backtests["fixed-income"].strategy
    assert strategy.notional_value > 0
    assert sum(security.coupons.sum() for security in strategy.securities) > 0


@pytest.mark.benchmark(group="history")
def test_strategy_prices(benchmark, prices, completed_strategy):
    result = benchmark(getattr, completed_strategy, "prices")

    assert result.index[-1] == prices.index[-1]


@pytest.mark.benchmark(group="history")
def test_strategy_data(benchmark, prices, completed_strategy):
    result = benchmark(getattr, completed_strategy, "data")

    assert result.index[-1] == prices.index[-1]
