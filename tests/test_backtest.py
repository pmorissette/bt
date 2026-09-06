from __future__ import division

import numpy as np
import pandas as pd
import pytest
import random

from unittest import mock

import bt


def test_backtest_copies_strategy():
    s = mock.MagicMock()
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )

    actual = bt.Backtest(s, data, progress_bar=False)

    assert id(s) != id(actual.strategy)


def test_backtest_dates_set():
    s = mock.MagicMock()
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )

    actual = bt.Backtest(s, data, progress_bar=False)

    # must account for 't0' addition
    assert len(actual.dates) == len(data.index) + 1
    assert actual.dates[1] == data.index[0]
    assert actual.dates[-1] == data.index[-1]


def test_backtest_rejects_unsorted_dates():
    s = mock.MagicMock()
    data = pd.DataFrame(
        index=pd.to_datetime(["2010-01-03", "2010-01-01", "2010-01-04"]),
        columns=["a", "b"],
        data=100,
    )

    with pytest.raises(ValueError, match="monotonic increasing"):
        bt.Backtest(s, data, progress_bar=False)


def test_backtest_auto_name():
    s = mock.MagicMock()
    s.name = "s"
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )

    actual = bt.Backtest(s, data, progress_bar=False)

    assert actual.name == "s"


def test_initial_capital_set():
    s = mock.MagicMock()
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )

    actual = bt.Backtest(s, data, initial_capital=302, progress_bar=False)
    actual.run()

    s = actual.strategy

    s.adjust.assert_called_with(302)


def test_run_loop():
    s = mock.MagicMock()
    # run loop checks on this
    s.bankrupt = False
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )

    actual = bt.Backtest(s, data, initial_capital=302, progress_bar=False)
    actual.run()

    s = actual.strategy

    # account for first update call to 'setup' initial state
    assert s.update.call_count == 10 + 1
    assert s.run.call_count == 5


def test_turnover():
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["a", "b"], data=100)

    data.loc[dts[1], "a"] = 105
    data.loc[dts[1], "b"] = 95

    data.loc[dts[2], "a"] = 110
    data.loc[dts[2], "b"] = 90

    data.loc[dts[3], "a"] = 115
    data.loc[dts[3], "b"] = 85

    s = bt.Strategy(
        "s", [bt.algos.SelectAll(), bt.algos.WeighEqually(), bt.algos.Rebalance()]
    )

    t = bt.Backtest(s, data, commissions=lambda x, y: 0, progress_bar=False)
    res = bt.run(t)

    t = res.backtests["s"]

    # these numbers were (tediously) calculated in excel
    assert np.allclose(t.turnover[dts[0]], 0.0 / 1000000)
    assert np.allclose(t.turnover[dts[1]], 24985.0 / 1000000)
    assert np.allclose(t.turnover[dts[2]], 24970.0 / 997490)
    assert np.allclose(t.turnover[dts[3]], 25160.0 / 992455)
    assert np.allclose(t.turnover[dts[4]], 76100.0 / 1015285)


def test_can_disable_progress_bar_from_run():
    from contextlib import redirect_stderr
    from io import StringIO

    # Create an in-memory buffer
    output_capture = StringIO()

    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )
    s = bt.Strategy("test", [
        bt.algos.SelectAll(),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance()
    ])

    b = bt.Backtest(s, data)

    # Redirect stderr to the buffer
    with redirect_stderr(output_capture):
        result = bt.run(b, progress_bar=False)

    # confirm that the output is empty
    assert output_capture.getvalue() is ""
    # confirm that we actually ran something
    assert  len(result.get_transactions()) > 0


def test_Results_helper_functions():

    names = ["foo", "bar"]
    dates = pd.date_range(
        start="2017-01-01", end="2017-12-31", freq=pd.tseries.offsets.BDay()
    )
    n = len(dates)
    rdf = pd.DataFrame(np.zeros((n, len(names))), index=dates, columns=names)

    np.random.seed(1)
    rdf[names[0]] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)
    rdf[names[1]] = np.random.normal(loc=0.04 / n, scale=0.05 / np.sqrt(n), size=n)

    pdf = 100 * np.cumprod(1 + rdf)

    # algo to fire on the beginning of every month and to run on the first date
    runDailyAlgo = bt.algos.RunDaily(run_on_first_date=True)

    # algo to set the weights
    #  it will only run when runMonthlyAlgo returns true
    #  which only happens on the first of every month
    weights = pd.Series([0.6, 0.4], index=rdf.columns)
    weighSpecifiedAlgo = bt.algos.WeighSpecified(**weights)

    # algo to rebalance the current weights to weights set by weighSpecified
    #  will only run when weighSpecifiedAlgo returns true
    #  which happens every time it runs
    rebalAlgo = bt.algos.Rebalance()

    # a strategy that rebalances monthly to specified weights
    strat = bt.Strategy("static", [runDailyAlgo, weighSpecifiedAlgo, rebalAlgo])

    backtest = bt.Backtest(strat, pdf, integer_positions=False, progress_bar=False)

    res = bt.run(backtest)

    assert type(res.get_security_weights()) is pd.DataFrame

    assert type(res.get_transactions()) is pd.DataFrame

    assert type(res.get_weights()) is pd.DataFrame


def test_Results_helper_functions_fi():

    names = ["foo", "bar"]
    dates = pd.date_range(
        start="2017-01-01", end="2017-12-31", freq=pd.tseries.offsets.BDay()
    )
    n = len(dates)
    rdf = pd.DataFrame(np.zeros((n, len(names))), index=dates, columns=names)

    np.random.seed(1)
    rdf[names[0]] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)
    rdf[names[1]] = np.random.normal(loc=0.04 / n, scale=0.05 / np.sqrt(n), size=n)

    pdf = 100 * np.cumprod(1 + rdf)
    notional = pd.Series(1e6, index=pdf.index)

    # algo to fire on the beginning of every month and to run on the first date
    runDailyAlgo = bt.algos.RunDaily(run_on_first_date=True)

    # algo to select all securities
    selectAll = bt.algos.SelectAll()

    # algo to set the weights
    #  it will only run when runMonthlyAlgo returns true
    #  which only happens on the first of every month
    weighRandomly = bt.algos.WeighRandomly()

    # algo to set the notional of the fixed income strategy
    setNotional = bt.algos.SetNotional("notional")

    # algo to rebalance the current weights to weights set by weighSpecified
    #  will only run when weighSpecifiedAlgo returns true
    #  which happens every time it runs
    rebalAlgo = bt.algos.Rebalance()

    # a strategy that rebalances monthly to specified weights
    strat = bt.FixedIncomeStrategy(
        "random", [runDailyAlgo, selectAll, weighRandomly, setNotional, rebalAlgo]
    )

    backtest = bt.Backtest(
        strat,
        pdf,
        initial_capital=0,
        integer_positions=False,
        progress_bar=False,
        additional_data={"mydata": pdf, "notional": notional},
    )
    bidoffer = 1.0
    backtest2 = bt.Backtest(
        strat,
        pdf,
        initial_capital=0,
        integer_positions=False,
        progress_bar=False,
        additional_data={
            "mydata": pdf,
            "notional": notional,
            "bidoffer": pd.DataFrame(bidoffer, pdf.index, pdf.columns),
        },
    )
    random.seed(1234)
    res = bt.run(backtest)
    random.seed(1234)
    res2 = bt.run(backtest2)

    assert type(res.get_security_weights()) is pd.DataFrame

    assert type(res.get_transactions()) is pd.DataFrame
    assert len(res.get_transactions()) > 0

    assert type(res.get_weights()) is pd.DataFrame

    # Make sure the insertion of the first row applies to additional data as well
    assert backtest.data.index.equals(backtest.additional_data["mydata"].index)

    # Check that bid/offer is accounted for
    transactions = res.get_transactions()
    transactions["price"] = transactions["price"] + 0.5 * bidoffer
    assert (
        res2.get_transactions().price - res2.get_transactions().price
    ).abs().sum() == 0


def test_nested_strategy_backtest_handles_initial_paper_trade_value():
    names = ["foo", "bar", "rf"]
    dates = pd.date_range("2017-01-01", "2017-12-31", freq=pd.tseries.offsets.BDay())
    n = len(dates)
    returns = pd.DataFrame(np.zeros((n, len(names))), index=dates, columns=names)

    np.random.seed(1)
    returns["foo"] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)
    returns["bar"] = np.random.normal(loc=0.04 / n, scale=0.05 / np.sqrt(n), size=n)
    returns["rf"] = 0.0

    prices = 100 * np.cumprod(1 + returns)

    weights = pd.Series([0.6, 0.2, 0.1], index=returns.columns)
    leaf = bt.Strategy(
        "leaf",
        [
            bt.algos.RunMonthly(run_on_first_date=True),
            bt.algos.WeighSpecified(**weights),
            bt.algos.Rebalance(),
        ],
    )
    middle = bt.Strategy(
        "middle",
        [
            bt.algos.RunMonthly(run_on_first_date=True),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
        children=[leaf, "foo"],
    )
    root = bt.Strategy(
        "root",
        [
            bt.algos.RunMonthly(run_on_first_date=True),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
        children=[middle, "bar"],
    )

    backtest = bt.Backtest(root, prices, integer_positions=False, progress_bar=False)

    result = bt.run(backtest)

    assert isinstance(result.prices, pd.DataFrame)
    assert np.isfinite(result.prices["root"]).all()
    assert result.prices["root"].iloc[0] == 100


def test_run_after_date_stats_include_first_transaction():
    dates = pd.date_range("2000-01-01", "2002-12-31", freq=pd.tseries.offsets.BDay())
    prices = pd.DataFrame(index=dates, data={"a": 100.0})
    prices.loc[dates[260]:, "a"] = np.linspace(100, 150, len(dates[260:]))

    strategy = bt.Strategy(
        "delayed",
        [
            bt.algos.RunAfterDate("2001-01-01"),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )

    result = bt.run(
        bt.Backtest(
            strategy,
            prices,
            commissions=lambda q, p: 100.0,
            progress_bar=False,
        )
    )

    first_transaction_date = result.get_transactions().index.get_level_values(0)[0]
    first_transaction_position = result.backtests["delayed"].strategy.prices.index.get_loc(first_transaction_date)
    stats_start = result.backtests["delayed"].strategy.prices.index[first_transaction_position - 1]
    assert result.stats["delayed"].start == stats_start
    assert result.prices.index[0] == stats_start
    assert result.prices.loc[first_transaction_date, "delayed"] < result.prices.loc[stats_start, "delayed"]


def test_30_min_data():
    names = ["foo"]
    dates = pd.date_range(start="2017-01-01", end="2017-12-31", freq="30min")
    n = len(dates)
    rdf = pd.DataFrame(np.zeros((n, len(names))), index=dates, columns=names)

    np.random.seed(1)
    rdf[names[0]] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)

    pdf = 100 * np.cumprod(1 + rdf)

    sma50 = pdf.rolling(50).mean()
    sma200 = pdf.rolling(200).mean()

    tw = sma200.copy()
    tw[sma50 > sma200] = 1.0
    tw[sma50 <= sma200] = -1.0
    tw[sma200.isnull()] = 0.0

    ma_cross = bt.Strategy("ma_cross", [bt.algos.WeighTarget(tw), bt.algos.Rebalance()])
    t = bt.Backtest(ma_cross, pdf, progress_bar=False)
    res = bt.run(t)

    wait = 1


def test_RenomalizedFixedIncomeResult():
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["a"], data=1.0)
    data.loc[dts[0], "a"] = 0.99
    data.loc[dts[1], "a"] = 1.01
    data.loc[dts[2], "a"] = 0.99
    data.loc[dts[3], "a"] = 1.01
    data.loc[dts[4], "a"] = 0.99

    weights = pd.DataFrame(index=dts, columns=["a"], data=1.0)
    weights.loc[dts[0], "a"] = 1.0
    weights.loc[dts[1], "a"] = 2.0
    weights.loc[dts[2], "a"] = 1.0
    weights.loc[dts[3], "a"] = 2.0
    weights.loc[dts[4], "a"] = 1.0

    coupons = pd.DataFrame(index=dts, columns=["a"], data=0.0)

    algos = [
        bt.algos.SelectAll(),
        bt.algos.WeighTarget(weights),
        bt.algos.SetNotional("notional"),
        bt.algos.Rebalance(),
    ]
    children = [bt.CouponPayingSecurity("a")]

    s = bt.FixedIncomeStrategy("s", algos, children=children)

    t = bt.Backtest(
        s,
        data,
        initial_capital=0,
        additional_data={"notional": pd.Series(1e6, dts), "coupons": coupons},
        progress_bar=False,
    )
    res = bt.run(t)

    t = res.backtests["s"]

    # Due to the relationship between the time varying notional and the prices,
    # the strategy has lost money, but price == 100, so "total return" is zero
    assert t.strategy.value < 0.0
    assert t.strategy.price == pytest.approx(100.0)
    assert res.stats["s"].total_return == 0

    # Renormalizing results to a constant size "fixes" this
    norm_res = bt.backtest.RenormalizedFixedIncomeResult(1e6, *res.backtest_list)
    assert norm_res.stats["s"].total_return == pytest.approx(t.strategy.value / 1e6, 16)

    # Check that using the lagged notional value series leads to the same results
    # as the original calculation. This proves that we can re-derive the price
    # series from the other data available on the strategy
    notl_values = t.strategy.notional_values.shift(1)
    notl_values[dts[0]] = 1e6  # The notional value *before* any trades are put on
    norm_res = bt.backtest.RenormalizedFixedIncomeResult(
        notl_values, *res.backtest_list
    )

    assert norm_res.stats["s"].total_return == res.stats["s"].total_return
    assert norm_res.prices.equals(res.prices)


@pytest.mark.parametrize("as_series", [False, True], ids=["dataframe", "series"])
def test_additional_data_auxiliary_bootstrap_boolean_dtype_no_warning(as_series):
    """Test that the bootstrap row stays missing without a bool concat warning."""
    import warnings

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["a", "b"], data=100.0)

    # Exercise NumPy bool while retaining the warning regression covered by this test.
    signal = pd.Series([True, False, True, False, True], index=dts, name="signal")
    if not as_series:
        signal = signal.to_frame()

    s = bt.Strategy(
        "test", [bt.algos.SelectAll(), bt.algos.WeighEqually(), bt.algos.Rebalance()]
    )

    # Require both the missing-row contract and warning-free pandas concatenation.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = bt.Backtest(s, data, additional_data={"signal": signal}, progress_bar=False)
        t.run()

        processed = t.additional_data["signal"]
        assert np.asarray(pd.isna(processed.iloc[0])).all()
        if as_series:
            pd.testing.assert_series_equal(processed.iloc[1:], signal, check_dtype=False, check_freq=False)
        else:
            pd.testing.assert_frame_equal(processed.iloc[1:], signal, check_dtype=False, check_freq=False)

        future_warnings = [warning for warning in w
                          if issubclass(warning.category, FutureWarning)
                          and "bool-dtype" in str(warning.message).lower()]
        assert len(future_warnings) == 0


@pytest.mark.parametrize(
    ("dtype", "values", "expected_dtype"),
    [
        ("int64", [1, 0, 1, 0, 1], "float64"),
        ("Int64", [1, 0, 1, 0, 1], "Int64"),
        ("boolean", [True, False, True, False, True], "boolean"),
        ("float64", [1, 0, 1, 0, 1], "float64"),
    ],
)
def test_additional_data_auxiliary_bootstrap_dtypes(dtype, values, expected_dtype):
    """Preserve dated auxiliary values while making the bootstrap row missing."""
    dates = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(100.0, index=dates, columns=["a"])
    auxiliary = pd.DataFrame({"value": pd.Series(values, index=dates, dtype=dtype)})
    strategy = bt.Strategy("test", [])

    backtest = bt.Backtest(
        strategy,
        data,
        additional_data={"auxiliary": auxiliary},
        progress_bar=False,
    )
    processed = backtest.additional_data["auxiliary"]

    # The synthetic row may widen dtype, but it must not alter dated observations.
    assert processed.index[0] == dates[0] - pd.DateOffset(days=1)
    assert processed.iloc[0].isna().all()
    assert str(processed.dtypes["value"]) == expected_dtype
    pd.testing.assert_frame_equal(processed.iloc[1:], auxiliary, check_dtype=False, check_freq=False)


def _impact_universe(n_periods=60, n_securities=3, seed=0):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    cols = [chr(ord("A") + i) for i in range(n_securities)]
    prices = pd.DataFrame(
        100 + np.random.randn(n_periods, n_securities).cumsum(axis=0),
        index=dates,
        columns=cols,
    )
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=cols)
    volatility = pd.DataFrame(0.02, index=dates, columns=cols)
    return prices, volume, volatility


def _ew_strategy(name="ew"):
    return bt.Strategy(
        name,
        [
            bt.algos.RunQuarterly(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )


@pytest.mark.parametrize("cost_model_type", [bt.SqrtCostModel, bt.AlmgrenChrissCostModel])
def test_backtest_integer_volume_matches_float_volume(cost_model_type):
    """Treat integer and equivalent float volume identically in cost models."""
    prices, float_volume, volatility = _impact_universe()
    integer_volume = float_volume.astype("int64")

    # Use float volume as the numerical reference for each nonlinear cost model.
    float_backtest = bt.Backtest(
        _ew_strategy(),
        prices,
        name="float_volume",
        commissions=cost_model_type(),
        volume=float_volume,
        volatility=volatility,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )
    integer_backtest = bt.Backtest(
        _ew_strategy(),
        prices,
        name="integer_volume",
        commissions=cost_model_type(),
        volume=integer_volume,
        volatility=volatility,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )

    float_backtest.run()
    integer_backtest.run()

    # Alignment changes representation only; observed volume must remain unchanged.
    aligned_integer_volume = integer_backtest.volume
    assert aligned_integer_volume is not None
    assert aligned_integer_volume.iloc[0].isna().all()
    pd.testing.assert_frame_equal(
        aligned_integer_volume.iloc[1:],
        integer_volume,
        check_dtype=False,
        check_freq=False,
    )

    # Derive trades independently from position changes rather than result helpers.
    integer_positions = pd.DataFrame({security.name: security.positions for security in integer_backtest.strategy.securities})
    float_positions = pd.DataFrame({security.name: security.positions for security in float_backtest.strategy.securities})
    integer_trades = integer_positions.diff()
    integer_trades.iloc[0] = integer_positions.iloc[0]
    float_trades = float_positions.diff()
    float_trades.iloc[0] = float_positions.iloc[0]
    pd.testing.assert_frame_equal(integer_trades, float_trades)

    # Cost-model accounting must match for numerically equivalent volume inputs.
    np.testing.assert_allclose(integer_backtest.strategy.prices, float_backtest.strategy.prices)
    np.testing.assert_allclose(integer_backtest.strategy.fees, float_backtest.strategy.fees)


def test_backtest_cost_model_runs_and_charges_fees():
    prices, volume, volatility = _impact_universe()
    bkt = bt.Backtest(
        _ew_strategy(),
        prices,
        commissions=bt.AlmgrenChrissCostModel(),
        volume=volume,
        volatility=volatility,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )

    bt.run(bkt)

    assert bkt.has_run
    assert bkt.strategy.fees.sum() > 0.0
    assert len(bkt.strategy.prices) == len(bkt.dates)


def test_backtest_cost_model_ac_zero_alpha_beta_matches_flat_commission():
    prices, volume, volatility = _impact_universe()
    eps = 0.0005

    cost_model = bt.Backtest(
        _ew_strategy("cost_model"),
        prices,
        commissions=bt.AlmgrenChrissCostModel(alpha=0, beta=0, epsilon=eps),
        volume=volume,
        volatility=volatility,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )
    legacy = bt.Backtest(
        _ew_strategy("legacy"),
        prices,
        commissions=lambda q, p: abs(q) * p * eps,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )

    bt.run(cost_model, legacy)

    np.testing.assert_allclose(
        cost_model.strategy.prices.values, legacy.strategy.prices.values
    )
    np.testing.assert_allclose(
        cost_model.strategy.fees.sum(), legacy.strategy.fees.sum()
    )


@pytest.mark.parametrize("lazy_security", [False, True], ids=["explicit", "lazy"])
def test_backtest_cost_model_matches_legacy_in_nested_strategies(
    lazy_security: bool,
):
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": 100.0}, index=dates)
    volume = pd.DataFrame({"A": 1_000_000.0}, index=dates)
    volatility = pd.DataFrame({"A": 0.02}, index=dates)
    epsilon = 0.01

    def trade_algos() -> list[bt.Algo]:
        return [
            bt.algos.RunOnDate(dates[0]),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ]

    def nested_strategy(name: str) -> bt.Strategy:
        security = "A" if lazy_security else bt.Security("A")
        leaf = bt.Strategy("leaf", trade_algos(), children=[security])
        return bt.Strategy(name, trade_algos(), children=[leaf])

    def legacy_commission(q: float, p: float) -> float:
        return abs(q) * p * epsilon

    cost_model = bt.Backtest(
        nested_strategy("cost_model"),
        prices,
        commissions=bt.AlmgrenChrissCostModel(alpha=0, beta=0, epsilon=epsilon),
        volume=volume,
        volatility=volatility,
        initial_capital=10_000.0,
        integer_positions=False,
        progress_bar=False,
    )
    legacy = bt.Backtest(
        nested_strategy("legacy"),
        prices,
        commissions=legacy_commission,
        initial_capital=10_000.0,
        integer_positions=False,
        progress_bar=False,
    )

    bt.run(cost_model, legacy)

    cost_leaf = cost_model.strategy["leaf"]
    legacy_leaf = legacy.strategy["leaf"]

    # Derive the real trade and fee directly from the cost-inclusive unit outlay.
    expected_quantity = 10_000.0 / (100.0 * (1.0 + epsilon))
    expected_fee = expected_quantity * 100.0 * epsilon
    assert cost_leaf["A"].position == pytest.approx(expected_quantity)
    assert cost_leaf.fees.sum() == pytest.approx(expected_fee)
    assert cost_model.strategy.value == pytest.approx(10_000.0 - expected_fee)
    assert cost_model.strategy.value == pytest.approx(legacy.strategy.value)

    # Paper strategies trade a fixed independent amount but obey the same cost contract.
    expected_paper_quantity = 1_000_000.0 / (100.0 * (1.0 + epsilon))
    expected_paper_fee = expected_paper_quantity * 100.0 * epsilon
    assert cost_leaf._paper["A"].position == pytest.approx(expected_paper_quantity)
    assert cost_leaf._paper.fees.sum() == pytest.approx(expected_paper_fee)
    assert cost_leaf.price == pytest.approx(legacy_leaf.price)


def test_backtest_cost_model_applies_to_dynamic_nested_strategy():
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    prices = pd.DataFrame({"A": 100.0}, index=dates)
    volume = pd.DataFrame({"A": 1_000_000.0}, index=dates)
    volatility = pd.DataFrame({"A": 0.02}, index=dates)

    def add_leaf(target):
        leaf = bt.Strategy(
            "leaf",
            [bt.algos.RunOnce(), bt.algos.SelectAll(), bt.algos.WeighEqually(), bt.algos.Rebalance()],
            children=["A"],
            parent=target,
        )
        leaf.setup_from_parent()
        leaf.update(target.now)
        target.allocate(target.value, leaf.name)
        return True

    strategy = bt.Strategy("root", [bt.algos.RunOnce(), add_leaf])
    backtest = bt.Backtest(
        strategy,
        prices,
        commissions=bt.AlmgrenChrissCostModel(alpha=0, beta=0, epsilon=0.01),
        volume=volume,
        volatility=volatility,
        initial_capital=10_000.0,
        integer_positions=False,
        progress_bar=False,
    )

    backtest.run()

    leaf = backtest.strategy["leaf"]
    expected_quantity = 10_000.0 / 101.0
    assert leaf["A"].position == pytest.approx(expected_quantity)
    assert leaf.fees.sum() == pytest.approx(expected_quantity)
    assert leaf._paper["A"].position == pytest.approx(1_000_000.0 / 101.0)


def test_backtest_cost_model_active_ac_costs_higher_than_flat_path():
    prices, volume, volatility = _impact_universe()
    eps = 0.0005

    flat = bt.Backtest(
        _ew_strategy("flat"),
        prices,
        commissions=bt.AlmgrenChrissCostModel(alpha=0, beta=0, epsilon=eps),
        volume=volume,
        volatility=volatility,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )
    active = bt.Backtest(
        _ew_strategy("active"),
        prices,
        commissions=bt.AlmgrenChrissCostModel(alpha=1, beta=1, epsilon=eps),
        volume=volume,
        volatility=volatility,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )

    bt.run(flat, active)

    assert active.strategy.fees.sum() > flat.strategy.fees.sum()


def test_backtest_sqrt_cost_model_runs():
    prices, volume, volatility = _impact_universe()
    bkt = bt.Backtest(
        _ew_strategy(),
        prices,
        commissions=bt.SqrtCostModel(Y=0.6),
        volume=volume,
        volatility=volatility,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )

    bt.run(bkt)
    assert bkt.strategy.fees.sum() > 0.0


def test_backtest_cost_model_validates_index_alignment():
    prices, volume, volatility = _impact_universe()
    misaligned = volume.iloc[1:].copy()
    with pytest.raises(ValueError, match="index"):
        bt.Backtest(
            _ew_strategy(),
            prices,
            commissions=bt.AlmgrenChrissCostModel(),
            volume=misaligned,
            volatility=volatility,
            progress_bar=False,
        )


def test_backtest_cost_model_validates_columns_alignment():
    prices, volume, volatility = _impact_universe()
    misaligned = volume.rename(columns={volume.columns[0]: "Z"})
    with pytest.raises(ValueError, match="columns"):
        bt.Backtest(
            _ew_strategy(),
            prices,
            commissions=bt.AlmgrenChrissCostModel(),
            volume=misaligned,
            volatility=volatility,
            progress_bar=False,
        )


def test_backtest_cost_model_requires_volume_and_volatility():
    prices, volume, volatility = _impact_universe()
    with pytest.raises(ValueError, match="required"):
        bt.Backtest(
            _ew_strategy(),
            prices,
            commissions=bt.AlmgrenChrissCostModel(),
            progress_bar=False,
        )


def test_backtest_cost_model_does_not_pollute_legacy_path():
    """Running a Backtest with a CostModel must not perturb the unmodified path."""
    prices, volume, volatility = _impact_universe()

    bt.run(
        bt.Backtest(
            _ew_strategy(),
            prices,
            commissions=bt.AlmgrenChrissCostModel(),
            volume=volume,
            volatility=volatility,
            progress_bar=False,
        )
    )
    legacy = bt.Backtest(_ew_strategy(), prices, progress_bar=False)
    bt.run(legacy)

    assert legacy.strategy.fees.sum() == 0.0


def test_backtest_cost_model_cost_scales_with_qty_via_volume():
    """Halving available volume (raising participation 10x) should raise AC depth/perm
    cost roughly proportionally."""
    prices, volume, volatility = _impact_universe()

    base_vol = volume.copy()
    thin_vol = volume / 10.0  # same trades execute at 10x participation

    base = bt.Backtest(
        _ew_strategy("base"),
        prices,
        commissions=bt.AlmgrenChrissCostModel(alpha=1, beta=1, epsilon=0.0),
        volume=base_vol,
        volatility=volatility,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )
    thin = bt.Backtest(
        _ew_strategy("thin"),
        prices,
        commissions=bt.AlmgrenChrissCostModel(alpha=1, beta=1, epsilon=0.0),
        volume=thin_vol,
        volatility=volatility,
        initial_capital=10_000_000.0,
        progress_bar=False,
    )

    bt.run(base, thin)

    # AC depth + perm scale linearly in (|q|/V) -> 10x participation -> ~10x cost
    assert thin.strategy.fees.sum() > 5 * base.strategy.fees.sum()
