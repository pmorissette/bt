from __future__ import division
from datetime import datetime, timedelta
from typing import ClassVar
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import random
import pickle

import bt
import bt.algos as algos


def test_algo_name():
    class TestAlgo(algos.Algo):
        pass

    actual = TestAlgo()

    assert actual.name == "TestAlgo"


class DummyAlgo(algos.Algo):
    def __init__(self, return_value=True):
        self.return_value = return_value
        self.called = False

    def __call__(self, target):
        self.called = True
        return self.return_value


def test_algo_stack():
    algo1 = DummyAlgo(return_value=True)
    algo2 = DummyAlgo(return_value=False)
    algo3 = DummyAlgo(return_value=True)

    target = mock.MagicMock()

    stack = bt.AlgoStack(algo1, algo2, algo3)

    actual = stack(target)
    assert not actual
    assert algo1.called
    assert algo2.called
    assert not algo3.called


def test_print_temp_data():
    target = mock.MagicMock()
    target.temp = {}
    target.temp["selected"] = ["c1", "c2"]
    target.temp["weights"] = [0.5, 0.5]

    algo = algos.PrintTempData()
    assert algo(target)

    algo = algos.PrintTempData("Selected: {selected}")
    assert algo(target)


def test_print_info():
    target = bt.Strategy("s", [])
    target.temp = {}

    algo = algos.PrintInfo()
    assert algo(target)

    algo = algos.PrintInfo("{now}: {name}")
    assert algo(target)


def test_run_once():
    algo = algos.RunOnce()
    assert algo(None)
    assert not algo(None)
    assert not algo(None)


def test_run_period():
    target = mock.MagicMock()

    dts = pd.date_range("2010-01-01", periods=35)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    algo = algos.RunPeriod()

    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data
    dts = target.data.index

    target.now = None
    assert not algo(target)

    # run on first date
    target.now = dts[0]
    assert not algo(target)

    # run on first supplied date
    target.now = dts[1]
    assert algo(target)

    # run on last date
    target.now = dts[len(dts) - 1]
    assert not algo(target)

    algo = algos.RunPeriod(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )

    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data
    dts = target.data.index

    # run on first date
    target.now = dts[0]
    assert not algo(target)

    # first supplied date
    target.now = dts[1]
    assert not algo(target)

    # run on last date
    target.now = dts[len(dts) - 1]
    assert algo(target)

    # date not in index
    target.now = datetime(2009, 2, 15)
    assert not algo(target)


def test_run_daily():
    target = mock.MagicMock()

    dts = pd.date_range("2010-01-01", periods=35)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    algo = algos.RunDaily()

    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    target.now = dts[1]
    assert algo(target)


@pytest.mark.parametrize(
    "dates,expected_start,expected_end",
    [
        (pd.date_range("2015-12-28", periods=9), "2016-01-04", "2016-01-03"),
        (pd.date_range("2018-12-28", periods=9), "2018-12-31", "2018-12-30"),
        (
            pd.to_datetime(["2018-12-27", "2018-12-28", "2018-12-31", "2019-01-02", "2019-01-03"]),
            "2018-12-31",
            "2018-12-28",
        ),
    ],
)
@pytest.mark.parametrize("run_on_end_of_period", [False, True])
@pytest.mark.parametrize("timezone", [None, "America/New_York"])
def test_run_weekly_does_not_split_an_iso_week_at_new_year(
    dates, expected_start, expected_end, run_on_end_of_period, timezone
):
    class RecordDates(bt.Algo):
        def __init__(self):
            self.dates = []

        def __call__(self, target):
            self.dates.append(target.now)
            return True

    data = pd.DataFrame({"asset": 100.0}, index=dates.tz_localize(timezone))
    strategy = bt.Strategy(
        "weekly",
        [
            algos.RunWeekly(
                run_on_first_date=False,
                run_on_end_of_period=run_on_end_of_period,
            ),
            RecordDates(),
        ],
    )
    backtest = bt.Backtest(strategy, data)
    backtest.run()

    expected = expected_end if run_on_end_of_period else expected_start
    assert backtest.strategy.stack.algos[1].dates == [pd.Timestamp(expected, tz=timezone)]


def test_run_weekly():
    dts = pd.date_range("2010-01-01", periods=367)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = algos.RunWeekly()
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # end of week
    target.now = dts[2]
    assert not algo(target)

    # new week
    target.now = dts[3]
    assert algo(target)

    algo = algos.RunWeekly(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # end of week
    target.now = dts[2]
    assert algo(target)

    # new week
    target.now = dts[3]
    assert not algo(target)

    dts = pd.DatetimeIndex(
        [datetime(2016, 1, 3), datetime(2017, 1, 8), datetime(2018, 1, 7)]
    )
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # check next year
    target.now = dts[1]
    assert algo(target)


def test_run_monthly():
    dts = pd.date_range("2010-01-01", periods=367)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = algos.RunMonthly()
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # end of month
    target.now = dts[30]
    assert not algo(target)

    # new month
    target.now = dts[31]
    assert algo(target)

    algo = algos.RunMonthly(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # end of month
    target.now = dts[30]
    assert algo(target)

    # new month
    target.now = dts[31]
    assert not algo(target)

    dts = pd.DatetimeIndex(
        [datetime(2016, 1, 3), datetime(2017, 1, 8), datetime(2018, 1, 7)]
    )
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # check next year
    target.now = dts[1]
    assert algo(target)


def test_run_quarterly():
    dts = pd.date_range("2010-01-01", periods=367)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = algos.RunQuarterly()
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # end of quarter
    target.now = dts[89]
    assert not algo(target)

    # new quarter
    target.now = dts[90]
    assert algo(target)

    algo = algos.RunQuarterly(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # end of quarter
    target.now = dts[89]
    assert algo(target)

    # new quarter
    target.now = dts[90]
    assert not algo(target)

    dts = pd.DatetimeIndex(
        [datetime(2016, 1, 3), datetime(2017, 1, 8), datetime(2018, 1, 7)]
    )
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # check next year
    target.now = dts[1]
    assert algo(target)


def test_run_yearly():
    dts = pd.date_range("2010-01-01", periods=367)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = algos.RunYearly()
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # end of year
    target.now = dts[364]
    assert not algo(target)

    # new year
    target.now = dts[365]
    assert algo(target)

    algo = algos.RunYearly(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )
    # adds the initial day
    backtest = bt.Backtest(bt.Strategy("", [algo]), data)
    target.data = backtest.data

    # end of year
    target.now = dts[364]
    assert algo(target)

    # new year
    target.now = dts[365]
    assert not algo(target)


def test_run_on_date():
    target = mock.MagicMock()
    target.now = pd.to_datetime("2010-01-01")

    algo = algos.RunOnDate("2010-01-01", "2010-01-02")
    assert algo(target)

    target.now = pd.to_datetime("2010-01-02")
    assert algo(target)

    target.now = pd.to_datetime("2010-01-03")
    assert not algo(target)


def test_run_if_out_of_bounds():
    algo = algos.RunIfOutOfBounds(0.5)
    dts = pd.date_range("2010-01-01", periods=3)

    s = bt.Strategy("s")
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    s.setup(data)

    s.temp["selected"] = ["c1", "c2"]
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}
    s.update(dts[0])
    s.children["c1"] = bt.core.SecurityBase("c1")
    s.children["c2"] = bt.core.SecurityBase("c2")

    s.children["c1"]._weight = 0.5
    s.children["c2"]._weight = 0.5
    assert not algo(s)

    s.children["c1"]._weight = 0.25
    s.children["c2"]._weight = 0.75
    assert not algo(s)

    s.children["c1"]._weight = 0.24
    s.children["c2"]._weight = 0.76
    assert algo(s)

    s.children["c1"]._weight = 0.75
    s.children["c2"]._weight = 0.25
    assert not algo(s)
    s.children["c1"]._weight = 0.76
    s.children["c2"]._weight = 0.24
    assert algo(s)

    s.temp["weights"] = {"c1": 0.0, "c2": 0.5}
    s.children["c1"]._weight = 0.0
    s.children["c2"]._weight = 0.5
    assert not algo(s)

    s.children["c1"]._weight = 0.2
    assert algo(s)
    s.children["c1"]._weight = -0.2
    assert algo(s)

    s.children["c1"]._weight = 0.0
    s.children["c2"]._weight = 0.9
    assert algo(s)


@pytest.mark.parametrize(
    ("initial_targets", "next_targets", "current_weight", "target_weight", "should_run", "expected_position"),
    [
        pytest.param({}, {"asset": 1.0}, 0.0, 1.0, True, 10.0, id="target-only-dict"),
        pytest.param({}, pd.Series({"asset": 1.0}), 0.0, 1.0, True, 10.0, id="target-only-series"),
        pytest.param({"asset": 1.0}, {}, 1.0, 0.0, True, 0.0, id="current-only"),
        pytest.param({}, {"asset": 0.0}, 0.0, 0.0, False, 0.0, id="target-only-zero"),
    ],
)
def test_run_if_out_of_bounds_checks_names_on_either_side(
    initial_targets: object,
    next_targets: object,
    current_weight: float,
    target_weight: float,
    should_run: bool,
    expected_position: float,
):
    algo = algos.RunIfOutOfBounds(0.5)
    rebalance = algos.Rebalance()
    date = pd.Timestamp("2020-01-01")
    strategy = bt.Strategy("s")
    strategy.setup(pd.DataFrame({"asset": [100.0]}, index=[date]))
    strategy.adjust(1000.0)
    strategy.update(date)

    # Establish an optional current position before checking the next sparse target.
    strategy.temp["weights"] = initial_targets
    assert rebalance(strategy)
    strategy.temp["weights"] = next_targets

    # Missing names have weight zero; apply the documented deviation rules independently.
    if target_weight == 0:
        expected_decision = current_weight != 0
    else:
        expected_decision = abs((current_weight - target_weight) / target_weight) > 0.5
    assert bool(expected_decision) is should_run
    child_exists_before = "asset" in strategy.children
    assert algo(strategy) is should_run
    assert ("asset" in strategy.children) is child_exists_before
    assert strategy.temp["weights"] is next_targets

    # Rebalance provides the downstream position oracle for each sparse-target direction.
    assert rebalance(strategy)
    position = strategy["asset"].position if "asset" in strategy.children else 0.0
    assert position == expected_position


@pytest.mark.parametrize(
    ("cash", "expected_position"),
    [
        pytest.param(0.0, 10.0, id="fully-invested"),
        pytest.param(0.5, 5.0, id="partly-invested"),
        pytest.param(1.0, 0.0, id="all-cash"),
    ],
)
def test_cash_targets_are_idempotent_and_in_bounds(cash: float, expected_position: float):
    date = pd.Timestamp("2020-01-01")
    strategy = bt.Strategy("s")
    strategy.setup(pd.DataFrame({"asset": [100.0]}, index=[date]))
    strategy.adjust(1000.0)
    strategy.update(date)
    strategy.temp["weights"] = {"asset": 1.0}
    strategy.temp["cash"] = cash

    rebalance = algos.Rebalance()
    gate = algos.RunIfOutOfBounds(0.01)

    # Full value is the independent base; cash scales the investable-slice target.
    assert rebalance(strategy)
    position = strategy["asset"].position if "asset" in strategy.children else 0.0
    outlay = strategy["asset"].outlays.loc[date] if "asset" in strategy.children else 0.0
    assert position == expected_position
    assert strategy.capital == pytest.approx(1000.0 * cash)
    assert strategy.value == pytest.approx(1000.0)
    assert not gate(strategy)

    # Reapplying unchanged targets must preserve allocation and transaction history.
    assert rebalance(strategy)
    position = strategy["asset"].position if "asset" in strategy.children else 0.0
    assert position == expected_position
    assert strategy.capital == pytest.approx(1000.0 * cash)
    assert strategy.value == pytest.approx(1000.0)
    assert not gate(strategy)
    if "asset" in strategy.children:
        assert strategy["asset"].outlays.loc[date] == outlay


@pytest.mark.parametrize(
    ("weights", "cash"),
    [
        pytest.param({"asset": 0.8}, 0.5, id="child-drift"),
        pytest.param({"asset": 0.625}, 0.2, id="cash-drift"),
    ],
)
def test_run_if_out_of_bounds_detects_cash_target_drift(weights: dict[str, float], cash: float):
    date = pd.Timestamp("2020-01-01")
    strategy = bt.Strategy("s")
    strategy.setup(pd.DataFrame({"asset": [100.0]}, index=[date]))
    strategy.adjust(1000.0)
    strategy.update(date)
    strategy.temp["weights"] = {"asset": 1.0}
    strategy.temp["cash"] = 0.5
    assert algos.Rebalance()(strategy)

    strategy.temp["weights"] = weights
    strategy.temp["cash"] = cash

    # Each case isolates one allocation beyond five percent of its effective target.
    assert algos.RunIfOutOfBounds(0.05)(strategy)


def test_run_if_out_of_bounds_handles_zero_value_cash_target():
    date = pd.Timestamp("2020-01-01")
    strategy = bt.Strategy("s")
    strategy.setup(pd.DataFrame(index=[date]))
    strategy.update(date)
    strategy.temp["weights"] = {}
    strategy.temp["cash"] = 0.5

    # Zero total value and zero capital have no cash-dollar mismatch.
    assert not algos.RunIfOutOfBounds(0.01)(strategy)


@pytest.mark.parametrize("hedge_type", [bt.HedgeSecurity, bt.CouponPayingHedgeSecurity])
def test_run_if_out_of_bounds_preserves_omitted_zero_weight_hedges(hedge_type: type[bt.core.SecurityBase]):
    date = pd.Timestamp("2020-01-01")
    prices = pd.DataFrame({"hedge": [100.0]}, index=[date])
    strategy = bt.FixedIncomeStrategy("s", children=[hedge_type("hedge")])
    strategy.setup(prices, coupons=prices * 0.0)
    strategy.update(date)
    strategy["hedge"].transact(-2)
    strategy.update(date)
    assert strategy["hedge"].weight == 0.0

    # An omitted zero-notional hedge is not an economic weight deviation or a close request.
    strategy.temp["weights"] = {}
    assert not algos.RunIfOutOfBounds(0.5)(strategy)
    assert algos.Rebalance()(strategy)
    assert strategy["hedge"].position == -2


def test_run_if_out_of_bounds_allows_sparse_target_rotation():
    dates = pd.date_range("2020-01-01", periods=2)
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    targets = pd.DataFrame({"A": [np.nan, 1.0], "B": [1.0, np.nan]}, index=dates)

    def make_backtest(name: str, gate: list[bt.Algo]) -> bt.Backtest:
        strategy = bt.Strategy(
            name,
            [algos.WeighTarget("targets"), *gate, algos.Rebalance()],
        )
        return bt.Backtest(
            strategy,
            prices,
            initial_capital=1000.0,
            additional_data={"targets": targets},
            progress_bar=False,
        )

    gated = make_backtest("gated", [algos.Or([algos.RunOnce(), algos.RunIfOutOfBounds(0.5)])])
    control = make_backtest("control", [])
    bt.run(gated, control)

    # At a constant price, 1,000 capital independently implies a complete 10-share rotation.
    for backtest in (gated, control):
        asset_a = backtest.strategy["A"]
        asset_b = backtest.strategy["B"]
        assert asset_b.positions.loc[dates[0]] == 10.0
        assert asset_a.position == 10.0
        assert asset_b.position == 0.0
        assert asset_a.outlays.loc[dates[1]] == 1000.0
        assert asset_b.outlays.loc[dates[1]] == -1000.0


def test_run_after_date():
    target = mock.MagicMock()
    target.now = pd.to_datetime("2010-01-01")

    algo = algos.RunAfterDate("2010-01-02")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-02")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-03")
    assert algo(target)


def test_run_after_days():
    target = mock.MagicMock()
    target.now = pd.to_datetime("2010-01-01")

    algo = algos.RunAfterDays(3)
    assert not algo(target)
    assert not algo(target)
    assert not algo(target)
    assert algo(target)


def test_set_notional():
    algo = algos.SetNotional("notional")

    s = bt.FixedIncomeStrategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    notional = pd.Series(index=dts[:2], data=[1e6, 5e6])

    s.setup(data, notional=notional)

    s.update(dts[0])
    assert algo(s)
    assert s.temp["notional_value"] == 1e6

    s.update(dts[1])
    assert algo(s)
    assert s.temp["notional_value"] == 5e6

    s.update(dts[2])
    assert not algo(s)


def test_rebalance():
    algo = algos.Rebalance()

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1}
    assert algo(s)
    assert s.value == 1000
    assert s.capital == 0
    c1 = s["c1"]
    assert c1.value == 1000
    assert c1.position == 10
    assert c1.weight == pytest.approx(1.0)

    s.temp["weights"] = {"c2": 1}

    assert algo(s)
    assert s.value == 1000
    assert s.capital == 0
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 1000
    assert c2.position == 10
    assert c2.weight == pytest.approx(1.0)


def test_rebalance_uses_current_price_for_stale_explicit_child():
    dates = pd.date_range("2020-01-01", periods=2)
    data = pd.DataFrame({"asset": [np.nan, 100.0]}, index=dates)
    strategy = bt.Strategy("strategy", children=[bt.Security("asset")])
    strategy.setup(data)
    strategy.adjust(1000.0)
    strategy.update(dates[0])
    strategy.update(dates[1])
    security = strategy["asset"]

    # Zero-position children stay on the bootstrap row until allocation.
    assert security.now == dates[0]
    assert pd.isna(security._price)
    strategy.temp["weights"] = {"asset": 1.0}

    assert algos.Rebalance()(strategy)
    assert security.now == dates[1]
    assert security.price == 100.0
    assert security.position == 10.0


@pytest.mark.parametrize(
    ("resolution", "invalid_price"),
    [
        pytest.param("existing", 0.0, id="existing-zero"),
        pytest.param("lazy", np.nan, id="lazy-nan"),
        pytest.param("default", None, id="default-missing"),
    ],
)
def test_rebalance_invalid_direct_target_preserves_state(
    resolution, invalid_price
):
    date = pd.Timestamp("2020-01-01")
    data = pd.DataFrame({"held": [100.0], "valid": [100.0]}, index=[date])
    children = [bt.Security("held"), bt.Security("valid")]
    if invalid_price is not None:
        data["invalid"] = invalid_price
        children.append(bt.Security("invalid", lazy_add=resolution == "lazy"))

    strategy = bt.Strategy("strategy", children=children)
    strategy.setup(data)
    strategy.adjust(1000.0)
    strategy.update(date)
    strategy.allocate(200.0, "held")
    strategy.update(date)
    # A valid first target and omitted holding expose any mutation before rejection.
    strategy.temp["weights"] = {"valid": 0.5, "invalid": 0.5}

    # Child resolution, portfolio state, and histories must remain exact.
    before_children = tuple(strategy.children)
    before_lazy_children = tuple(strategy._lazy_children)
    before_positions = {
        name: child.position for name, child in strategy.children.items()
    }
    before_histories = {
        name: (child.positions.copy(), child.outlays.copy())
        for name, child in strategy.children.items()
    }
    before_capital = strategy.capital
    before_value = strategy.value
    before_weights = strategy.temp["weights"].copy()
    before_stale = strategy.root.stale

    with pytest.raises(ValueError, match="Cannot allocate capital to invalid"):
        algos.Rebalance()(strategy)

    assert tuple(strategy.children) == before_children
    assert tuple(strategy._lazy_children) == before_lazy_children
    assert {
        name: child.position for name, child in strategy.children.items()
    } == before_positions
    assert strategy.capital == before_capital
    assert strategy.value == before_value
    assert strategy.temp["weights"] == before_weights
    assert strategy.root.stale == before_stale
    for name, (positions, outlays) in before_histories.items():
        pd.testing.assert_series_equal(strategy[name].positions, positions)
        pd.testing.assert_series_equal(strategy[name].outlays, outlays)


@pytest.mark.parametrize("missing_column", [False, True])
def test_rebalance_invalid_fixed_income_target_preserves_state(missing_column):
    date = pd.Timestamp("2020-01-01")
    data = pd.DataFrame(
        {"held": [100.0], "valid": [100.0], "invalid": [np.nan]}, index=[date]
    )
    if missing_column:
        data = data.drop(columns="invalid")
    strategy = bt.FixedIncomeStrategy(
        "strategy",
        children=[
            bt.FixedIncomeSecurity("held"),
            bt.FixedIncomeSecurity("valid"),
            bt.FixedIncomeSecurity("invalid", lazy_add=True),
        ],
    )
    strategy.setup(data)
    strategy.update(date)
    strategy.transact(2.0, "held")
    strategy.update(date)
    # Quantity-notional targets use transact but share the direct-price gate.
    strategy.temp["notional_value"] = 1000.0
    strategy.temp["weights"] = {"valid": 0.5, "invalid": 0.5}

    # Preserve both quantity-notional and market-value state on rejection.
    before_children = tuple(strategy.children)
    before_lazy_children = tuple(strategy._lazy_children)
    before_positions = {
        name: child.position for name, child in strategy.children.items()
    }
    before_capital = strategy.capital
    before_value = strategy.value
    before_notional_value = strategy.notional_value

    with pytest.raises(ValueError, match="Cannot allocate capital to invalid"):
        algos.Rebalance()(strategy)

    assert tuple(strategy.children) == before_children
    assert tuple(strategy._lazy_children) == before_lazy_children
    assert {
        name: child.position for name, child in strategy.children.items()
    } == before_positions
    assert strategy.capital == before_capital
    assert strategy.value == before_value
    assert strategy.notional_value == before_notional_value


@pytest.mark.parametrize("security_type", [bt.FixedIncomeSecurity, bt.CouponPayingSecurity])
@pytest.mark.parametrize("lazy", [False, True])
def test_rebalance_opens_zero_price_fixed_income_target(security_type, lazy):
    date = pd.Timestamp("2020-01-01")
    data = pd.DataFrame({"asset": [0.0]}, index=[date])
    strategy = bt.FixedIncomeStrategy(
        "strategy", children=[security_type("asset", lazy_add=lazy)]
    )
    strategy.setup(data, coupons=data * 0.0)
    strategy.update(date)
    strategy.temp["notional_value"] = 10.0
    strategy.temp["weights"] = {"asset": 1.0}

    assert algos.Rebalance()(strategy)
    assert strategy["asset"].position == 10.0
    assert strategy.notional_value == 10.0
    assert strategy.capital == 0.0
    assert strategy.value == 0.0


@pytest.mark.parametrize("security_type", [bt.FixedIncomeSecurity, bt.CouponPayingSecurity])
@pytest.mark.parametrize("position", [10.0, -10.0])
@pytest.mark.parametrize("target_weight", [0.0, 1e-17, 0.5, 1.5])
def test_rebalance_adjusts_zero_price_fixed_income_target(security_type, position, target_weight):
    dates = pd.date_range("2020-01-01", periods=2)
    data = pd.DataFrame({"asset": [100.0, 0.0]}, index=dates)
    strategy = bt.FixedIncomeStrategy("strategy", children=[security_type("asset")])
    strategy.setup(data, coupons=data * 0.0)
    strategy.update(dates[0])
    strategy.transact(position, "asset")
    strategy.update(dates[0])
    strategy.update(dates[1])
    before_capital = strategy.capital
    strategy.temp["notional_value"] = abs(position)
    strategy.temp["weights"] = {"asset": np.sign(position) * target_weight}

    assert algos.Rebalance()(strategy)
    expected_position = 0.0 if target_weight < 1e-16 else position * target_weight
    assert strategy["asset"].position == expected_position
    assert strategy.notional_value == abs(expected_position)
    assert strategy.capital == before_capital
    assert strategy.value == before_capital


@pytest.mark.parametrize("target_weight", [None, 0.0, 1e-17], ids=["omitted", "explicit-zero", "near-zero"])
@pytest.mark.parametrize("amount", [1000.0, -1000.0])
def test_rebalance_closes_zero_price_position(amount, target_weight):
    dates = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame({"asset": [100.0, 0.0]}, index=dates)
    strategy = bt.Strategy("strategy", children=["asset"])
    strategy.setup(data)
    strategy.update(dates[0])
    strategy.adjust(1000.0)
    strategy.allocate(amount, "asset")
    strategy.update(dates[0])
    strategy.update(dates[1])
    security = strategy["asset"]
    initial_position = security.position
    initial_cash = strategy.capital
    initial_value = strategy.value
    # Omission and an explicit neutral target both close the live position.
    strategy.temp["weights"] = (
        {} if target_weight is None else {"asset": target_weight}
    )

    assert algos.Rebalance()(strategy)

    transactions = strategy.get_transactions()
    assert security.position == 0.0
    assert strategy.capital == initial_cash
    assert strategy.value == initial_value
    assert transactions.iloc[-1]["quantity"] == -initial_position
    assert transactions.iloc[-1]["price"] == 0.0


def test_rebalance_closes_omitted_zero_value_nested_strategy():
    dates = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame({"asset": [100.0, 0.0]}, index=dates)
    sleeve = bt.Strategy("sleeve", children=["asset"])
    strategy = bt.Strategy("strategy", children=[sleeve])
    strategy.setup(data)
    strategy.update(dates[0])
    strategy.adjust(1000.0)
    strategy.rebalance(1.0, "sleeve")
    sleeve = strategy["sleeve"]
    sleeve.allocate(1000.0, "asset")
    strategy.update(dates[0])
    strategy.update(dates[1])
    initial_cash = strategy.capital
    initial_value = strategy.value
    strategy.temp["weights"] = {}

    assert algos.Rebalance()(strategy)

    assert sleeve["asset"].position == 0.0
    assert strategy.capital == initial_cash
    assert strategy.value == initial_value


def test_rebalance_with_commissions():
    algo = algos.Rebalance()

    s = bt.Strategy("s")
    s.set_commissions(lambda q, p: 1)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1}

    assert algo(s)
    assert s.value == 999
    assert s.capital == 99
    c1 = s["c1"]
    assert c1.value == 900
    assert c1.position == 9
    assert c1.weight == pytest.approx(900 / 999.0)

    s.temp["weights"] = {"c2": 1}

    assert algo(s)
    assert s.value == 997
    assert s.capital == 97
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 900
    assert c2.position == 9
    assert c2.weight == pytest.approx(900.0 / 997)


def test_rebalance_with_cash():
    algo = algos.Rebalance()

    s = bt.Strategy("s")
    s.set_commissions(lambda q, p: 1)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1}
    # set cash amount
    s.temp["cash"] = 0.5

    assert algo(s)
    assert s.value == 999
    assert s.capital == 599
    c1 = s["c1"]
    assert c1.value == 400
    assert c1.position == 4
    assert c1.weight == pytest.approx(400.0 / 999)

    s.temp["weights"] = {"c2": 1}
    # change cash amount
    s.temp["cash"] = 0.25

    assert algo(s)
    assert s.value == 997
    assert s.capital == 297
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 700
    assert c2.position == 7
    assert c2.weight == pytest.approx(700.0 / 997)


def test_rebalance_updatecount():

    algo = algos.Rebalance()

    s = bt.Strategy("s")
    s.use_integer_positions(False)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4", "c5"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 0.25, "c2": 0.25, "c3": 0.25, "c4": 0.25}

    update = bt.core.SecurityBase.update
    bt.core.SecurityBase._update_call_count = 0

    def side_effect(self, *args, **kwargs):
        bt.core.SecurityBase._update_call_count += 1
        return update(self, *args, **kwargs)

    with mock.patch.object(bt.core.SecurityBase, "update", side_effect) as mock_update:
        assert algo(s)

    assert s.value == 1000
    assert s.capital == 0

    # Update is called once when each weighted security is created (4)
    # and once for each security after all allocations are made (4)
    assert bt.core.SecurityBase._update_call_count == 8

    s.update(dts[1])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    update = bt.core.SecurityBase.update
    bt.core.SecurityBase._update_call_count = 0

    def side_effect(self, *args, **kwargs):
        bt.core.SecurityBase._update_call_count += 1
        return update(self, *args, **kwargs)

    with mock.patch.object(bt.core.SecurityBase, "update", side_effect) as mock_update:
        assert algo(s)

    # Update is called once for each weighted security before allocation (4)
    # and once for each security after all allocations are made (4)
    assert bt.core.SecurityBase._update_call_count == 8

    s.update(dts[2])
    s.temp["weights"] = {"c1": 0.25, "c2": 0.25, "c3": 0.25, "c4": 0.25}

    update = bt.core.SecurityBase.update
    bt.core.SecurityBase._update_call_count = 0

    def side_effect(self, *args, **kwargs):
        bt.core.SecurityBase._update_call_count += 1
        return update(self, *args, **kwargs)

    with mock.patch.object(bt.core.SecurityBase, "update", side_effect) as mock_update:
        assert algo(s)

    # Update is called once for each weighted security before allocation (2)
    # and once for each security after all allocations are made (4)
    assert bt.core.SecurityBase._update_call_count == 6


def test_rebalance_fixedincome():
    algo = algos.Rebalance()
    c1 = bt.Security("c1")
    c2 = bt.CouponPayingSecurity("c2")
    s = bt.FixedIncomeStrategy("s", children=[c1, c2])

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    coupons = pd.DataFrame(index=dts, columns=["c2"], data=0)
    s.setup(data, coupons=coupons)
    s.update(dts[0])
    s.temp["notional_value"] = 1000
    s.temp["weights"] = {"c1": 1}
    s.temp["cash"] = 0.5
    assert algo(s)
    assert s.value == pytest.approx(0.0)
    assert s.notional_value == 1000
    assert s.capital == -1000
    c1 = s["c1"]
    assert c1.value == 1000
    assert c1.notional_value == 1000
    assert c1.position == 10
    assert c1.weight == pytest.approx(1.0)
    assert not algos.RunIfOutOfBounds(0.01)(s)

    s.temp["weights"] = {"c2": 1}

    assert algo(s)
    assert s.value == pytest.approx(0.0)
    assert s.notional_value == 1000
    assert s.capital == -1000 * 100
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.notional_value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 1000 * 100
    assert c2.notional_value == 1000
    assert c2.position == 1000
    assert c2.weight == pytest.approx(1.0)


@pytest.mark.parametrize("lazy_add", [False, True], ids=["explicit", "lazy"])
def test_rebalance_fixed_income_security_targets_notional(lazy_add):
    date = pd.Timestamp("2020-01-01")
    prices = pd.DataFrame({"bond": [100.0]}, index=[date])
    strategy = bt.FixedIncomeStrategy(
        "strategy",
        children=[bt.FixedIncomeSecurity("bond", multiplier=2, lazy_add=lazy_add)],
    )
    strategy.setup(prices)
    strategy.update(date)
    strategy.temp["notional_value"] = 1000.0
    strategy.temp["weights"] = {"bond": 1.0}

    # Both child-resolution paths must dispatch the target as quantity notional.
    assert algos.Rebalance()(strategy)
    security = strategy["bond"]
    assert security.fixed_income
    assert security.position == 1000.0
    assert security.notional_value == 1000.0
    assert security.value == 200000.0

    # Repeating the same Algo target must not add another transaction.
    outlays = security.outlays.copy()
    assert algos.Rebalance()(strategy)
    assert security.position == 1000.0
    pd.testing.assert_series_equal(security.outlays, outlays)


def test_select_all():
    algo = algos.SelectAll()

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # make sure don't keep nan
    s.update(dts[1])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = algos.SelectAll(include_no_data=True)

    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = algos.SelectAll(include_negative=True)

    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_randomly_n_none():
    algo = algos.SelectRandomly(n=None)  # Behaves like SelectAll

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # make sure don't keep nan
    s.update(dts[1])

    assert algo(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = algos.SelectRandomly(n=None, include_no_data=True)

    assert algo2(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = algos.SelectRandomly(n=None, include_negative=True)

    assert algo3(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_randomly():

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    data.loc[dts[0], "c1"] = np.nan
    data.loc[dts[0], "c2"] = 95
    data.loc[dts[0], "c3"] = -5

    s.setup(data)
    s.update(dts[0])

    algo = algos.SelectRandomly(n=1)
    assert algo(s)
    assert s.temp.pop("selected") == ["c2"]

    random.seed(1000)
    algo = algos.SelectRandomly(n=1, include_negative=True)
    assert algo(s)
    assert s.temp.pop("selected") == ["c3"]

    random.seed(1009)
    algo = algos.SelectRandomly(n=1, include_no_data=True)
    assert algo(s)
    assert s.temp.pop("selected") == ["c1"]

    random.seed(1009)
    # If selected already set, it will further filter it
    s.temp["selected"] = ["c2"]
    algo = algos.SelectRandomly(n=1, include_no_data=True)
    assert algo(s)
    assert s.temp.pop("selected") == ["c2"]


def test_select_these():
    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    s.setup(data)
    s.update(dts[0])

    algo = algos.SelectThese(["c1", "c2"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    algo = algos.SelectThese(["c1"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c1" in selected

    # make sure don't keep nan
    s.update(dts[1])

    algo = algos.SelectThese(["c1", "c2"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = algos.SelectThese(["c1", "c2"], include_no_data=True)

    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = algos.SelectThese(["c1", "c2"], include_negative=True)

    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_where_all():
    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)

    s.setup(data, where=where)
    s.update(dts[0])

    algo = algos.SelectWhere("where")
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # make sure don't keep nan
    s.update(dts[1])

    algo = algos.SelectThese(["c1", "c2"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = algos.SelectWhere("where", include_no_data=True)

    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = algos.SelectWhere("where", include_negative=True)

    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_where():
    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)
    where.loc[dts[1]] = False
    where.loc[dts[2], "c1"] = False

    algo = algos.SelectWhere("where")

    s.setup(data, where=where)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    s.update(dts[1])
    assert algo(s)
    assert s.temp["selected"] == []

    s.update(dts[2])
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_where_legacy():
    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)
    where.loc[dts[1]] = False
    where.loc[dts[2], "c1"] = False

    algo = algos.SelectWhere(where)

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    s.update(dts[1])
    assert algo(s)
    assert s.temp["selected"] == []

    s.update(dts[2])
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_where_missing_date():
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    signal = pd.DataFrame([[True, False]], index=[dts[0]], columns=data.columns)
    s = bt.Strategy("s")
    s.setup(data, where=signal)
    s.update(dts[1])

    assert not algos.SelectWhere("where")(s)
    assert "selected" not in s.temp


def test_select_where_missing_date_stops_algo_stack():
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    signal = pd.DataFrame([[True, False]], index=[dts[0]], columns=data.columns)
    s = bt.Strategy("s")
    s.setup(data)
    s.update(dts[1])

    stack = bt.AlgoStack(algos.SelectAll(), algos.SelectWhere(signal), algos.WeighEqually())
    assert not stack(s)
    assert s.temp["selected"] == ["c1", "c2"]
    assert "weights" not in s.temp


def test_select_regex():
    s = bt.Strategy("s")
    algo = algos.SelectRegex("c1")

    s.temp["selected"] = ["a1", "c1", "c2", "c11", "cc1"]
    assert algo(s)
    assert s.temp["selected"] == ["c1", "c11", "cc1"]

    algo = algos.SelectRegex("^c1$")
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


@pytest.mark.parametrize("perm, expected", [
    ({}, ["c", "a", "b", "a"]),
    ({"rolled": {"b"}}, ["c", "a", "a"]),
    ({"closed": {"a"}}, ["c", "b"]),
    ({"rolled": {"a", "b"}, "closed": {"b", "c"}}, []),
])
@pytest.mark.parametrize("selected_count", [4, 31, 32])
def test_select_active_preserves_order_and_registries(perm, expected, selected_count):
    target = bt.Strategy("s")
    # Check the short path and both sides of the preparation cutoff with the same oracle.
    padding = [f"other_{i}" for i in range(selected_count - 4)]
    selected = ["c", "a", "b", "a"] + padding
    target.temp = {"selected": selected, "other": 42}
    target.perm = perm
    original = {key: value.copy() for key, value in perm.items()}
    references = perm.copy()

    assert algos.SelectActive()(target)
    assert target.temp == {"selected": expected + padding, "other": 42}
    assert selected == ["c", "a", "b", "a"] + padding
    assert target.perm == original
    assert all(target.perm[key] is value for key, value in references.items())


@pytest.mark.parametrize("representation", ["iterator", "list_subclass", "label_subclass"])
def test_select_active_observes_registry_updates_during_selection(representation):
    target = bt.Strategy("s")
    target.perm["closed"] = set()
    # Reach the size gate so each hook still challenges snapshot eligibility.
    padding = [f"other_{i}" for i in range(30)]

    def labels():
        yield "first"
        target.perm["closed"].add("second")
        yield "second"
        yield from padding

    class UpdatingList(list):
        def __iter__(self):
            return labels()

    class UpdatingLabel(str):
        def __hash__(self):
            target.perm["closed"].add("second")
            return super().__hash__()

    selected = {"iterator": labels(), "list_subclass": UpdatingList(["first", "second"] + padding), "label_subclass": [UpdatingLabel("first"), "second"] + padding}[representation]
    # A snapshot before iteration/hash hooks would incorrectly retain the later label.
    stack = bt.AlgoStack(algos.SelectThese(selected, include_no_data=True, include_negative=True), algos.SelectActive())
    assert stack(target)
    assert target.temp["selected"] == ["first"] + padding
    assert target.perm["closed"] == {"second"}


@pytest.mark.parametrize("registry", ["rolled", "closed"])
def test_select_active_preserves_registry_label_hooks(registry):
    target = bt.Strategy("s")
    target.perm["closed"] = set()

    class ClosingLabel(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            target.perm["closed"].add("second")
            return super().__eq__(other)

    target.perm[registry] = {ClosingLabel("first")}
    padding = [f"other_{i}" for i in range(30)]
    target.temp["selected"] = ["first", "second"] + padding
    # Registry equality can close a later candidate, even with an ordinary selection list.
    assert algos.SelectActive()(target)
    assert target.temp["selected"] == padding
    assert "second" in target.perm["closed"]


def test_select_active_preserves_consumable_registry():
    target = bt.Strategy("s")
    target.perm["closed"] = iter(["second"])
    padding = [f"other_{i}" for i in range(30)]
    target.temp["selected"] = ["first", "second"] + padding
    # The first union consumes the registry; the second candidate sees it exhausted.
    assert algos.SelectActive()(target)
    assert target.temp["selected"] == ["first", "second"] + padding


@pytest.mark.parametrize("container", [list, tuple, iter])
def test_select_active_no_work_does_not_evaluate_registries(container):
    target = bt.Strategy("s")
    target.perm = {"rolled": None, "closed": None}
    target.temp["selected"] = container([])
    assert algos.SelectActive()(target)
    assert target.temp["selected"] == []
    assert target.perm == {"rolled": None, "closed": None}


def test_resolve_on_the_run():
    s = bt.Strategy("s")
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "b1"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c2"] = -5

    on_the_run = pd.DataFrame(index=dts, columns=["c"], data="c1")
    on_the_run.loc[dts[2], "c"] = "c2"

    s.setup(data, on_the_run=on_the_run)
    s.update(dts[0])

    s.temp["selected"] = ["c", "b1"]
    algo = algos.ResolveOnTheRun("on_the_run")
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "b1" in selected

    # make sure don't keep nan
    s.update(dts[1])

    s.temp["selected"] = ["c", "b1"]
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "b1" in selected

    # if specify include_no_data then 2
    algo2 = algos.ResolveOnTheRun("on_the_run", include_no_data=True)
    s.temp["selected"] = ["c", "b1"]
    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "b1" in selected

    # behavior on negative prices
    s.update(dts[2])

    s.temp["selected"] = ["c", "b1"]
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "b1" in selected

    algo3 = algos.ResolveOnTheRun("on_the_run", include_negative=True)
    s.temp["selected"] = ["c", "b1"]
    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c2" in selected
    assert "b1" in selected


def test_select_types():
    c1 = bt.Security("c1")
    c2 = bt.CouponPayingSecurity("c2")
    c3 = bt.HedgeSecurity("c3")
    c4 = bt.CouponPayingHedgeSecurity("c4")
    c5 = bt.FixedIncomeSecurity("c5")

    s = bt.Strategy("p", children=[c1, c2, c3, c4, c5])

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4", "c5"], data=100.0)
    coupons = pd.DataFrame(index=dts, columns=["c2", "c4"], data=0.0)
    s.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i])

    algo = algos.SelectTypes(
        include_types=(bt.Security, bt.HedgeSecurity), exclude_types=()
    )
    assert algo(s)
    assert s.temp.pop("selected") == ["c1", "c3"]

    algo = algos.SelectTypes(
        include_types=(bt.core.SecurityBase,), exclude_types=(bt.CouponPayingSecurity,)
    )
    assert algo(s)
    assert s.temp.pop("selected") == ["c1", "c3", "c5"]

    s.temp["selected"] = ["c1", "c2", "c3"]
    algo = algos.SelectTypes(include_types=(bt.core.SecurityBase,))
    assert algo(s)
    assert s.temp.pop("selected") == ["c1", "c2", "c3"]


@pytest.mark.parametrize(
    "prior,expected",
    [
        pytest.param([], [], id="empty-list"),
        pytest.param(["beta", "missing", "beta"], ["beta"], id="partial-list"),
        pytest.param(["beta", "alpha"], ["alpha", "beta"], id="reordered-list"),
        pytest.param(pd.Series(["beta"], index=["alpha"]), ["alpha"], id="series-index-membership"),
        pytest.param(pd.Index(["beta", "alpha"]), ["alpha", "beta"], id="index"),
        pytest.param(("beta", "alpha"), ["alpha", "beta"], id="tuple"),
        pytest.param(["beta", []], ["beta"], id="unhashable-nonmatch"),
        pytest.param([np.str_("beta")], ["beta"], id="numpy-string"),
    ],
)
def test_select_types_prior_selection(prior, expected):
    target = bt.Strategy("parent", children=[bt.Security("alpha"), bt.Strategy("sleeve"), bt.Security("beta")])
    original = prior.copy(deep=True) if isinstance(prior, pd.Series) else list(prior)
    marker = object()
    target.temp["other"] = marker
    selector = algos.SelectThese(prior, include_no_data=True, include_negative=True)

    assert bt.AlgoStack(selector, algos.SelectTypes(include_types=(bt.Security,)))(target)

    assert target.temp["selected"] == expected
    assert target.temp["other"] is marker
    assert selector.tickers is prior
    if isinstance(prior, pd.Series):
        # Series membership uses its index, not the values produced by iteration.
        pd.testing.assert_series_equal(prior, original)
    else:
        assert list(prior) == original


@pytest.mark.parametrize("extra", [["missing"], [[]], [np.str_("missing")]])
def test_select_types_large_prior_selection_preserves_child_order(extra):
    names = [f"asset_{i}" for i in range(100)]
    target = bt.Strategy("parent", children=[bt.Security(name) for name in names])
    prior = list(reversed(names[::2])) + [names[0]] + extra
    target.temp["selected"] = prior

    assert algos.SelectTypes()(target)

    assert target.temp["selected"] == names[::2]
    assert prior == list(reversed(names[::2])) + [names[0]] + extra


def test_select_types_preserves_custom_membership_and_prior_rereads():
    target = bt.Strategy("parent", children=[bt.Security("alpha"), bt.Security("beta")])

    class Selection(list):
        def __contains__(self, name):
            # Caller-defined membership can replace the selection for the next child.
            target.temp["selected"] = ["beta"]
            return name == "alpha"

    prior = Selection(["alpha"])
    target.temp["selected"] = prior

    assert algos.SelectTypes()(target)

    assert target.temp["selected"] == ["alpha", "beta"]
    assert prior == ["alpha"]


@pytest.mark.parametrize("custom_child", [False, True], ids=["prior-label", "child-label"])
@pytest.mark.parametrize("filler_count", [0, 40])
def test_select_types_preserves_string_subclass_equality(custom_child, filler_count):
    class Label(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            # Custom equality need not agree with ordinary string hash membership.
            return str(self).casefold() == str(other).casefold()

    name = Label("ALPHA") if custom_child else "alpha"
    filler = [f"extra_{i}" for i in range(filler_count)]
    prior = (["alpha"] if custom_child else [Label("ALPHA")]) + filler
    target = bt.Strategy("parent", children=[bt.Security(child) for child in [name] + filler])
    target.temp["selected"] = prior

    assert algos.SelectTypes()(target)

    assert target.temp["selected"] == [name] + filler
    assert prior == (["alpha"] if custom_child else [Label("ALPHA")]) + filler


@pytest.mark.parametrize("children", [[], [bt.Strategy("sleeve")]], ids=["no-children", "no-type-matches"])
def test_select_types_without_eligible_children_does_not_read_prior(children):
    target = bt.Strategy("parent", children=children)
    target.temp["selected"] = None

    assert algos.SelectTypes(include_types=(bt.Security,))(target)

    assert target.temp["selected"] == []


def test_weight_equally():
    algo = algos.WeighEqually()

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert "c1" in weights
    assert weights["c1"] == pytest.approx(0.5)
    assert "c2" in weights
    assert weights["c2"] == pytest.approx(0.5)


def test_weight_specified():
    algo = algos.WeighSpecified(c1=0.6, c2=0.4)

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert "c1" in weights
    assert weights["c1"] == pytest.approx(0.6)
    assert "c2" in weights
    assert weights["c2"] == pytest.approx(0.4)


def test_scale_weights():
    s = bt.Strategy("s")
    algo = algos.ScaleWeights(-0.5)

    s.temp["weights"] = {"c1": 0.5, "c2": -0.4, "c3": 0}
    assert algo(s)
    assert s.temp["weights"] == pytest.approx({"c1": -0.25, "c2": 0.2, "c3": 0})


def test_select_has_data():
    algo = algos.SelectHasData(min_count=3, lookback=pd.DateOffset(days=3))

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=10)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[0], "c1"] = np.nan
    data.loc[dts[1], "c1"] = np.nan

    s.setup(data)
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected


def test_select_has_data_preselected():
    algo = algos.SelectHasData(min_count=3, lookback=pd.DateOffset(days=3))

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[0], "c1"] = np.nan
    data.loc[dts[1], "c1"] = np.nan

    s.setup(data)
    s.update(dts[2])
    s.temp["selected"] = ["c1"]

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 0


def _selector_with_independent_price_flags(selector_name, columns):
    if selector_name == "SelectAll":
        return algos.SelectAll(include_no_data=True, include_negative=False)
    if selector_name == "SelectThese":
        return algos.SelectThese(
            columns, include_no_data=True, include_negative=False
        )
    if selector_name == "SelectHasData":
        return algos.SelectHasData(
            lookback=pd.DateOffset(days=1),
            min_count=1,
            include_no_data=True,
            include_negative=False,
        )
    if selector_name == "SelectWhere":
        return algos.SelectWhere(
            "signal", include_no_data=True, include_negative=False
        )
    if selector_name == "SelectRandomly":
        return algos.SelectRandomly(
            n=None, include_no_data=True, include_negative=False
        )
    if selector_name == "ResolveOnTheRun":
        return algos.ResolveOnTheRun(
            "on_the_run", include_no_data=True, include_negative=False
        )
    raise ValueError(f"Unknown selector: {selector_name}")


@pytest.mark.parametrize(
    "selector_name",
    [
        "SelectAll",
        "SelectThese",
        "SelectHasData",
        "SelectWhere",
        "SelectRandomly",
        "ResolveOnTheRun",
    ],
)
def test_selector_filters_treat_include_no_data_and_include_negative_independently(
    selector_name,
):
    # Give every selector enough history while crossing missing and non-positive prices.
    dts = pd.date_range("2010-01-01", periods=2)
    columns = ["missing", "negative", "zero", "live"]
    data = pd.DataFrame(1.0, index=dts, columns=columns)
    data.loc[dts[-1]] = [np.nan, -1.0, 0.0, 1.0]
    signal = pd.DataFrame(True, index=dts, columns=columns)
    aliases = [f"alias_{column}" for column in columns]
    on_the_run = pd.DataFrame([columns, columns], index=dts, columns=aliases)

    strategy = bt.Strategy("strategy")
    strategy.setup(data, signal=signal, on_the_run=on_the_run)
    strategy.update(dts[-1])
    if selector_name == "ResolveOnTheRun":
        strategy.temp["selected"] = aliases

    selector = _selector_with_independent_price_flags(selector_name, columns)
    assert selector(strategy)
    assert list(strategy.temp["selected"]) == ["missing", "live"]


@pytest.mark.parametrize(
    "include_no_data, include_negative, expected",
    [
        (False, False, ["live"]),
        (False, True, ["negative", "zero", "live"]),
        (True, False, ["missing", "live"]),
        (True, True, ["missing", "negative", "zero", "live"]),
    ],
)
def test_select_all_include_no_data_and_include_negative_matrix(
    include_no_data, include_negative, expected
):
    # The expected labels are the independent truth table for the two public flags.
    dt = pd.Timestamp("2010-01-01")
    data = pd.DataFrame(
        [[np.nan, -1.0, 0.0, 1.0]],
        index=[dt],
        columns=["missing", "negative", "zero", "live"],
    )
    strategy = bt.Strategy("strategy")
    strategy.setup(data)
    strategy.update(dt)

    selector = algos.SelectAll(
        include_no_data=include_no_data, include_negative=include_negative
    )
    assert selector(strategy)
    assert list(strategy.temp["selected"]) == expected


@pytest.mark.parametrize("adverse_price", [-1.0, 0.0])
def test_selector_price_flags_protect_rebalance(adverse_price):
    # A non-positive candidate must be removed before equal weighting and allocation.
    data = pd.DataFrame(
        [[adverse_price, 1.0]],
        index=[pd.Timestamp("2010-01-01")],
        columns=["adverse", "live"],
    )
    strategy = bt.Strategy(
        "strategy",
        [
            algos.SelectThese(
                ["adverse", "live"],
                include_no_data=True,
                include_negative=False,
            ),
            algos.WeighEqually(),
            algos.Rebalance(),
        ],
    )
    backtest = bt.Backtest(strategy, data, initial_capital=10000.0)

    bt.run(backtest)

    assert "adverse" not in backtest.strategy.children
    assert backtest.strategy["live"].position == pytest.approx(10000.0)


def test_selector_price_flags_preserve_on_the_run_aliases():
    # The documented fixed-income stack selects an absent alias before resolving it.
    dt = pd.Timestamp("2010-01-01")
    data = pd.DataFrame([[1.0]], index=[dt], columns=["bond"])
    on_the_run = pd.DataFrame([["bond"]], index=[dt], columns=["alias"])
    strategy = bt.Strategy("strategy")
    strategy.setup(data, on_the_run=on_the_run)
    strategy.update(dt)

    stack = bt.AlgoStack(
        algos.SelectThese(["alias"], include_no_data=True),
        algos.ResolveOnTheRun("on_the_run"),
    )

    assert stack(strategy)
    assert strategy.temp["selected"] == ["bond"]


@mock.patch("ffn.calc_erc_weights")
def test_weigh_erc(mock_erc):
    algo = algos.WeighERC(lookback=pd.DateOffset(days=5))

    mock_erc.return_value = pd.Series({"c1": 0.3, "c2": 0.7})

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    assert mock_erc.called
    rets = mock_erc.call_args[0][0]
    assert len(rets) == 4
    assert "c1" in rets
    assert "c2" in rets

    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c1"] == pytest.approx(0.3)
    assert weights["c2"] == pytest.approx(0.7)


def test_weigh_target():
    algo = algos.WeighTarget("target")

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    target = pd.DataFrame(index=dts[:2], columns=["c1", "c2"], data=0.5)
    target.loc[dts[1], "c1"] = 1.0
    target.loc[dts[1], "c2"] = 0.0

    s.setup(data, target=target)

    s.update(dts[0])
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c1"] == pytest.approx(0.5)
    assert weights["c2"] == pytest.approx(0.5)

    s.update(dts[1])
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c1"] == pytest.approx(1.0)
    assert weights["c2"] == pytest.approx(0.0)

    s.update(dts[2])
    assert not algo(s)


def test_weigh_inv_vol():
    algo = algos.WeighInvVol(lookback=pd.DateOffset(days=5))

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    # high vol c1
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[2], "c1"] = 95
    data.loc[dts[3], "c1"] = 105
    data.loc[dts[4], "c1"] = 95

    # low vol c2
    data.loc[dts[1], "c2"] = 100.1
    data.loc[dts[2], "c2"] = 99.9
    data.loc[dts[3], "c2"] = 100.1
    data.loc[dts[4], "c2"] = 99.9

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c2"] > weights["c1"]
    assert weights["c1"] == pytest.approx(0.020, 3)
    assert weights["c2"] == pytest.approx(0.980, 3)


def _make_inv_vol_missing_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=9)
    returns = pd.DataFrame(
        {
            "A": [0, 0.08, -0.04, 0.01, 0.06, -0.03, 0.02, -0.01, 0.03],
            "B": [0, 0.01, 0.015, -0.005, 0.012, 0.008, -0.004, 0.009, 0.006],
            "C": [0, 0.03, -0.02, 0.01, 0.04, -0.01, 0.02, -0.015, 0.025],
        },
        index=dates,
    )
    prices = 100 * (1 + returns).cumprod()

    # Combine complete, gappy, all-missing, and zero-volatility columns.
    prices["D"] = np.nan
    prices["E"] = 100.0
    prices.loc[dates[4], ["C", "E"]] = np.nan
    return prices


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_weigh_inv_vol_missing_data_is_column_local(dtype: str):
    prices = _make_inv_vol_missing_data().astype(dtype)
    strategy = bt.Strategy("s")
    strategy.setup(prices)
    strategy.update(prices.index[-1])
    strategy.temp["selected"] = list(prices.columns)

    algo = algos.WeighInvVol(lookback=pd.DateOffset(days=20), lag=pd.DateOffset(days=0))
    assert algo(strategy)

    # Calculate the sample volatility from each asset's explicit valid returns.
    return_samples = [
        [0.08, -0.04, 0.01, 0.06, -0.03, 0.02, -0.01, 0.03],
        [0.01, 0.015, -0.005, 0.012, 0.008, -0.004, 0.009, 0.006],
        [0.03, -0.02, 0.01, 0.02, -0.015, 0.025],
    ]
    expected = 1.0 / np.array([np.std(sample, ddof=1) for sample in return_samples])
    expected /= expected.sum()

    # Missing-only and zero-volatility assets remain excluded from the weights.
    weights = strategy.temp["weights"]
    assert isinstance(weights, pd.Series)
    assert weights.index.tolist() == ["A", "B", "C"]
    np.testing.assert_allclose(weights.to_numpy(dtype=float), expected)


def test_weigh_inv_vol_missing_data_reaches_rebalance():
    prices = _make_inv_vol_missing_data()
    final_weights = {}

    # Compare equivalent public stacks with and without the unrelated gappy asset.
    for columns in (["A", "B"], ["A", "B", "C"]):
        strategy = bt.Strategy(
            "s",
            [
                algos.RunOnDate(prices.index[-1]),
                algos.SelectAll(),
                algos.WeighInvVol(lookback=pd.DateOffset(days=20), lag=pd.DateOffset(days=0)),
                algos.Rebalance(),
            ],
        )
        backtest = bt.Backtest(
            strategy,
            prices.reindex(columns=columns),
            initial_capital=1_000_000,
            integer_positions=False,
            progress_bar=False,
        )
        backtest.run()
        final_weights[tuple(columns)] = {name: backtest.strategy.children[name].weight for name in columns}

    # Adding C may change normalization, but not A's share of unchanged A and B.
    weights_ab = final_weights[("A", "B")]
    weights_abc = final_weights[("A", "B", "C")]
    share_ab = weights_ab["A"] / (weights_ab["A"] + weights_ab["B"])
    share_abc = weights_abc["A"] / (weights_abc["A"] + weights_abc["B"])
    assert share_abc == pytest.approx(share_ab)


@mock.patch("ffn.calc_mean_var_weights")
def test_weigh_mean_var(mock_mv):
    algo = algos.WeighMeanVar(lookback=pd.DateOffset(days=5))

    mock_mv.return_value = pd.Series({"c1": 0.3, "c2": 0.7})

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    assert mock_mv.called
    rets = mock_mv.call_args[0][0]
    assert len(rets) == 4
    assert "c1" in rets
    assert "c2" in rets

    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c1"] == pytest.approx(0.3)
    assert weights["c2"] == pytest.approx(0.7)


def test_weigh_randomly():
    s = bt.Strategy("s")
    s.temp["selected"] = ["c1", "c2", "c3"]

    algo = algos.WeighRandomly()
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 3
    assert sum(weights.values()) == pytest.approx(1.0)

    algo = algos.WeighRandomly((0.3, 0.5), 0.95)
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 3
    assert sum(weights.values()) == pytest.approx(0.95)
    for c in s.temp["selected"]:
        assert weights[c] <= 0.5
        assert weights[c] >= 0.3


@pytest.mark.parametrize("bounds", [(0.6, 1.0), (0.0, 0.4), (0.8, 0.2)])
def test_weigh_randomly_rejects_infeasible_bounds_without_changing_weights(bounds):
    strategy = bt.Strategy("s")
    strategy.temp["selected"] = ["c1", "c2"]
    original_weights = {"c1": 0.5, "c2": 0.5}
    strategy.temp["weights"] = original_weights

    # Rejection at the weight owner must precede replacement of valid targets.
    with pytest.raises(ValueError):
        algos.WeighRandomly(bounds)(strategy)

    assert strategy.temp["weights"] is original_weights


def test_weigh_randomly_rejection_prevents_rebalance_liquidation():
    dates = pd.date_range("2026-01-01", periods=1)
    prices = pd.DataFrame({"c1": [100.0], "c2": [100.0]}, index=dates)
    strategy = bt.Strategy("s", children=["c1", "c2"])
    strategy.setup(prices)
    strategy.update(dates[0])
    strategy.adjust(1000.0)
    strategy.allocate(500.0, "c1")
    strategy.allocate(500.0, "c2")
    strategy.temp["selected"] = ["c1", "c2"]
    original_weights = {"c1": 0.5, "c2": 0.5}
    strategy.temp["weights"] = original_weights

    # The real weighting-to-rebalance chain must stop before selling either holding.
    stack = bt.AlgoStack(algos.WeighRandomly((0.6, 1.0)), algos.Rebalance())
    with pytest.raises(ValueError, match="solution not possible"):
        stack(strategy)

    assert strategy.temp["weights"] is original_weights
    assert strategy["c1"].position == 5.0
    assert strategy["c2"].position == 5.0
    assert strategy.capital == 0.0
    assert strategy.value == 1000.0


def test_weigh_randomly_accepts_empty_selection():
    strategy = bt.Strategy("s")
    strategy.temp["selected"] = []

    assert algos.WeighRandomly()(strategy)
    assert strategy.temp["weights"] == {}


@pytest.mark.parametrize("bounds", [(0.5, 1.0), (0.0, 0.5)])
def test_weigh_randomly_accepts_exact_feasibility_boundary(bounds):
    strategy = bt.Strategy("s")
    strategy.temp["selected"] = ["c1", "c2"]

    assert algos.WeighRandomly(bounds)(strategy)
    assert strategy.temp["weights"] == pytest.approx({"c1": 0.5, "c2": 0.5})


def test_set_stat():
    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    stat = pd.DataFrame(index=dts, columns=["c1", "c2"], data=4.0)
    stat.loc[dts[1], "c1"] = 5.0
    stat.loc[dts[1], "c2"] = 6.0

    algo = algos.SetStat("test_stat")

    s.setup(data, test_stat=stat)
    s.update(dts[0])
    print()
    print(s.get_data("test_stat"))
    assert algo(s)
    stat = s.temp["stat"]
    assert stat["c1"] == pytest.approx(4.0)
    assert stat["c2"] == pytest.approx(4.0)

    s.update(dts[1])
    assert algo(s)
    stat = s.temp["stat"]
    assert stat["c1"] == pytest.approx(5.0)
    assert stat["c2"] == pytest.approx(6.0)


def test_set_stat_legacy():
    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    stat = pd.DataFrame(index=dts, columns=["c1", "c2"], data=4.0)
    stat.loc[dts[1], "c1"] = 5.0
    stat.loc[dts[1], "c2"] = 6.0

    algo = algos.SetStat(stat)

    s.setup(data)
    s.update(dts[0])
    assert algo(s)
    stat = s.temp["stat"]
    assert stat["c1"] == pytest.approx(4.0)
    assert stat["c2"] == pytest.approx(4.0)

    s.update(dts[1])
    assert algo(s)
    stat = s.temp["stat"]
    assert stat["c1"] == pytest.approx(5.0)
    assert stat["c2"] == pytest.approx(6.0)


def test_stat_total_return():
    algo = algos.StatTotalReturn(lookback=pd.DateOffset(days=3))

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    stat = s.temp["stat"]
    assert len(stat) == 2
    assert stat["c1"] == pytest.approx(105.0 / 100 - 1)
    assert stat["c2"] == pytest.approx(95.0 / 100 - 1)


def test_stat_multi_period_return():
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(
        {
            "c1": [100.0, 100.0, 100.0, 100.0, 130.0],
            "c2": [50.0, 100.0, 100.0, 100.0, 110.0],
        },
        index=dts,
    )
    strategy = bt.Strategy("s")
    strategy.setup(data)
    strategy.update(dts[-1])
    strategy.temp["selected"] = ["c1", "c2"]
    algo = algos.StatMultiPeriodReturn(lookbacks=[pd.DateOffset(days=1), pd.DateOffset(days=4)])

    assert algo(strategy)
    assert strategy.temp["stat"]["c1"] == pytest.approx(0.3)
    assert strategy.temp["stat"]["c2"] == pytest.approx(0.65)


def test_stat_multi_period_return_validation():
    lookbacks = [pd.DateOffset(days=1), pd.DateOffset(days=2)]

    with pytest.raises(ValueError, match="lookbacks cannot be empty"):
        algos.StatMultiPeriodReturn([])
    with pytest.raises(ValueError, match="same length"):
        algos.StatMultiPeriodReturn(lookbacks, weights=[1])
    with pytest.raises(ValueError, match="non-negative"):
        algos.StatMultiPeriodReturn(lookbacks, weights=[-1, 2])
    with pytest.raises(ValueError, match="sum to zero"):
        algos.StatMultiPeriodReturn(lookbacks, weights=[0, 0])

    algo = algos.StatMultiPeriodReturn(lookbacks, weights=[1e-9, 1e-9])
    assert algo.weights == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param([1.0, np.nan], id="nan"),
        pytest.param([1.0, np.inf], id="positive-infinity"),
        pytest.param([1.0, -np.inf], id="negative-infinity"),
        pytest.param([np.inf, -np.inf], id="opposite-infinities"),
        pytest.param([np.finfo(float).max, np.finfo(float).max], id="non-finite-sum"),
    ],
)
def test_stat_multi_period_return_rejects_non_finite_weights(weights: list[float]):
    lookbacks = [pd.DateOffset(days=1), pd.DateOffset(days=2)]

    # Validate both caller values and their derived sum before normalization.
    with np.errstate(over="raise", invalid="raise"), pytest.raises(ValueError, match="weights and their sum must be finite"):
        algos.StatMultiPeriodReturn(lookbacks, weights=weights)


def test_select_momentum_rejects_non_finite_weights_before_rebalancing():
    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "c1": [50.0, 100.0, 100.0, 100.0, 100.0],
            "c2": [100.0, 100.0, 100.0, 50.0, 100.0],
        },
        index=dates,
    )
    strategy = bt.Strategy("s")
    strategy.setup(prices)
    strategy.update(dates[-1])
    strategy.adjust(2000.0)
    strategy.allocate(1000.0, "c1")
    strategy.allocate(1000.0, "c2")
    strategy.temp["selected"] = ["c1", "c2"]

    # Rejection must precede the empty-selection path that would liquidate both holdings.
    with pytest.raises(ValueError, match="weights and their sum must be finite"):
        algos.SelectMomentum(
            n=1,
            lookback=[pd.DateOffset(days=1), pd.DateOffset(days=4)],
            weights=[1.0, np.nan],
        )(strategy)
        algos.WeighEqually()(strategy)
        algos.Rebalance()(strategy)

    assert strategy.temp["selected"] == ["c1", "c2"]
    assert "stat" not in strategy.temp
    assert "weights" not in strategy.temp
    assert strategy["c1"].position == 10.0
    assert strategy["c2"].position == 10.0
    assert strategy.capital == 0.0
    assert strategy.value == 2000.0


def test_stat_multi_period_return_lag_and_history():
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame({"c1": [100.0, 110.0, 120.0, 130.0, 200.0]}, index=dts)
    strategy = bt.Strategy("s")
    strategy.setup(data)
    strategy.update(dts[-1])
    strategy.temp["selected"] = ["c1"]

    algo = algos.StatMultiPeriodReturn(
        [pd.DateOffset(days=1)],
        lag=pd.DateOffset(days=1),
    )
    assert algo(strategy)
    assert strategy.temp["stat"]["c1"] == pytest.approx(130.0 / 120.0 - 1)

    algo = algos.StatMultiPeriodReturn(
        [pd.DateOffset(days=1)],
        lag=pd.DateOffset(days=5),
    )
    assert not algo(strategy)


def test_select_n():
    algo = algos.SelectN(n=1, sort_descending=True)

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp["stat"] = data.calc_total_return()

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c1" in selected

    algo = algos.SelectN(n=1, sort_descending=False)
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # return 2 we have if all_or_none false
    algo = algos.SelectN(n=3, sort_descending=False)
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # return 0 we have if all_or_none true
    algo = algos.SelectN(n=3, sort_descending=False, all_or_none=True)
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 0


def test_select_n_perc():
    algo = algos.SelectN(n=0.5, sort_descending=True)

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp["stat"] = data.calc_total_return()

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c1" in selected


def test_select_momentum():
    algo = algos.SelectMomentum(n=1, lookback=pd.DateOffset(days=3))

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    actual = s.temp["selected"]
    assert len(actual) == 1
    assert "c1" in actual


def test_select_momentum_multiple_lookbacks():
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(
        {
            "c1": [100.0, 100.0, 100.0, 100.0, 130.0],
            "c2": [50.0, 100.0, 100.0, 100.0, 110.0],
        },
        index=dts,
    )
    strategy = bt.Strategy("s")
    strategy.setup(data)
    strategy.update(dts[-1])
    strategy.temp["selected"] = ["c1", "c2"]
    lookbacks = [pd.DateOffset(days=1), pd.DateOffset(days=4)]

    assert algos.SelectMomentum(n=1, lookback=lookbacks)(strategy)
    assert strategy.temp["selected"] == ["c2"]

    strategy.temp["selected"] = ["c1", "c2"]
    assert algos.SelectMomentum(n=1, lookback=lookbacks, weights=[10, 1])(strategy)
    assert strategy.temp["selected"] == ["c1"]

    strategy.temp["selected"] = ["c1", "c2"]
    lookbacks = np.array([pd.DateOffset(days=4)])
    assert algos.SelectMomentum(n=1, lookback=lookbacks)(strategy)
    assert strategy.temp["selected"] == ["c2"]

    with pytest.raises(ValueError, match="weights require multiple"):
        algos.SelectMomentum(n=1, weights=[1])


def test_limit_weights():

    s = bt.Strategy("s")
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.temp["weights"] = {"c1": 0.6, "c2": 0.2, "c3": 0.2}

    algo = algos.LimitWeights(0.5)
    assert algo(s)
    w = s.temp["weights"]
    assert w["c1"] == pytest.approx(0.5)
    assert w["c2"] == pytest.approx(0.25)
    assert w["c3"] == pytest.approx(0.25)

    algo = algos.LimitWeights(0.3)
    assert algo(s)
    w = s.temp["weights"]
    assert w == {}

    s.temp["weights"] = {"c1": 0.4, "c2": 0.3, "c3": 0.3}
    algo = algos.LimitWeights(0.5)
    assert algo(s)
    w = s.temp["weights"]
    assert w["c1"] == pytest.approx(0.4)
    assert w["c2"] == pytest.approx(0.3)
    assert w["c3"] == pytest.approx(0.3)


def test_limit_deltas():
    s = bt.Strategy("s")
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.temp["weights"] = {"c1": 1}

    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp["weights"]
    assert w["c1"] == pytest.approx(0.1)

    s.temp["weights"] = {"c1": 0.05}
    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp["weights"]
    assert w["c1"] == pytest.approx(0.05)

    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}
    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp["weights"]
    assert len(w) == 2
    assert w["c1"] == pytest.approx(0.1)
    assert w["c2"] == pytest.approx(0.1)

    s.temp["weights"] = {"c1": 0.5, "c2": -0.5}
    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp["weights"]
    assert len(w) == 2
    assert w["c1"] == pytest.approx(0.1)
    assert w["c2"] == pytest.approx(-0.1)

    s.temp["weights"] = {"c1": 0.5, "c2": -0.5}
    algo = algos.LimitDeltas({"c1": 0.1})
    assert algo(s)
    w = s.temp["weights"]
    assert len(w) == 2
    assert w["c1"] == pytest.approx(0.1)
    assert w["c2"] == pytest.approx(-0.5)

    s.temp["weights"] = {"c1": 0.5, "c2": -0.5}
    algo = algos.LimitDeltas({"c1": 0.1, "c2": 0.3})
    assert algo(s)
    w = s.temp["weights"]
    assert len(w) == 2
    assert w["c1"] == pytest.approx(0.1)
    assert w["c2"] == pytest.approx(-0.3)

    # set exisitng weight
    s.children["c1"] = bt.core.SecurityBase("c1")
    s.children["c1"]._weight = 0.3
    s.children["c2"] = bt.core.SecurityBase("c2")
    s.children["c2"]._weight = -0.7

    s.temp["weights"] = {"c1": 0.5, "c2": -0.5}
    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp["weights"]
    assert len(w) == 2
    assert w["c1"] == pytest.approx(0.4)
    assert w["c2"] == pytest.approx(-0.6)


def test_rebalance_over_time():
    target = mock.MagicMock()
    rb = mock.MagicMock()

    algo = algos.RebalanceOverTime(n=2)
    # patch in rb function
    algo._rb = rb

    target.temp = {}
    target.temp["weights"] = {"a": 1, "b": 0}

    a = mock.MagicMock()
    a.weight = 0.0
    b = mock.MagicMock()
    b.weight = 1.0
    target.children = {"a": a, "b": b}

    assert algo(target)
    w = target.temp["weights"]
    assert len(w) == 2
    assert w["a"] == pytest.approx(0.5)
    assert w["b"] == pytest.approx(0.5)

    assert rb.called
    called_tgt = rb.call_args[0][0]
    called_tgt_w = called_tgt.temp["weights"]
    assert len(called_tgt_w) == 2
    assert called_tgt_w["a"] == pytest.approx(0.5)
    assert called_tgt_w["b"] == pytest.approx(0.5)

    # update weights for next call
    a.weight = 0.5
    b.weight = 0.5

    # clear out temp - same as would Strategy
    target.temp = {}

    assert algo(target)
    w = target.temp["weights"]
    assert len(w) == 2
    assert w["a"] == pytest.approx(1.0)
    assert w["b"] == pytest.approx(0.0)

    assert rb.call_count == 2

    # update weights for next call
    # should do nothing now
    a.weight = 1
    b.weight = 0

    # clear out temp - same as would Strategy
    target.temp = {}

    assert algo(target)
    # no diff in call_count since last time
    assert rb.call_count == 2


def test_rebalance_over_time_phases_out_omitted_target():
    target = mock.MagicMock()
    rb = mock.MagicMock()

    algo = algos.RebalanceOverTime(n=2)
    algo._rb = rb

    target.temp = {"weights": {"a": 1.0}}
    a = mock.MagicMock()
    a.weight = 0.5
    b = mock.MagicMock()
    b.weight = 0.5
    target.children = {"a": a, "b": b}

    assert algo(target)

    # Sparse and explicit-zero targets must follow the same interpolation path.
    assert list(target.temp["weights"]) == ["a", "b"]
    assert target.temp["weights"] == pytest.approx({"a": 0.75, "b": 0.25})
    assert rb.call_args[0][0] is target


@pytest.mark.parametrize("hedge_type", [bt.HedgeSecurity, bt.CouponPayingHedgeSecurity])
@pytest.mark.parametrize("close_hedge", [False, True])
def test_rebalance_over_time_preserves_omitted_hedges(hedge_type, close_hedge):
    dates = pd.date_range("2020-01-01", periods=2)
    prices = pd.DataFrame(100.0, index=dates, columns=["a", "b", "hedge"])
    strategy = bt.FixedIncomeStrategy(
        "s", children=[bt.CouponPayingSecurity("a"), bt.CouponPayingSecurity("b"), hedge_type("hedge")]
    )
    strategy.use_integer_positions(False)
    strategy.setup(prices, coupons=prices * 0.0)
    strategy.adjust(10000.0)
    strategy.update(dates[0])
    strategy["a"].transact(5)
    strategy["b"].transact(5)
    strategy["hedge"].transact(-2)
    strategy.update(dates[0])
    assert strategy["hedge"].weight == 0.0

    algo = algos.RebalanceOverTime(n=2)
    strategy.temp["weights"] = {"a": 1.0}
    if close_hedge:
        strategy.temp["weights"]["hedge"] = 0.0
    assert algo(strategy)
    assert strategy["a"].position == pytest.approx(7.5)
    assert strategy["b"].position == pytest.approx(2.5)
    assert strategy["hedge"].position == (0 if close_hedge else -2)

    strategy.temp = {}
    strategy.update(dates[1])
    assert algo(strategy)
    assert strategy["a"].position == pytest.approx(10.0)
    assert strategy["b"].position == 0.0
    assert strategy["hedge"].position == (0 if close_hedge else -2)


def test_rebalance_over_time_supports_sparse_weigh_target():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {"a": [100.0, 100.0, 100.0, 100.0], "b": [100.0, 100.0, 100.0, 200.0]},
        index=dates,
    )
    targets = pd.DataFrame({"a": [0.5, 1.0], "b": [0.5, np.nan]}, index=dates[[0, 2]])
    strategy = bt.Strategy(
        "s",
        [
            algos.WeighTarget("targets"),
            algos.run_always(algos.RebalanceOverTime(n=2)),
        ],
    )
    backtest = bt.Backtest(
        strategy,
        prices,
        initial_capital=1000.0,
        integer_positions=False,
        additional_data={"targets": targets},
    )

    bt.run(backtest)

    # The first plan reaches 50/50 before the sparse target starts phasing out b.
    a = backtest.strategy.children["a"]
    b = backtest.strategy.children["b"]
    assert [a.positions.loc[dates[1]], b.positions.loc[dates[1]]] == pytest.approx([5.0, 5.0])
    assert [a.positions.loc[dates[2]], b.positions.loc[dates[2]]] == pytest.approx([7.5, 2.5])
    assert backtest.strategy.cash.loc[dates[2]] == pytest.approx(0.0)

    # Holding 2.5 shares of b through its price change independently yields 1,250.
    assert backtest.strategy.value == pytest.approx(1250.0)
    assert backtest.strategy.price == pytest.approx(125.0)


@pytest.mark.parametrize(
    ("cash", "initial_position", "target_weight", "n"),
    [
        pytest.param(0.5, 5.0, 1.0, 2, id="matched"),
        pytest.param(0.5, 2.0, 1.0, 2, id="increase"),
        pytest.param(0.5, 5.0, 0.4, 2, id="decrease"),
        pytest.param(0.5, 5.0, None, 2, id="omitted"),
        pytest.param(0.5, 5.0, 0.0, 2, id="explicit-zero"),
        pytest.param(0.5, 5.0, 0.4, 1, id="one-period"),
        pytest.param(0.0, 5.0, 1.0, 2, id="no-reserve"),
        pytest.param(1.0, 5.0, 1.0, 2, id="full-cash-transition"),
    ],
)
def test_rebalance_over_time_cash_coordinates(cash, initial_position, target_weight, n):
    dates = pd.date_range("2020-01-01", periods=n + 2)
    strategy = bt.Strategy("s", children=[bt.Security("x")])
    strategy.use_integer_positions(False)
    strategy.setup(pd.DataFrame(100.0, index=dates, columns=["x"]))
    strategy.adjust(1000.0)
    strategy.update(dates[0])
    strategy["x"].transact(initial_position)
    strategy.update(dates[0])
    weights = {} if target_weight is None else {"x": target_weight}
    original_weights = weights.copy()
    algo = algos.RebalanceOverTime(n=n)

    # Current weights use total NAV; targets use the slice left after reserving cash.
    initial_weight = initial_position * 100.0 / 1000.0
    final_weight = (1.0 - cash) * (target_weight or 0.0)
    for phase in range(1, n + 1):
        strategy.temp = {"cash": cash}
        strategy.update(dates[phase])
        if phase == 1:
            strategy.temp["weights"] = weights
        assert algo(strategy)
        expected = initial_weight + (final_weight - initial_weight) * phase / n
        # Rebalance's established full-cash path closes immediately, not gradually.
        if cash == 1.0:
            expected = 0.0
        assert strategy["x"].weight == pytest.approx(expected)
        assert strategy["x"].position == pytest.approx(expected * 10.0)
        assert strategy.capital == pytest.approx(1000.0 * (1.0 - expected))
        assert strategy.value == pytest.approx(1000.0)
        assert weights == original_weights

    strategy.temp = {"cash": cash}
    strategy.update(dates[-1])
    assert algo(strategy)
    assert "weights" not in strategy.temp


@pytest.mark.parametrize(("later_price", "fee"), [(100.0, 1.0), (102.0, 0.0), (98.0, 0.0)])
def test_rebalance_over_time_matched_cash_target_backtest(later_price, fee):
    class CashPlan(bt.Algo):
        def __call__(self, target):
            target.temp["cash"] = 0.5
            if target.now == dates[0]:
                target.transact(500.0, "x")
            elif target.now == dates[1]:
                target.temp["weights"] = {"x": 1.0}
            return True

    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"x": [100.0, 100.0, later_price, later_price]}, index=dates)
    strategy = bt.Strategy("s", [CashPlan(), algos.RebalanceOverTime(n=2)])
    # Keep the initial holding exactly at its cash-scaled target even with fees enabled.
    backtest = bt.Backtest(
        strategy,
        prices,
        initial_capital=100000.0 + fee,
        commissions=lambda q, p: fee,
        integer_positions=False,
        progress_bar=False,
    )
    backtest.run()
    s = backtest.strategy
    assert s["x"].positions.loc[dates[1]] == pytest.approx(500.0)
    assert s.cash.loc[dates[1]] == pytest.approx(50000.0)
    # This market move occurs before the second phase: 500 shares must stay exposed.
    assert s.values.loc[dates[2]] == pytest.approx(50000.0 + 500.0 * later_price)
    assert s.fees.loc[dates[1]] == pytest.approx(0.0)
    assert s.fees.loc[dates[2]] == pytest.approx(0.0)


def test_require():
    target = mock.MagicMock()
    target.temp = {}

    algo = algos.Require(lambda x: len(x) > 0, "selected")
    assert not algo(target)

    target.temp["selected"] = []
    assert not algo(target)

    target.temp["selected"] = ["a", "b"]
    assert algo(target)


def test_run_every_n_periods():
    target = mock.MagicMock()
    target.temp = {}

    algo = algos.RunEveryNPeriods(n=3, offset=0)

    target.now = pd.to_datetime("2010-01-01")
    assert algo(target)
    # run again w/ no date change should not trigger
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-02")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-03")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-04")
    assert algo(target)

    target.now = pd.to_datetime("2010-01-05")
    assert not algo(target)


def test_run_every_n_periods_offset():
    target = mock.MagicMock()
    target.temp = {}

    algo = algos.RunEveryNPeriods(n=3, offset=1)

    target.now = pd.to_datetime("2010-01-01")
    assert not algo(target)
    # run again w/ no date change should not trigger
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-02")
    assert algo(target)

    target.now = pd.to_datetime("2010-01-03")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-04")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-05")
    assert algo(target)


def test_not():
    target = mock.MagicMock()
    target.temp = {}

    # run except on the 1/2/18
    runOnDateAlgo = algos.RunOnDate(pd.to_datetime("2018-01-02"))
    notAlgo = algos.Not(runOnDateAlgo)

    target.now = pd.to_datetime("2018-01-01")
    assert notAlgo(target)

    target.now = pd.to_datetime("2018-01-02")
    assert not notAlgo(target)


def test_or():
    target = mock.MagicMock()
    target.temp = {}

    # run on the 1/2/18
    runOnDateAlgo = algos.RunOnDate(pd.to_datetime("2018-01-02"))
    runOnDateAlgo2 = algos.RunOnDate(pd.to_datetime("2018-01-03"))
    runOnDateAlgo3 = algos.RunOnDate(pd.to_datetime("2018-01-04"))
    runOnDateAlgo4 = algos.RunOnDate(pd.to_datetime("2018-01-04"))

    orAlgo = algos.Or([runOnDateAlgo, runOnDateAlgo2, runOnDateAlgo3, runOnDateAlgo4])

    # verify it returns false when neither is true
    target.now = pd.to_datetime("2018-01-01")
    assert not orAlgo(target)

    # verify it returns true when the first is true
    target.now = pd.to_datetime("2018-01-02")
    assert orAlgo(target)

    # verify it returns true when the second is true
    target.now = pd.to_datetime("2018-01-03")
    assert orAlgo(target)

    # verify it returns true when both algos return true
    target.now = pd.to_datetime("2018-01-04")
    assert orAlgo(target)


@pytest.mark.parametrize("covar_method", ["standard", "ledoit-wolf"])
def test_TargetVol(covar_method):

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=7)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    # high vol c1
    data.loc[dts[0], "c1"] = 95
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[2], "c1"] = 95
    data.loc[dts[3], "c1"] = 105
    data.loc[dts[4], "c1"] = 95
    data.loc[dts[5], "c1"] = 105
    data.loc[dts[6], "c1"] = 95

    # low vol c2
    data.loc[dts[0], "c2"] = 99
    data.loc[dts[1], "c2"] = 101
    data.loc[dts[2], "c2"] = 99
    data.loc[dts[3], "c2"] = 101
    data.loc[dts[4], "c2"] = 99
    data.loc[dts[5], "c2"] = 101
    data.loc[dts[6], "c2"] = 99
    data["c3"] = data["c2"]

    targetVolAlgo = algos.TargetVol(
        0.1,
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=1),
        covar_method=covar_method,
        annualization_factor=1,
    )

    s.setup(data)
    s.update(dts[6])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    assert targetVolAlgo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert np.isclose(weights["c2"], weights["c1"])

    unannualized_c2_weight = weights["c1"]

    s.temp["weights"] = {"c3": 1.0}
    assert targetVolAlgo(s)
    assert targetVolAlgo.target_volatility == 0.1
    assert not np.isclose(s.temp["weights"]["c3"], 1.0)

    targetVolAlgo = algos.TargetVol(
        0.1 * np.sqrt(252),
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=1),
        covar_method=covar_method,
        annualization_factor=252,
    )

    s.setup(data)
    s.update(dts[6])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    assert targetVolAlgo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert np.isclose(weights["c2"], weights["c1"])

    assert np.isclose(unannualized_c2_weight, weights["c2"])


@pytest.mark.parametrize("covar_method", ["standard", "ledoit-wolf"])
@pytest.mark.parametrize(
    ("target_volatility", "expected_error"),
    [
        (0.1, "zero-volatility"),
        ({"c1": 0.1}, "zero-volatility"),
        (0.0, None),
        ({"c1": 0.0}, None),
        ({"c2": 0.1}, None),
    ],
)
def test_TargetVol_zero_volatility(
    covar_method: str,
    target_volatility: "float | dict[str, float]",
    expected_error: "str | None",
):
    s = bt.Strategy("s")
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)

    target_vol_algo = algos.TargetVol(
        target_volatility,
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
        covar_method=covar_method,
        annualization_factor=1,
    )

    s.setup(data)
    s.update(dts[-1])
    initial_weights = {"c1": 1.0}
    s.temp["weights"] = initial_weights.copy()

    # Preserve satisfied or omitted targets, and reject infeasible targets before mutation.
    if expected_error is None:
        assert target_vol_algo(s)
    else:
        with pytest.raises(ValueError, match=expected_error):
            target_vol_algo(s)
    assert s.temp["weights"] == initial_weights


def test_TargetVol_rejects_non_finite_volatility():
    s = bt.Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    target_vol_algo = algos.TargetVol(
        0.1,
        lookback=pd.DateOffset(days=1),
        lag=pd.DateOffset(days=0),
        annualization_factor=1,
    )

    s.setup(data)
    s.update(dts[-1])
    s.temp["weights"] = {"c1": 1.0}

    with pytest.raises(ValueError, match="non-finite"):
        target_vol_algo(s)


@pytest.mark.parametrize("covar_method", ["standard", "ledoit-wolf"])
@pytest.mark.parametrize("target_volatility", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("use_mapping", [False, True])
def test_TargetVol_rejects_non_finite_target(covar_method, target_volatility, use_mapping):
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame({"c1": [100.0, 102.0, 99.0, 103.0, 100.0]}, index=dts)
    data["c2"] = data["c1"]
    s = bt.Strategy("s")
    s.setup(data)
    s.update(dts[-1])
    initial_weights = {"c1": 0.5, "c2": 0.5}
    s.temp["weights"] = initial_weights.copy()
    if use_mapping:
        target_volatility = {"c1": 0.1, "c2": target_volatility}
    target_vol_algo = algos.TargetVol(
        target_volatility,
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
        covar_method=covar_method,
    )

    with pytest.raises(ValueError, match="non-finite target volatility"):
        target_vol_algo(s)
    assert s.temp["weights"] == initial_weights


@pytest.mark.parametrize("covar_method", ["standard", "ledoit-wolf"])
@pytest.mark.parametrize("integer_positions", [False, True])
def test_TargetVol_zero_volatility_backtest(covar_method, integer_positions):
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame({"c1": 100.0}, index=dts)
    s = bt.Strategy(
        "s",
        [
            algos.RunAfterDate(dts[-2]),
            algos.WeighSpecified(c1=1.0),
            algos.TargetVol(
                0.1,
                lookback=pd.DateOffset(days=4),
                lag=pd.DateOffset(days=0),
                covar_method=covar_method,
            ),
            algos.Rebalance(),
        ],
    )
    backtest = bt.Backtest(s, data, integer_positions=integer_positions, progress_bar=False)

    with pytest.raises(ValueError, match="zero-volatility"):
        backtest.run()


@pytest.mark.parametrize("covar_method", ["standard", "ledoit-wolf"])
def test_PTE_Rebalance(covar_method):

    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=30 * 4)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    # high vol c1
    # low vol c2
    for i, dt in enumerate(dts[:-2]):
        if i % 2 == 0:
            data.loc[dt, "c1"] = 95
            data.loc[dt, "c2"] = 101
        else:
            data.loc[dt, "c1"] = 105
            data.loc[dt, "c2"] = 99

    dt = dts[-2]
    data.loc[dt, "c1"] = 115
    data.loc[dt, "c2"] = 97

    s.setup(data)
    s.update(dts[-2])
    s.adjust(1000000)
    s.rebalance(0.4, "c1")
    s.rebalance(0.6, "c2")

    wdf = pd.DataFrame(np.zeros(data.shape), columns=data.columns, index=data.index)

    wdf["c1"] = 0.5
    wdf["c2"] = 0.5

    PTE_rebalance_Algo = bt.algos.PTE_Rebalance(
        0.01,
        wdf,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=1),
        covar_method=covar_method,
        annualization_factor=252,
    )

    assert PTE_rebalance_Algo(s)

    s.rebalance(0.5, "c1")
    s.rebalance(0.5, "c2")

    assert not PTE_rebalance_Algo(s)


@pytest.mark.parametrize("multiplier", [1, 10])
def test_PTE_Rebalance_security_multiplier(multiplier: int):
    dates = pd.date_range("2010-01-01", periods=8)
    data = pd.DataFrame(
        {
            "c1": [100, 105, 95, 110, 90, 115, 105, 120],
            "c2": [100, 99, 101, 98, 102, 97, 103, 100],
        },
        index=dates,
    )
    strategy = bt.Strategy(
        "s",
        children=[bt.Security("c1", multiplier=multiplier), bt.Security("c2")],
    )
    strategy.use_integer_positions(False)
    strategy.setup(data)
    strategy.update(dates[-1])
    strategy.adjust(1_000_000)
    strategy.rebalance(0.4, "c1")
    strategy.rebalance(0.6, "c2")
    target_weights = pd.DataFrame({"c1": 0.4, "c2": 0.6}, index=dates)
    algo = bt.algos.PTE_Rebalance(
        0.01,
        target_weights,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=1),
    )

    # Independently reconstruct economic weights from quantities and contract sizes.
    c1_weight: float = strategy["c1"].position * 120 * multiplier / strategy.value
    c2_weight: float = strategy["c2"].position * 100 / strategy.value
    assert c1_weight == pytest.approx(0.4)
    assert c2_weight == pytest.approx(0.6)
    assert not algo(strategy)


@pytest.mark.parametrize("covar_method", ["standard", "ledoit-wolf"])
@pytest.mark.parametrize("lazy", [False, True], ids=["explicit", "lazy"])
@pytest.mark.parametrize(
    ("multiplier", "reverse", "large_weight", "quote_factor", "target_weight", "should_rebalance"),
    [
        (1, False, 1.0, 1.0, 1.0, False),
        (10, False, 1.0, 1.0, 1.0, False),
        (10, True, 1.0, 1.0, 1.0, False),
        (10, True, 1.0, 1.0, 0.55, True),
        (10, False, -10.0, 1.0, -4.5, False),
        (10, False, 1.0, 1.1, 1.1, False),
    ],
)
def test_PTE_Rebalance_nested_multipliers(covar_method, lazy, multiplier, reverse, large_weight, quote_factor, target_weight, should_rebalance):
    dates = pd.date_range("2020-01-01", periods=9)
    quotes = np.array([100, 104, 99, 108, 102, 112, 105, 118, 110], dtype=float)
    data = pd.DataFrame({"a": quotes}, index=dates)
    sleeves = [
        bt.Strategy("small", children=[bt.Security("a", multiplier=1, lazy_add=lazy)]),
        bt.Strategy("large", children=[bt.Security("a", multiplier=multiplier, lazy_add=lazy)]),
    ]
    if reverse:
        sleeves.reverse()
    # Retain the underlying quote in the parent's otherwise sleeve-only universe.
    strategy = bt.Strategy("root", children=sleeves + [bt.Security("a", lazy_add=True)])
    strategy.use_integer_positions(False)
    strategy.setup(data)
    strategy.update(dates[-1])
    strategy.adjust(100_000)
    for name in ["small", "large"]:
        strategy.allocate(50_000, name)
        strategy[name].rebalance(1.0 if name == "small" else large_weight, "a")

    # Each sleeve owns half the NAV; allocations, not summed contract counts, define exposure.
    # The short case cancels raw quantities while retaining nonzero economic exposure.
    assert strategy.value == pytest.approx(100_000)
    for name, contract_size, weight in [("small", 1, 1.0), ("large", multiplier, large_weight)]:
        assert strategy[name]["a"].position == pytest.approx(50_000 * weight / (110 * contract_size))
    quantities = strategy.positions.copy()
    if quote_factor != 1.0:
        # PTE reads the current universe quote, even when the leaf's cached value differs.
        strategy.universe.at[dates[-1], "a"] = 110 * quote_factor
    target_weights = pd.DataFrame({"a": target_weight}, index=dates)
    algo = bt.algos.PTE_Rebalance(
        0.01,
        target_weights,
        lookback=pd.DateOffset(days=8),
        covar_method=covar_method,
    )

    # One asset makes the covariance oracle elementary: sample variance for standard,
    # population variance for one-dimensional Ledoit-Wolf (shrinkage has no effect).
    returns = quotes[1:] / quotes[:-1] - 1
    variance = np.var(returns, ddof=1 if covar_method == "standard" else 0)
    expected_weight = 0.5 * (1.0 + large_weight) * quote_factor
    expected_pte = abs(expected_weight - target_weight) * np.sqrt(252 * variance)
    assert (expected_pte > 0.01) == should_rebalance
    with np.errstate(all="raise"):
        assert bool(algo(strategy)) == should_rebalance
    pd.testing.assert_frame_equal(strategy.positions, quantities)
    pd.testing.assert_frame_equal(target_weights, pd.DataFrame({"a": target_weight}, index=dates))
    assert strategy.value == pytest.approx(100_000)


def test_TargetVol_standard_uses_pairwise_covariance():
    s = bt.Strategy("s")

    dts = pd.date_range("2010-01-01", periods=8)
    data = pd.DataFrame(
        {
            "c1": [100, 110, 99, 115, 103, 120, 108, 125],
            "c2": [100, 102, 104, np.nan, 107, 109, 111, 114],
        },
        index=dts,
    )

    targetVolAlgo = algos.TargetVol(
        0.1,
        lookback=pd.DateOffset(days=7),
        covar_method="standard",
        annualization_factor=1,
    )

    s.setup(data)
    s.update(dts[-1])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    returns = bt.ffn.to_returns(data)
    weights = np.array([0.5, 0.5])
    expected_vol = np.sqrt(weights.T @ returns.cov().values @ weights)

    assert targetVolAlgo(s)
    assert s.temp["weights"]["c1"] == pytest.approx(0.5 * 0.1 / expected_vol)


def test_close_dead_closes_zero_price_position():
    dates = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame({"asset": [100.0, 0.0]}, index=dates)
    strategy = bt.Strategy("strategy", children=["asset"])
    strategy.setup(data)
    strategy.update(dates[0])
    strategy.adjust(1000.0)
    strategy.allocate(1000.0, "asset")
    strategy.update(dates[0])
    strategy.update(dates[1])
    strategy.temp["weights"] = {"asset": 1.0}

    assert algos.CloseDead()(strategy)

    assert strategy["asset"].position == 0.0
    assert "asset" not in strategy.temp["weights"]


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_close_dead_preserves_missing_prices_in_mixed_dtype_universe(dtype):
    date = pd.Timestamp("2010-01-01")
    data = pd.DataFrame(
        {"missing": [np.nan], "dead": pd.array([0], dtype=dtype), "live": pd.array([1], dtype=dtype)},
        index=[date],
    )
    strategy = bt.Strategy("strategy", children=[bt.Security(name) for name in data.columns])
    strategy.setup(data)
    strategy.update(date)
    strategy.temp["weights"] = {"missing": 0.5, "dead": 0.25, "live": 0.25}

    # A mixed-dtype row can turn NumPy NaN into pd.NA, whose truth value raises.
    assert algos.CloseDead()(strategy)

    assert strategy.temp["weights"] == {"missing": 0.5, "live": 0.25}
    assert all(child.position == 0 for child in strategy.children.values())


def test_close_dead_observes_price_changes_from_earlier_close():
    target = mock.Mock()
    target.now = pd.Timestamp("2010-01-01")
    target.universe = pd.DataFrame({"first": pd.array([0], dtype="Int64"), "later": [100.0]}, index=[target.now])
    target.children = {"first": None, "later": None}
    target.temp = {"weights": {"first": 0.5, "later": 0.5}}

    # Closing one child may change the market data seen by the next child.
    def close(child):
        target.universe.at[target.now, "later"] = 0.0

    target.close.side_effect = close
    assert algos.CloseDead()(target)

    assert target.close.call_args_list == [mock.call("first"), mock.call("later")]
    assert target.temp["weights"] == {}


@pytest.mark.parametrize("temp", [{}, {"weights": {}}])
def test_close_dead_without_work_does_not_require_market_data(temp):
    target = mock.Mock(spec=["temp", "children"])
    target.temp = temp
    target.children = {}

    assert algos.CloseDead()(target)


def test_close_positions_after_date():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    c3 = bt.Security("c3")
    s = bt.Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    cutoffs = pd.DataFrame({"date": [dts[1], dts[2]]}, index=["c1", "c2"])

    algo = algos.ClosePositionsAfterDates("cutoffs")

    s.setup(data, cutoffs=cutoffs)

    s.update(dts[0])
    s.transact(100, "c1")
    s.transact(100, "c2")
    s.transact(100, "c3")
    algo(s)
    assert c1.position == 100
    assert c2.position == 100
    assert c3.position == 100

    # Don't run anything on dts[1], even though that's when c1 closes
    s.update(dts[2])
    algo(s)
    assert c1.position == 0
    assert c2.position == 0
    assert c3.position == 100
    assert s.perm["closed"] == set(["c1", "c2"])


def test_roll_positions_after_date():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    c3 = bt.Security("c3")
    s = bt.Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    roll = pd.DataFrame(
        {"date": [dts[1], dts[2]], "target": ["c3", "c1"], "factor": [0.5, 2.0]},
        index=["c1", "c2"],
    )

    algo = algos.RollPositionsAfterDates("roll")

    s.setup(data, roll=roll)

    s.update(dts[0])
    s.transact(100, "c1")
    s.transact(100, "c2")
    s.transact(100, "c3")
    algo(s)
    assert c1.position == 100
    assert c2.position == 100
    assert c3.position == 100

    # Don't run anything on dts[1], even though that's when c1 closes
    s.update(dts[2])
    algo(s)
    assert c1.position == 200  # From c2
    assert c2.position == 0
    assert c3.position == 100 + 50
    assert s.perm["rolled"] == set(["c1", "c2"])


def test_roll_positions_after_date_aggregates_finite_factors():
    dates = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(100.0, index=dates, columns=["old_a", "old_b", "new"])
    strategy = bt.FixedIncomeStrategy(
        "strategy",
        children=[
            bt.FixedIncomeSecurity("old_a"),
            bt.FixedIncomeSecurity("old_b"),
            bt.FixedIncomeSecurity("new", lazy_add=True),
        ],
    )
    roll = pd.DataFrame(
        {
            "date": [dates[0], dates[0]],
            "target": ["new", "new"],
            "factor": [0.0, -0.5],
        },
        index=["old_a", "old_b"],
    )
    strategy.setup(data, roll=roll)
    strategy.update(dates[0])
    strategy.transact(10, "old_a")
    strategy.transact(20, "old_b")

    assert algos.RollPositionsAfterDates("roll")(strategy)

    # Both due rows roll atomically into the same lazily-created target.
    assert strategy["old_a"].position == 0
    assert strategy["old_b"].position == 0
    assert strategy["new"].position == -10
    assert strategy.perm["rolled"] == {"old_a", "old_b"}


@pytest.mark.parametrize(
    ("invalid_factor", "old_b_position", "new_position"),
    [
        pytest.param(np.nan, 20, 0, id="nan"),
        pytest.param(np.inf, 20, 0, id="positive-infinity"),
        pytest.param(-np.inf, 20, 0, id="negative-infinity"),
        pytest.param(pd.NA, 20, 0, id="missing"),
        pytest.param(np.finfo(float).max, 20, 0, id="non-finite-result"),
        pytest.param(1e308, 1, 1e308, id="destination-overflow"),
        pytest.param(1e308, -1, -1e308, id="destination-negative-overflow"),
    ],
)
def test_roll_positions_after_date_rejects_non_finite_rolls(invalid_factor, old_b_position, new_position):
    dates = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(100.0, index=dates, columns=["old_a", "old_b", "new"])
    if new_position:
        data["new"] = 0.0
    strategy = bt.FixedIncomeStrategy(
        "strategy",
        children=[
            bt.FixedIncomeSecurity("old_a"),
            bt.FixedIncomeSecurity("old_b"),
            bt.FixedIncomeSecurity("new", lazy_add=True),
        ],
    )
    roll = pd.DataFrame(
        {
            "date": [dates[0], dates[0]],
            "target": ["new", "new"],
            "factor": [0.5, invalid_factor],
        },
        index=["old_a", "old_b"],
    )
    strategy.setup(data, roll=roll)
    strategy.update(dates[0])
    strategy.transact(10, "old_a")
    strategy.transact(old_b_position, "old_b")
    if new_position:
        strategy.transact(new_position, "new")

    # A later invalid row must leave the entire due roll set untouched.
    before_children = tuple(strategy.children)
    before_lazy_children = tuple(strategy._lazy_children)
    before_positions = {name: strategy[name].position for name in before_children}
    before_capital = strategy.capital
    before_value = strategy.value
    before_perm = strategy.perm.copy()
    before_histories = {
        name: (strategy[name].positions.copy(), strategy[name].outlays.copy())
        for name in before_children
    }

    with pytest.raises(ValueError, match="finite"):
        algos.RollPositionsAfterDates("roll")(strategy)

    assert tuple(strategy.children) == before_children
    assert tuple(strategy._lazy_children) == before_lazy_children
    assert {name: strategy[name].position for name in before_children} == before_positions
    assert strategy.capital == before_capital
    assert strategy.value == before_value
    assert strategy.perm == before_perm
    for name, (positions, outlays) in before_histories.items():
        pd.testing.assert_series_equal(strategy[name].positions, positions)
        pd.testing.assert_series_equal(strategy[name].outlays, outlays)


@pytest.mark.parametrize("destination_due", [False, True])
def test_roll_positions_after_date_allows_finite_destination_positions(destination_due):
    dates = pd.date_range("2010-01-01", periods=2)
    strategy = bt.FixedIncomeStrategy("strategy", children=[bt.FixedIncomeSecurity("old"), bt.FixedIncomeSecurity("new")])
    roll = pd.DataFrame({"date": [dates[0]], "target": ["new"], "factor": [1e308]}, index=["old"])
    if destination_due:
        roll.loc["new"] = [dates[0], "old", 0.0]
    strategy.setup(pd.DataFrame(0.0, index=dates, columns=["old", "new"]), roll=roll)
    strategy.update(dates[0])
    strategy.transact(1.0, "old")
    strategy.transact(1e308 if destination_due else -1e308, "new")

    # A destination can offset the transfer or be closed before receiving it.
    assert algos.RollPositionsAfterDates("roll")(strategy)

    assert strategy["old"].position == 0
    assert strategy["new"].position == (1e308 if destination_due else 0.0)
    assert strategy.value == 0
    assert strategy.perm["rolled"] == ({"old", "new"} if destination_due else {"old"})


def test_replay_transactions():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    s = bt.Strategy("s", children=[c1, c2])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]

    transactions = pd.DataFrame(
        [
            (pd.Timestamp("2009-12-01 00"), "c1", 100, 99.5),
            (pd.Timestamp("2010-01-01 10"), "c1", -100, 101),
            (pd.Timestamp("2010-01-02 00"), "c2", 50, 103),
        ],
        columns=["Date", "Security", "quantity", "price"],
    )
    transactions = transactions.set_index(["Date", "Security"])

    algo = algos.ReplayTransactions("transactions")
    s.setup(
        data, bidoffer={}, transactions=transactions
    )  # Pass bidoffer so it will track bidoffer paid
    s.adjust(1000)
    s.update(dts[0])
    algo(s)
    assert c1.position == 100
    assert c2.position == 0
    assert c1.bidoffer_paid == -50

    s.update(dts[1])
    algo(s)
    assert c1.position == 0
    assert c2.position == 50
    assert c1.bidoffer_paid == -100
    assert c2.bidoffer_paid == 150


def test_replay_transactions_consistency():
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)

    transactions = pd.DataFrame(
        [
            (pd.Timestamp("2010-01-01 00"), "c1", -100.0, 101.0),
            (pd.Timestamp("2010-01-02 00"), "c2", 50.0, 103.0),
        ],
        columns=["Date", "Security", "quantity", "price"],
    )
    transactions = transactions.set_index(["Date", "Security"])

    algo = algos.ReplayTransactions("transactions")
    # Exercise the public Backtest path without predeclaring blotter securities.
    strategy = bt.Strategy("strategy", algos=[algo])
    backtest = bt.backtest.Backtest(
        strategy,
        data,
        name="Test",
        additional_data={"bidoffer": {}, "transactions": transactions},
    )
    out = bt.run(backtest)
    t1 = transactions.sort_index(axis=1)
    t2 = out.get_transactions().sort_index(axis=1)
    assert t1.equals(t2)


@pytest.mark.parametrize("algo_name", ["replay", "rfq"])
@pytest.mark.parametrize(
    ("construction", "expected_multiplier"),
    [
        ("absent", 1),
        ("string", 1),
        ("explicit", 1),
        ("lazy-multiplier", 2),
        ("explicit-multiplier", 2),
    ],
)
def test_replayed_transactions_resolve_lazy_securities(
    algo_name, construction, expected_multiplier
):
    date = pd.Timestamp("2010-01-01")
    data = pd.DataFrame({"x": [100.0]}, index=[date])
    transactions = pd.DataFrame(
        [(date, "x", 1.0, 99.0)],
        columns=["Date", "Security", "quantity", "price"],
    ).set_index(["Date", "Security"])

    # Exercise every native construction source, including a property-bearing lazy node.
    if construction == "absent":
        children = None
    elif construction == "string":
        children = ["x"]
    elif construction == "explicit":
        children = [bt.Security("x")]
    elif construction == "lazy-multiplier":
        children = [bt.Security("x", multiplier=2, lazy_add=True)]
    else:
        children = [bt.Security("x", multiplier=2)]

    # Replay and RFQ use separate public transaction-dispatch loops.
    if algo_name == "replay":
        algo = algos.ReplayTransactions("transactions")
        additional_data = {"transactions": transactions}
    else:
        algo = algos.SimulateRFQTransactions(
            "rfqs", lambda rfqs, target: transactions.loc[rfqs.index]
        )
        additional_data = {"rfqs": transactions.drop(columns="price")}

    strategy = bt.Strategy("strategy", children=children)
    strategy.setup(data, bidoffer={}, **additional_data)
    strategy.adjust(1000.0)
    strategy.update(date)

    assert algo(strategy)

    # Derive the portfolio and reconstructed transaction values independently.
    security = strategy["x"]
    assert security.multiplier == expected_multiplier
    assert security.position == 1.0
    assert security.bidoffer_paid == -1.0 * expected_multiplier
    assert strategy.capital == 1000.0 - 99.0 * expected_multiplier
    assert strategy.value == 1000.0 + expected_multiplier
    expected_transactions = transactions.copy()
    expected_transactions["price"] = 100.0 + (99.0 - 100.0) * expected_multiplier
    pd.testing.assert_frame_equal(
        strategy.get_transactions().sort_index(axis=1),
        expected_transactions.sort_index(axis=1),
    )


@pytest.mark.parametrize("algo_name", ["replay", "rfq"])
def test_replayed_transactions_preserve_fixed_income_dispatch(algo_name):
    date = pd.Timestamp("2010-01-01")
    data = pd.DataFrame({"bond": [101.0]}, index=[date])
    transactions = pd.DataFrame(
        [(date, "bond", 5.0, 99.0)],
        columns=["Date", "Security", "quantity", "price"],
    ).set_index(["Date", "Security"])

    if algo_name == "replay":
        algo = algos.ReplayTransactions("transactions")
        additional_data = {"transactions": transactions}
    else:
        algo = algos.SimulateRFQTransactions(
            "rfqs", lambda rfqs, target: transactions.loc[rfqs.index]
        )
        additional_data = {"rfqs": transactions.drop(columns="price")}

    strategy = bt.FixedIncomeStrategy(
        "strategy", children=[bt.FixedIncomeSecurity("bond")]
    )
    strategy.setup(data, bidoffer={}, **additional_data)
    strategy.update(date)

    assert algo(strategy)

    # Fixed-income dispatch keeps quantity-based notional and price P&L semantics.
    assert strategy["bond"].position == 5.0
    assert strategy.notional_value == 5.0
    assert strategy.value == 10.0
    assert strategy.capital == -495.0


@pytest.mark.parametrize("algo_name", ["replay", "rfq"])
def test_replayed_transactions_reject_unknown_securities(algo_name):
    date = pd.Timestamp("2010-01-01")
    data = pd.DataFrame({"x": [100.0]}, index=[date])
    transactions = pd.DataFrame(
        [(date, "missing", 1.0, 99.0)],
        columns=["Date", "Security", "quantity", "price"],
    ).set_index(["Date", "Security"])

    if algo_name == "replay":
        algo = algos.ReplayTransactions("transactions")
        additional_data = {"transactions": transactions}
    else:
        algo = algos.SimulateRFQTransactions(
            "rfqs", lambda rfqs, target: transactions.loc[rfqs.index]
        )
        additional_data = {"rfqs": transactions.drop(columns="price")}

    strategy = bt.Strategy("strategy")
    strategy.setup(data, bidoffer={}, **additional_data)
    strategy.adjust(1000.0)
    strategy.update(date)

    # An undeclared label without market data must fail before creating a child.
    with pytest.raises(KeyError, match="missing"):
        algo(strategy)

    assert "missing" not in strategy.children
    assert strategy.capital == 1000.0
    assert strategy.value == 1000.0


@pytest.mark.parametrize("algo_name", ["replay", "rfq"])
@pytest.mark.parametrize(
    ("security", "quantity", "price", "error", "match"),
    [
        pytest.param("missing", 1.0, 99.0, KeyError, "missing", id="unknown-security"),
        pytest.param("y", 1.0, np.nan, ValueError, "finite", id="nan-price"),
        pytest.param("y", 1.0, np.inf, ValueError, "finite", id="infinite-price"),
        pytest.param("y", 1.0, "bad", ValueError, "numeric", id="nonnumeric-price"),
        pytest.param("y", np.inf, 99.0, ValueError, "finite", id="infinite-quantity"),
        pytest.param("y", "bad", 99.0, ValueError, "numeric", id="nonnumeric-quantity"),
        pytest.param("y", np.complex128(1 + 2j), 99.0, ValueError, "real", id="complex-quantity"),
        pytest.param("y", 1.0, np.complex128(99 + 2j), ValueError, "real", id="complex-price"),
    ],
)
def test_replayed_transactions_preflight_complete_batch(
    algo_name, security, quantity, price, error, match
):
    date = pd.Timestamp("2010-01-01")
    data = pd.DataFrame({"x": [100.0], "y": [100.0]}, index=[date])
    transactions = pd.DataFrame(
        [(date, "x", 1.0, 99.0), (date, security, quantity, price)],
        columns=["Date", "Security", "quantity", "price"],
        dtype=object,
    ).set_index(["Date", "Security"])

    if algo_name == "replay":
        algo = algos.ReplayTransactions("transactions")
        additional_data = {"transactions": transactions}
    else:
        algo = algos.SimulateRFQTransactions(
            "rfqs", lambda rfqs, target: transactions.loc[rfqs.index]
        )
        additional_data = {"rfqs": transactions.drop(columns="price")}

    # Keep the first valid target lazy so any partial dispatch is observable.
    strategy = bt.Strategy(
        "strategy",
        children=[
            bt.Security("x", lazy_add=True),
            bt.Security("y", lazy_add=True),
        ],
    )
    strategy.setup(data, bidoffer={}, **additional_data)
    strategy.adjust(1000.0)
    strategy.update(date)
    original_lazy_child = strategy._lazy_children["x"]

    with pytest.raises(error, match=match):
        algo(strategy)

    assert strategy.children == {}
    assert strategy._lazy_children["x"] is original_lazy_child
    assert set(strategy._lazy_children) == {"x", "y"}
    assert strategy.capital == 1000.0
    assert strategy.value == 1000.0


@pytest.mark.parametrize("algo_name", ["replay", "rfq"])
@pytest.mark.parametrize("input_kind", ["market-price", "decimal-quantity", "decimal-price"])
def test_replayed_transactions_normalize_numeric_inputs(algo_name, input_kind):
    from decimal import Decimal

    date = pd.Timestamp("2010-01-01")
    quantity = Decimal("1") if input_kind == "decimal-quantity" else 1.0
    price = None if input_kind == "market-price" else Decimal("99") if input_kind == "decimal-price" else 99.0
    transactions = pd.DataFrame(
        [(date, "x", 1.0, 99.0), (date, "y", quantity, price)],
        columns=["Date", "Security", "quantity", "price"],
        dtype=object,
    ).set_index(["Date", "Security"])
    if algo_name == "replay":
        algo = algos.ReplayTransactions("transactions")
        additional_data = {"transactions": transactions}
    else:
        algo = algos.SimulateRFQTransactions("rfqs", lambda rfqs, target: transactions)
        additional_data = {"rfqs": transactions.drop(columns="price")}

    strategy = bt.Strategy("strategy")
    strategy.setup(pd.DataFrame(100.0, index=[date], columns=["x", "y"]), bidoffer={}, **additional_data)
    strategy.adjust(1000.0)
    strategy.update(date)

    assert algo(strategy)
    assert strategy["x"].position == 1.0
    assert strategy["y"].position == 1.0
    expected_price = 100.0 if price is None else 99.0
    assert strategy.capital == 1000.0 - 99.0 - expected_price
    assert strategy.value == strategy.capital + 200.0
    assert strategy["x"].bidoffer_paid == -1.0
    assert strategy["y"].bidoffer_paid == expected_price - 100.0


@pytest.mark.parametrize("algo_name", ["replay", "rfq"])
@pytest.mark.parametrize("quantity", [0.0, np.nan], ids=["zero", "nan"])
def test_replayed_transactions_preserve_noop_quantity_with_unused_price(
    algo_name, quantity
):
    date = pd.Timestamp("2010-01-01")
    data = pd.DataFrame({"x": [100.0]}, index=[date])
    transactions = pd.DataFrame(
        [(date, "x", quantity, np.nan)],
        columns=["Date", "Security", "quantity", "price"],
    ).set_index(["Date", "Security"])

    if algo_name == "replay":
        algo = algos.ReplayTransactions("transactions")
        additional_data = {"transactions": transactions}
    else:
        algo = algos.SimulateRFQTransactions(
            "rfqs", lambda rfqs, target: transactions.loc[rfqs.index]
        )
        additional_data = {"rfqs": transactions.drop(columns="price")}

    strategy = bt.Strategy("strategy")
    strategy.setup(data, bidoffer={}, **additional_data)
    strategy.adjust(1000.0)
    strategy.update(date)

    # The transaction owner ignores price when quantity cannot move the position.
    assert algo(strategy)
    assert strategy["x"].position == 0.0
    assert strategy.capital == 1000.0
    assert strategy.value == 1000.0


@pytest.mark.parametrize("algo_name", ["replay", "rfq"])
def test_replayed_transactions_reject_strategy_targets(algo_name):
    date = pd.Timestamp("2010-01-01")
    data = pd.DataFrame({"x": [100.0], "nested": [100.0]}, index=[date])
    transactions = pd.DataFrame(
        [(date, "nested", 1.0, 99.0)],
        columns=["Date", "Security", "quantity", "price"],
    ).set_index(["Date", "Security"])

    if algo_name == "replay":
        algo = algos.ReplayTransactions("transactions")
        additional_data = {"transactions": transactions}
    else:
        algo = algos.SimulateRFQTransactions(
            "rfqs", lambda rfqs, target: transactions.loc[rfqs.index]
        )
        additional_data = {"rfqs": transactions.drop(columns="price")}

    strategy = bt.Strategy("strategy")
    strategy.setup(data, bidoffer={}, **additional_data)
    strategy.adjust(1000.0)
    strategy.update(date)
    strategy.children["nested"] = bt.Strategy("nested")

    with pytest.raises(TypeError, match="not a security"):
        algo(strategy)

    assert strategy.capital == 1000.0
    assert strategy.value == 1000.0


def test_simulate_rfq_transactions():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    s = bt.Strategy("s", children=[c1, c2])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]

    rfqs = pd.DataFrame(
        [
            ("A", pd.Timestamp("2009-12-01 00"), "c1", 100),
            ("B", pd.Timestamp("2010-01-01 10"), "c1", -100),
            ("C", pd.Timestamp("2010-01-01 12"), "c1", 75),
            ("D", pd.Timestamp("2010-01-02 00"), "c2", 50),
        ],
        columns=["id", "Date", "Security", "quantity"],
    )
    rfqs = rfqs.set_index(["Date", "Security"])

    def model(rfqs, target):
        # Dummy model - in practice this model would rely on positions and values in target
        transactions = rfqs[["quantity"]]
        prices = {"A": 99.5, "B": 101, "D": 103}
        transactions["price"] = rfqs.id.apply(lambda x: prices.get(x))
        return transactions.dropna()

    algo = algos.SimulateRFQTransactions("rfqs", model)

    s.setup(
        data, bidoffer={}, rfqs=rfqs
    )  # Pass bidoffer so it will track bidoffer paid
    s.adjust(1000)
    s.update(dts[0])
    algo(s)
    assert c1.position == 100
    assert c2.position == 0
    assert c1.bidoffer_paid == -50

    s.update(dts[1])
    algo(s)
    assert c1.position == 0
    assert c2.position == 50
    assert c1.bidoffer_paid == -100
    assert c2.bidoffer_paid == 150


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("failure", ["inf", "-inf", "product", "multiplier", "aggregate"])
def test_update_risk_rejects_infinite_exposure_before_mutation(nested, initialized, failure):
    dates = pd.date_range("2020-01-01", periods=2)
    prices = pd.DataFrame(100.0, index=dates, columns=["first", "last"])
    unit_risk = pd.DataFrame(3.0, index=dates, columns=prices.columns)
    multiplier = 2 if failure == "multiplier" else 1
    children = [bt.Security("first"), bt.Security("last", multiplier=multiplier)]
    sleeve = bt.Strategy("sleeve", children=children)
    root = bt.Strategy("root", children=[sleeve] if nested else children)
    root.setup(prices, unit_risk={"Test": unit_risk, "Other": unit_risk.copy()})
    root.adjust(1000)
    root.update(dates[0])
    target = root["sleeve"] if nested else root
    target.transact(1, "first")
    target.transact(2 if failure == "product" else 1, "last")
    algo = algos.UpdateRisk("Test", history=3)
    if initialized:
        assert algos.UpdateRisk("Other", history=1)(root)
        assert algo(root)
    root.update(dates[1])
    unit_risk.loc[dates[1], "last"] = {
        "inf": np.inf,
        "-inf": -np.inf,
        "product": 1e308,
        "multiplier": 1e308,
        "aggregate": 1e308,
    }[failure]
    if failure == "aggregate":
        unit_risk.loc[dates[1], "first"] = 1e308

    # A later bad leaf or total must not publish even the first valid sibling's risk.
    before = pickle.dumps(root)
    with np.errstate(over="raise"), pytest.raises((ValueError, FloatingPointError)):
        algo(root)
    assert pickle.dumps(root) == before


@pytest.mark.parametrize("dtype, missing", [("float64", np.nan), ("Float64", pd.NA)])
@pytest.mark.parametrize("history", [0, 2])
def test_update_risk_preserves_missing_exposure(dtype, missing, history):
    dates = pd.date_range("2020-01-01", periods=1)
    prices = pd.DataFrame({"asset": [100.0]}, index=dates)
    unit_risk = pd.DataFrame({"asset": pd.Series([missing], index=dates, dtype=dtype)})
    target = bt.Strategy("target", children=[bt.Security("asset")])
    target.setup(prices, unit_risk={"Test": unit_risk})
    target.adjust(1000)
    target.update(dates[0])
    target.transact(2, "asset")

    # Missing risk remains available to the downstream consumer's existing policy.
    assert algos.UpdateRisk("Test", history=history)(target)
    for node in (target, target["asset"]):
        assert pd.isna(node.risk["Test"])
        if history:
            assert pd.isna(node.risks.loc[dates[0], "Test"])
        else:
            assert not hasattr(node, "risks")


def test_update_risk():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    s = bt.Strategy("s", children=[c1, c2])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95
    c1 = s["c1"]
    c2 = s["c2"]

    algo = algos.UpdateRisk("Test", history=False)

    s.setup(data, unit_risk={"Test": data})
    s.adjust(1000)

    s.update(dts[0])
    assert algo(s)
    assert s.risk["Test"] == 0
    assert c1.risk["Test"] == 0
    assert c2.risk["Test"] == 0

    s.transact(1, "c1")
    s.transact(5, "c2")
    assert algo(s)
    assert s.risk["Test"] == 600
    assert c1.risk["Test"] == 100
    assert c2.risk["Test"] == 500

    s.update(dts[1])
    assert algo(s)
    assert s.risk["Test"] == 105 + 5 * 95
    assert c1.risk["Test"] == 105
    assert c2.risk["Test"] == 5 * 95

    assert not hasattr(s, "risks")
    assert not hasattr(c1, "risks")
    assert not hasattr(c2, "risks")


def test_update_risk_history_1():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    s = bt.Strategy("s", children=[c1, c2])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95
    c1 = s["c1"]
    c2 = s["c2"]

    algo = algos.UpdateRisk("Test", history=1)

    s.setup(data, unit_risk={"Test": data})
    s.adjust(1000)

    s.update(dts[0])
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 0

    s.transact(1, "c1")
    s.transact(5, "c2")
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 600

    s.update(dts[1])
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 600
    assert s.risks["Test"].iloc[1] == 105 + 5 * 95

    assert not hasattr(c1, "risks")
    assert not hasattr(c2, "risks")


def test_update_risk_history_2():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    s = bt.Strategy("s", children=[c1, c2])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95
    c1 = s["c1"]
    c2 = s["c2"]

    algo = algos.UpdateRisk("Test", history=2)

    s.setup(data, unit_risk={"Test": data})
    s.adjust(1000)

    s.update(dts[0])
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 0
    assert c1.risks["Test"].iloc[0] == 0
    assert c2.risks["Test"].iloc[0] == 0

    s.transact(1, "c1")
    s.transact(5, "c2")
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 600
    assert c1.risks["Test"].iloc[0] == 100
    assert c2.risks["Test"].iloc[0] == 500

    s.update(dts[1])
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 600
    assert c1.risks["Test"].iloc[0] == 100
    assert c2.risks["Test"].iloc[0] == 500
    assert s.risks["Test"].iloc[1] == 105 + 5 * 95
    assert c1.risks["Test"].iloc[1] == 105
    assert c2.risks["Test"].iloc[1] == 5 * 95


@pytest.mark.parametrize("shallow_history", [0, 1, 2])
def test_update_risk_history_is_independent_of_measure_order(shallow_history):
    dts = pd.date_range("2020-01-01", periods=2)
    data = pd.DataFrame({"asset": [100.0, 100.0]}, index=dts)
    unit_risk = {
        "IR01": pd.DataFrame({"asset": [3.0, 4.0]}, index=dts),
        "DV01": pd.DataFrame({"asset": [5.0, 7.0]}, index=dts),
    }

    def run(measures):
        # Exercise history boundaries at root, sleeve, and security depths.
        target = bt.Strategy("root")
        target.setup(data, unit_risk=unit_risk)
        sleeve = bt.Strategy(
            "sleeve",
            children=[bt.Security("asset", multiplier=2)],
            parent=target,
        )
        sleeve.setup_from_parent()
        target.adjust(1000)
        target.update(dts[0])
        sleeve.transact(2, "asset")

        updates = {
            "IR01": algos.UpdateRisk("IR01", history=shallow_history),
            "DV01": algos.UpdateRisk("DV01", history=shallow_history + 1),
        }
        stack = bt.AlgoStack(*(updates[measure] for measure in measures))
        assert stack(target)
        return target

    # Reversing independent measures must not change values or history availability.
    shallow_first = run(("IR01", "DV01"))
    deep_first = run(("DV01", "IR01"))
    shallow_nodes = [
        shallow_first,
        shallow_first["sleeve"],
        shallow_first["sleeve"]["asset"],
    ]
    deep_nodes = [
        deep_first,
        deep_first["sleeve"],
        deep_first["sleeve"]["asset"],
    ]

    for depth, (actual, control) in enumerate(zip(shallow_nodes, deep_nodes)):
        assert actual.risk == control.risk == {"IR01": 12.0, "DV01": 20.0}
        # A history depth tracks nodes whose zero-based depth is below it.
        expected_history = {}
        if depth < shallow_history:
            expected_history["IR01"] = 12.0
        if depth < shallow_history + 1:
            expected_history["DV01"] = 20.0

        if expected_history:
            assert actual.risks.loc[dts[0]].dropna().to_dict() == expected_history
            pd.testing.assert_frame_equal(
                actual.risks.sort_index(axis=1), control.risks.sort_index(axis=1)
            )
        else:
            assert not hasattr(actual, "risks")
            assert not hasattr(control, "risks")


def test_update_risk_history_starts_when_deeper_measure_is_first_tracked():
    dts = pd.date_range("2020-01-01", periods=2)
    data = pd.DataFrame({"asset": [100.0, 100.0]}, index=dts)
    unit_risk = {
        "IR01": pd.DataFrame({"asset": [3.0, 4.0]}, index=dts),
        "DV01": pd.DataFrame({"asset": [5.0, 7.0]}, index=dts),
    }
    target = bt.Strategy("root")
    target.setup(data, unit_risk=unit_risk)
    sleeve = bt.Strategy(
        "sleeve",
        children=[bt.Security("asset", multiplier=2)],
        parent=target,
    )
    sleeve.setup_from_parent()
    target.adjust(1000)
    target.update(dts[0])
    sleeve.transact(2, "asset")
    shallow = algos.UpdateRisk("IR01", history=0)
    deeper = algos.UpdateRisk("DV01", history=3)

    # Establish current-only IR01 before deeper DV01 tracking starts a date later.
    assert shallow(target)
    target.update(dts[1])
    assert shallow(target)
    assert deeper(target)

    # Late history initialization must not fabricate the untracked measure's past.
    for node in (target, target["sleeve"], target["sleeve"]["asset"]):
        assert node.risk == {"IR01": 16.0, "DV01": 28.0}
        assert list(node.risks.columns) == ["DV01"]
        assert pd.isna(node.risks.loc[dts[0], "DV01"])
        assert node.risks.loc[dts[1], "DV01"] == 28.0


def test_hedge_risk():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    c3 = bt.Security("c3")
    s = bt.Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    risk1 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk2 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk1["c1"] = 1
    risk1["c2"] = 10
    risk2["c1"] = 2
    risk2["c2"] = 5
    risk2["c3"] = 10

    stack = bt.core.AlgoStack(
        algos.UpdateRisk("Risk1"),
        algos.UpdateRisk("Risk2"),
        algos.SelectThese(["c2", "c3"]),
        algos.HedgeRisks(["Risk1", "Risk2"]),
        algos.UpdateRisk("Risk1"),
        algos.UpdateRisk("Risk2"),
    )

    s.setup(data, unit_risk={"Risk1": risk1, "Risk2": risk2})
    s.adjust(1000)

    s.update(dts[0])
    s.transact(100, "c1")
    stack(s)

    # Check that risk is hedged!
    assert s.risk["Risk1"] == 0
    assert s.risk["Risk2"] == pytest.approx(0, 13)
    # Check that positions are nonzero (trivial solution)
    assert c1.position == 100
    assert c2.position == -10
    assert c3.position == pytest.approx(-(100 * 2 - 10 * 5) / 10.0, 13)


@pytest.mark.parametrize("missing_measure", ["Risk1", "Risk2"])
@pytest.mark.parametrize("present_but_none", [False, True])
def test_hedge_risk_reports_missing_unit_risk(missing_measure, present_but_none):
    dates = pd.date_range("2020-01-01", periods=1)
    prices = pd.DataFrame({"hedge": [100.0]}, index=dates)
    unit_risk = {measure: prices * 0.01 for measure in ("Risk1", "Risk2") if measure != missing_measure}
    if present_but_none:
        unit_risk[missing_measure] = None
    strategy = bt.Strategy("strategy", children=[bt.Security("hedge")])
    strategy.setup(prices, unit_risk=unit_risk)
    strategy.adjust(1000.0)
    strategy.update(dates[0])
    strategy.transact(5.0, "hedge")
    strategy.update(dates[0])
    strategy.risk = {"Risk1": 5.0, "Risk2": 10.0}
    strategy.temp["selected"] = ["hedge"]

    with pytest.raises(ValueError, match=f"unit_risk for {missing_measure} .* on strategy"):
        algos.HedgeRisks(["Risk1", "Risk2"])(strategy)

    assert strategy["hedge"].position == 5.0
    assert strategy.capital == 500.0


@pytest.mark.parametrize(("hedge_unit_risk", "hedge_multiplier"), [(1, 10), (10, 1)])
@pytest.mark.parametrize("lazy_add", [False, True])
def test_hedge_risk_security_multiplier(hedge_unit_risk: int, hedge_multiplier: int, lazy_add: bool):
    source = bt.Security("source")
    hedge = bt.HedgeSecurity("hedge", multiplier=hedge_multiplier, lazy_add=lazy_add)
    strategy = bt.Strategy("strategy", children=[source, hedge])
    dates = pd.date_range("2010-01-01", periods=1)
    prices = pd.DataFrame(100, index=dates, columns=["source", "hedge"])
    unit_risk = pd.DataFrame({"source": [1], "hedge": [hedge_unit_risk]}, index=dates)
    stack = bt.core.AlgoStack(
        algos.UpdateRisk("Risk"),
        algos.SelectThese(["hedge"]),
        algos.HedgeRisks(["Risk"]),
        algos.UpdateRisk("Risk"),
    )

    strategy.setup(prices, unit_risk={"Risk": unit_risk})
    strategy.update(dates[0])
    strategy.transact(100, "source")
    stack(strategy)

    # Equivalent per-contract sensitivities must produce the same hedge and residual.
    assert strategy["hedge"].position == -10
    assert vars(strategy)["risk"]["Risk"] == 0


@pytest.mark.parametrize(
    ("quantity", "throw_nan", "message"),
    [(np.inf, True, "infinite"), (np.inf, False, "infinite"), (-np.inf, True, "infinite"), (-np.inf, False, "infinite"), (np.nan, True, "nan")],
)
@pytest.mark.parametrize("lazy_add", [False, True])
def test_hedge_risk_rejects_invalid_vector_before_trading(quantity, throw_nan, message, lazy_add):
    dates = pd.date_range("2020-01-01", periods=1)
    prices = pd.DataFrame(100.0, index=dates, columns=["first", "last"])
    strategy = bt.Strategy("strategy", children=[bt.Security("first"), bt.HedgeSecurity("last", lazy_add=lazy_add)])
    strategy.setup(prices, unit_risk={"Risk": prices * 0.01})
    strategy.adjust(1000.0)
    strategy.update(dates[0])
    strategy.risk = {"Risk": 10.0}
    strategy.temp["selected"] = ["first", "last"]
    children = strategy.children.copy()
    lazy_children = strategy._lazy_children.copy()
    state = (strategy.capital, strategy["first"].position, strategy.stale, strategy.risk.copy())
    history = strategy.data.copy(deep=True)

    # Isolate validation timing from solver numerics: the first hedge is tradable,
    # but a later invalid result must prevent both fills and lazy-child creation.
    with (
        mock.patch.object(algos.np, "matmul", return_value=np.array([[2.0], [quantity]])),
        mock.patch.object(strategy, "transact", wraps=strategy.transact) as transact,
        pytest.raises(ValueError, match=f"last has {message} hedge notional"),
        np.errstate(all="raise"),
    ):
        algos.HedgeRisks(["Risk"], pseudo=True, throw_nan=throw_nan)(strategy)

    transact.assert_not_called()
    assert strategy.children == children
    assert strategy._lazy_children == lazy_children
    assert (strategy.capital, strategy["first"].position, strategy.stale, strategy.risk) == state
    pd.testing.assert_frame_equal(strategy.data, history)


@pytest.mark.parametrize("target_risk", [1e308, -1e308])
@pytest.mark.parametrize("throw_nan", [False, True])
def test_hedge_risk_rejects_overflow_from_finite_inputs(target_risk, throw_nan):
    dates = pd.date_range("2020-01-01", periods=1)
    prices = pd.DataFrame({"hedge": [100.0]}, index=dates)
    strategy = bt.Strategy("strategy", children=[bt.HedgeSecurity("hedge", lazy_add=True)])
    strategy.setup(prices, unit_risk={"Risk": pd.DataFrame({"hedge": [1e-320]}, index=dates)})
    strategy.adjust(1000.0)
    strategy.update(dates[0])
    strategy.risk = {"Risk": target_risk}
    strategy.temp["selected"] = ["hedge"]

    # Finite risk divided by subnormal unit risk needs roughly 1e628 contracts,
    # beyond float64. Reject the real solver result before even registering a child.
    with pytest.raises(ValueError, match="hedge has infinite hedge notional"), np.errstate(all="raise"):
        algos.HedgeRisks(["Risk"], throw_nan=throw_nan)(strategy)

    assert not strategy.children
    assert list(strategy._lazy_children) == ["hedge"]
    assert strategy.capital == 1000.0
    assert strategy.risk == {"Risk": target_risk}


def test_hedge_risk_preserves_nan_dispatch_when_disabled():
    dates = pd.date_range("2020-01-01", periods=1)
    prices = pd.DataFrame(100.0, index=dates, columns=["skipped", "traded"])
    strategy = bt.Strategy("strategy")
    strategy.setup(prices, unit_risk={"Risk": prices * 0.01})
    strategy.adjust(1000.0)
    strategy.update(dates[0])
    strategy.risk = {"Risk": 10.0}
    strategy.temp["selected"] = ["skipped", "traded"]

    # False-mode NaN still reaches native dispatch (including child creation),
    # while a finite sibling trades in the original selection order.
    with (
        mock.patch.object(algos.np, "matmul", return_value=np.array([[np.nan], [2.0]])),
        mock.patch.object(strategy, "transact", wraps=strategy.transact) as transact,
        np.errstate(all="raise"),
    ):
        assert algos.HedgeRisks(["Risk"], pseudo=True, throw_nan=False)(strategy)

    assert [call.args[1] for call in transact.call_args_list] == ["skipped", "traded"]
    assert strategy["skipped"].position == 0.0
    assert strategy["traded"].position == 2.0
    assert strategy.capital == 800.0


def test_hedge_risk_nan():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    c3 = bt.Security("c3")
    s = bt.Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    risk1 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk2 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk1["c1"] = 1
    risk1["c2"] = 10
    risk2["c1"] = float("nan")
    risk2["c2"] = 5
    risk2["c3"] = 10

    stack = bt.core.AlgoStack(
        algos.UpdateRisk("Risk1"),
        algos.UpdateRisk("Risk2"),
        algos.SelectThese(["c2", "c3"]),
        algos.HedgeRisks(["Risk1", "Risk2"], throw_nan=False),
    )
    stack_throw = bt.core.AlgoStack(
        algos.UpdateRisk("Risk1"),
        algos.UpdateRisk("Risk2"),
        algos.SelectThese(["c2", "c3"]),
        algos.HedgeRisks(["Risk1", "Risk2"]),
    )

    s.setup(data, unit_risk={"Risk1": risk1, "Risk2": risk2})
    s.adjust(1000)

    s.update(dts[0])
    s.transact(100, "c1")
    assert stack(s)

    did_throw = False
    try:
        stack_throw(s)
    except ValueError:
        did_throw = True
    assert did_throw


def test_hedge_risk_pseudo_under():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    c3 = bt.Security("c3")
    s = bt.Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    risk1 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk2 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk1["c1"] = 1
    risk1["c2"] = 10
    risk2["c1"] = 2
    risk2["c3"] = 10

    stack = bt.core.AlgoStack(
        algos.UpdateRisk("Risk1"),
        algos.UpdateRisk("Risk2"),
        algos.SelectThese(["c2"]),
        algos.HedgeRisks(["Risk1", "Risk2"], pseudo=True),
        algos.UpdateRisk("Risk1"),
        algos.UpdateRisk("Risk2"),
    )

    s.setup(data, unit_risk={"Risk1": risk1, "Risk2": risk2})
    s.adjust(1000)

    s.update(dts[0])
    s.transact(100, "c1")
    stack(s)

    # Check that risk is hedged!
    assert s.risk["Risk1"] == 0
    assert s.risk["Risk2"] != 0
    # Check that positions are nonzero (trivial solution)
    assert c1.position == 100
    assert c2.position == -10
    assert c3.position == 0


def test_hedge_risk_pseudo_over():
    c1 = bt.Security("c1")
    c2 = bt.Security("c2")
    c3 = bt.Security("c3")
    s = bt.Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c2 = s["c2"]
    c3 = s["c3"]

    risk1 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk1["c1"] = 1
    risk1["c2"] = 10
    risk1["c3"] = 10  # Same risk as c2

    stack = bt.core.AlgoStack(
        algos.UpdateRisk("Risk1"),
        algos.SelectThese(["c2", "c3"]),
        algos.HedgeRisks(["Risk1"], pseudo=True),
        algos.UpdateRisk("Risk1"),
    )

    s.setup(data, unit_risk={"Risk1": risk1})
    s.adjust(1000)

    s.update(dts[0])
    s.transact(100, "c1")
    stack(s)

    # Check that risk is hedged!
    assert s.risk["Risk1"] == 0
    # Check that positions are nonzero and risk is evenly split between hedge instruments
    assert c1.position == 100
    assert c2.position == -5
    assert c3.position == -5


def test_margin():
    algo = algos.Margin(0.1, 0.66666666667)

    s = bt.Strategy("s", algos=[algos.WeighSpecified(c1=2), algos.Rebalance()])

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1"], data=1)

    yesterday = dts[0] - timedelta(days=1)
    algo._last_date = yesterday

    s.setup(data)
    s.update(dts[0])
    s.adjust(1000)
    s.run()

    algo(s)

    # checked that we charged some margin interest
    fees = np.sum(s.fees)
    assert pytest.approx(0.26, 0.01) == fees

    # check that we've liquidated things to get us back to the maintenance requirement
    assert pytest.approx(1499, 0.001) == sum(child.value for child in s.children.values())

    assert pytest.approx(999.73, 0.001) == s.value


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("requirement", [0.1, 2.0 / 3.0])
def test_margin_interest_reduces_return_without_external_flow(nested, requirement):
    dates = pd.date_range("2024-01-01", periods=2)
    data = pd.DataFrame({"asset": [100.0, 100.0]}, index=dates)
    sleeve = bt.Strategy("sleeve", children=["asset"])
    root = bt.Strategy("root", children=[sleeve]) if nested else sleeve
    root.use_integer_positions(False)
    root.setup(data)
    root.adjust(1000.0)
    root.update(dates[0])
    target = root["sleeve"] if nested else root
    if nested:
        target.allocate(1000.0)
    target.allocate(2000.0, "asset")
    root.update(dates[0])
    margin = algos.Margin(0.1, requirement)
    margin(target)

    root.update(dates[1])
    margin(target)
    root.update(dates[1])

    expected_fee = 1000.0 * (1.1 ** (1.0 / 365.25) - 1.0)
    assert target.fees.loc[dates[1]] == pytest.approx(expected_fee)
    # Liquidation can return capital to the parent, but interest is never a flow.
    assert target.flows.loc[dates[1]] == pytest.approx(-root.capital if nested else 0.0)
    assert root.flows.loc[dates[1]] == 0.0
    assert root.value == pytest.approx(1000.0 - expected_fee)
    assert root.price == pytest.approx(100.0 * (1000.0 - expected_fee) / 1000.0)


@pytest.mark.parametrize(
    ("leverage", "requirement"),
    [(1.5, 0.8), (3.0, 0.5)],
)
def test_margin_liquidation_scales_with_leverage(leverage: float, requirement: float):
    algo = algos.Margin(0, requirement)
    s = bt.Strategy("s", algos=[algos.WeighSpecified(c1=leverage), algos.Rebalance()])
    # Fractional positions isolate the liquidation formula from integer rounding.
    s.use_integer_positions(False)
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=1)
    algo._last_date = dts[0] - timedelta(days=1)

    s.setup(data)
    s.update(dts[0])
    s.adjust(1000)
    s.run()

    equity = s.value
    invested_value = s["c1"].value
    assert invested_value == pytest.approx(equity * leverage)
    assert equity / invested_value < requirement

    algo(s)

    # With no interest or transaction costs, liquidation preserves equity and targets E / R.
    expected_invested_value = equity / requirement
    invested_value = s["c1"].value
    assert s.value == pytest.approx(equity)
    assert invested_value == pytest.approx(expected_invested_value)
    assert s.value / invested_value == pytest.approx(requirement)


def test_corporate_actions():
    dts = pd.date_range("2010-01-01", periods=3)

    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    divs = pd.DataFrame(index=dts, columns=["c1", "c2"], data=0.0)
    divs.loc[dts[1], "c1"] = 2.0
    splits = pd.DataFrame(index=dts, columns=["c1", "c2"], data=1.0)
    splits.loc[dts[2], "c2"] = 10.0

    algo = algos.CorporateActions(divs, splits)

    s = bt.Strategy("s", children=["c1", "c2", "c3"])
    s.setup(data)
    s.adjust(30000)

    s.update(dts[0])
    s.allocate(10000, "c1", update=True)
    s.allocate(10000, "c2", update=True)
    s.allocate(10000, "c3", update=True)

    assert algo(s)
    assert s.capital == 0
    assert s["c1"].position == 100
    assert s["c2"].position == 100
    assert s["c3"].position == 100

    s.update(dts[1])

    assert algo(s)
    assert s.capital == 100 * 2.0
    assert s["c1"].position == 100
    assert s["c2"].position == 100
    assert s["c3"].position == 100

    s.update(dts[2])

    assert algo(s)
    assert s.capital == 100 * 2.0
    assert s["c1"].position == 100
    assert s["c2"].position == 100 * 10.0
    assert s["c3"].position == 100


@pytest.mark.parametrize("action", ["split", "dividend"])
@pytest.mark.parametrize("amount", [np.inf, -np.inf])
@pytest.mark.parametrize("dtype", ["float64", "float32", "Float32"])
@pytest.mark.parametrize("nested", [False, True])
def test_corporate_actions_rejects_nonfinite_before_mutation(action, amount, dtype, nested):
    dates = pd.date_range("2024-01-01", periods=2)
    data = pd.DataFrame({"first": [100.0, 50.0], "bad": [100.0, 100.0]}, index=dates)
    target = bt.Strategy("target", children=[bt.Security("first"), bt.Security("bad")])
    root = bt.Strategy("root", children=[target]) if nested else target
    # Strategy copies supplied children; exercise the installed target.
    target = root["target"] if nested else root
    root.use_integer_positions(False)
    root.setup(data)
    root.adjust(1000.0)
    root.update(dates[0])
    if nested:
        root.allocate(1000.0, "target")
    target["first"].transact(2.0)
    target["bad"].transact(2.0)
    root.update(dates[0])
    root.update(dates[1])
    splits = pd.DataFrame({"first": [2.0], "bad": pd.Series([amount if action == "split" else 1.0], dtype=dtype).values}, index=dates[1:])
    dividends = pd.DataFrame({"first": [1.0], "bad": pd.Series([amount if action == "dividend" else 0.0], dtype=dtype).values}, index=dates[1:])
    algo = algos.CorporateActions(dividends, splits)
    # A later invalid action must not leave the first valid split applied.
    before = pickle.dumps(root)
    with pytest.raises(ValueError, match="Corporate action .* must be finite"):
        algo(target)
    assert pickle.dumps(root) == before


def test_corporate_actions_reads_each_action_row_once():
    class LocCountingFrame(pd.DataFrame):
        _metadata: ClassVar[list[str]] = ["loc_accesses"]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loc_accesses = 0

        @property
        def _constructor(self):
            return LocCountingFrame

        @property
        def loc(self):
            self.loc_accesses += 1
            return super().loc

    date = pd.Timestamp("2024-01-02")
    names = ["long", "short", "flat"]
    data = pd.DataFrame(100.0, index=[date], columns=names)
    dividends = LocCountingFrame([[1.0, 2.0, 4.0]], index=[date], columns=names)
    splits = LocCountingFrame([[2.0, 0.5, 3.0]], index=[date], columns=names)
    algo = algos.CorporateActions(dividends, splits)
    target = bt.Strategy("target", children=[bt.Security(name) for name in names])
    target.setup(data)
    target.update(date)
    target.adjust(100.0, update=False)
    target["long"]._position = 10.0
    target["short"]._position = -5.0

    assert algo(target)

    # Splits precede dividends, so cash uses the adjusted long and short positions.
    assert target["long"].position == pytest.approx(20.0)
    assert target["short"].position == pytest.approx(-2.5)
    assert target["flat"].position == pytest.approx(0.0)
    assert target.capital == pytest.approx(115.0)
    assert target.root.stale
    assert algo.splits.loc_accesses == 1
    assert algo.dividends.loc_accesses == 1


@pytest.mark.parametrize("action", ["split", "dividend"])
@pytest.mark.parametrize("dtype", ["float32", "Float32"])
def test_corporate_actions_preserves_mixed_column_arithmetic(action, dtype):
    date = pd.Timestamp("2024-01-02")
    data = pd.DataFrame({"asset": [1.0]}, index=[date])
    actions = pd.DataFrame(
        {
            "asset": pd.Series([0.1], index=[date], dtype=dtype),
            "unheld": pd.Series([0.25], index=[date], dtype="float64"),
        }
    )
    empty = pd.DataFrame()
    algo = algos.CorporateActions(empty if action == "split" else actions, actions if action == "split" else empty)
    target = bt.Strategy("target", children=[bt.Security("asset")])
    target.setup(data)
    target.adjust(100_000_000.0)
    target.update(date)
    target["asset"].transact(10_000_000.0)
    target.update(date)
    position = target["asset"].position
    capital = target.capital
    # Preserve the existing scalar lookup's dtype and arithmetic exactly.
    expected = float(actions.loc[date, "asset"] * position)

    assert algo(target)

    assert target["asset"].position == (expected if action == "split" else position)
    assert target.capital == (capital if action == "split" else capital + expected)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("include_zero_dividend_row", [False, True])
def test_corporate_actions_refreshes_stale_split_state_before_rebalance(nested: bool, include_zero_dividend_row: bool):
    dts = pd.date_range("2020-01-01", periods=2)
    data = pd.DataFrame({"c1": [100.0, 50.0], "c2": [100.0, 100.0]}, index=dts)

    # A nested target must invalidate the root rather than a target-local flag.
    if nested:
        root = bt.Strategy("root")
        root.use_integer_positions(False)
        root.setup(data)
        target = bt.Strategy("target", children=["c1", "c2"], parent=root)
        target.setup_from_parent()
    else:
        target = root = bt.Strategy("target", children=["c1", "c2"])
        root.use_integer_positions(False)
        root.setup(data)

    # Establish an exact 50/50 portfolio before the unadjusted split date.
    root.adjust(2000.0)
    root.update(dts[0])
    if nested:
        root.allocate(2000.0, "target")
    target.allocate(1000.0, "c1")
    target.allocate(1000.0, "c2")
    root.update(dts[0])
    root.update(dts[1])

    # A zero dividend row is inert and must not control split invalidation.
    dividend_index = dts[1:] if include_zero_dividend_row else dts[:0]
    dividends = pd.DataFrame(0.0, index=dividend_index, columns=["c1", "c2"])
    splits = pd.DataFrame({"c1": [2.0], "c2": [1.0]}, index=dts[1:])
    stack = bt.AlgoStack(
        algos.CorporateActions(dividends, splits),
        algos.WeighSpecified(c1=0.5, c2=0.5),
        algos.Rebalance(),
    )

    assert stack(target)

    # The split doubles quantity as price halves, so no rebalance trade is needed.
    assert target["c1"].position == pytest.approx(20.0)
    assert target["c2"].position == pytest.approx(10.0)
    assert target["c1"].value == pytest.approx(1000.0)
    assert target["c2"].value == pytest.approx(1000.0)
    assert target["c1"].weight == pytest.approx(0.5)
    assert target["c2"].weight == pytest.approx(0.5)
    assert target["c1"].outlays.loc[dts[1]] == pytest.approx(0.0)
    assert target["c2"].outlays.loc[dts[1]] == pytest.approx(0.0)
    assert target.capital == pytest.approx(0.0)
    assert target.value == pytest.approx(2000.0)
    assert root.value == pytest.approx(2000.0)
