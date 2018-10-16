from __future__ import division
from datetime import datetime

import sys
if sys.version_info < (3, 3):
    import mock
else:
    from unittest import mock


import pandas as pd
import numpy as np
from nose.tools import assert_almost_equal as aae

import bt
import bt.algos as algos


def test_algo_name():
    class TestAlgo(algos.Algo):
        pass

    actual = TestAlgo()

    assert actual.name == 'TestAlgo'


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


def test_run_once():
    algo = algos.RunOnce()
    assert algo(None)
    assert not algo(None)
    assert not algo(None)


def test_run_period():
    target = mock.MagicMock()

    dts = pd.date_range('2010-01-01', periods=35)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    algo = algos.RunPeriod()

    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
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
        run_on_first_date=False,
        run_on_end_of_period=True,
        run_on_last_date=True
    )

    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
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

    dts = pd.date_range('2010-01-01', periods=35)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    algo = algos.RunDaily()

    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('',[algo]),
        data
    )
    target.data = backtest.data

    target.now = dts[1]
    assert algo(target)



def test_run_weekly():
    dts = pd.date_range('2010-01-01', periods=367)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = algos.RunWeekly()
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # end of week
    target.now = dts[2]
    assert not algo(target)

    # new week
    target.now = dts[3]
    assert algo(target)

    algo = algos.RunWeekly(
        run_on_first_date=False,
        run_on_end_of_period=True,
        run_on_last_date=True
    )
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # end of week
    target.now = dts[2]
    assert algo(target)

    # new week
    target.now = dts[3]
    assert not algo(target)

    dts = pd.DatetimeIndex([datetime(2016, 1, 3), datetime(2017, 1, 8),datetime(2018, 1, 7)])
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # check next year
    target.now = dts[1]
    assert algo(target)


def test_run_monthly():
    dts = pd.date_range('2010-01-01', periods=367)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = algos.RunMonthly()
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # end of month
    target.now = dts[30]
    assert not algo(target)

    # new month
    target.now = dts[31]
    assert algo(target)

    algo = algos.RunMonthly(
        run_on_first_date=False,
        run_on_end_of_period=True,
        run_on_last_date=True
    )
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # end of month
    target.now = dts[30]
    assert algo(target)

    # new month
    target.now = dts[31]
    assert not algo(target)

    dts = pd.DatetimeIndex([datetime(2016, 1, 3), datetime(2017, 1, 8), datetime(2018, 1, 7)])
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # check next year
    target.now = dts[1]
    assert algo(target)


def test_run_quarterly():
    dts = pd.date_range('2010-01-01', periods=367)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = algos.RunQuarterly()
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # end of quarter
    target.now = dts[89]
    assert not algo(target)

    # new quarter
    target.now = dts[90]
    assert algo(target)

    algo = algos.RunQuarterly(
        run_on_first_date=False,
        run_on_end_of_period=True,
        run_on_last_date=True
    )
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # end of quarter
    target.now = dts[89]
    assert algo(target)

    # new quarter
    target.now = dts[90]
    assert not algo(target)

    dts = pd.DatetimeIndex([datetime(2016, 1, 3), datetime(2017, 1, 8), datetime(2018, 1, 7)])
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # check next year
    target.now = dts[1]
    assert algo(target)


def test_run_yearly():
    dts = pd.date_range('2010-01-01', periods=367)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = algos.RunYearly()
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # end of year
    target.now = dts[364]
    assert not algo(target)

    # new year
    target.now = dts[365]
    assert algo(target)

    algo = algos.RunYearly(
        run_on_first_date=False,
        run_on_end_of_period=True,
        run_on_last_date=True
    )
    # adds the initial day
    backtest = bt.Backtest(
        bt.Strategy('', [algo]),
        data
    )
    target.data = backtest.data

    # end of year
    target.now = dts[364]
    assert algo(target)

    # new year
    target.now = dts[365]
    assert not algo(target)


def test_run_on_date():
    target = mock.MagicMock()
    target.now = pd.to_datetime('2010-01-01')

    algo = algos.RunOnDate('2010-01-01', '2010-01-02')
    assert algo(target)

    target.now = pd.to_datetime('2010-01-02')
    assert algo(target)

    target.now = pd.to_datetime('2010-01-03')
    assert not algo(target)


def test_rebalance():
    algo = algos.Rebalance()

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp['weights'] = {'c1': 1}

    assert algo(s)
    assert s.value == 1000
    assert s.capital == 0
    c1 = s['c1']
    assert c1.value == 1000
    assert c1.position == 10
    assert c1.weight == 1.

    s.temp['weights'] = {'c2': 1}

    assert algo(s)
    assert s.value == 1000
    assert s.capital == 0
    c2 = s['c2']
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 1000
    assert c2.position == 10
    assert c2.weight == 1.


def test_rebalance_with_commissions():
    algo = algos.Rebalance()

    s = bt.Strategy('s')
    s.set_commissions(lambda q, p: 1)

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp['weights'] = {'c1': 1}

    assert algo(s)
    assert s.value == 999
    assert s.capital == 99
    c1 = s['c1']
    assert c1.value == 900
    assert c1.position == 9
    assert c1.weight == 900 / 999.

    s.temp['weights'] = {'c2': 1}

    assert algo(s)
    assert s.value == 997
    assert s.capital == 97
    c2 = s['c2']
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 900
    assert c2.position == 9
    assert c2.weight == 900. / 997


def test_rebalance_with_cash():
    algo = algos.Rebalance()

    s = bt.Strategy('s')
    s.set_commissions(lambda q, p: 1)

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp['weights'] = {'c1': 1}
    # set cash amount
    s.temp['cash'] = 0.5

    assert algo(s)
    assert s.value == 999
    assert s.capital == 599
    c1 = s['c1']
    assert c1.value == 400
    assert c1.position == 4
    assert c1.weight == 400.0 / 999

    s.temp['weights'] = {'c2': 1}
    # change cash amount
    s.temp['cash'] = 0.25

    assert algo(s)
    assert s.value == 997
    assert s.capital == 297
    c2 = s['c2']
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 700
    assert c2.position == 7
    assert c2.weight == 700.0 / 997


def test_select_all():
    algo = algos.SelectAll()

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)
    data['c1'][dts[1]] = np.nan
    data['c2'][dts[1]] = 95

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 2
    assert 'c1' in selected
    assert 'c2' in selected

    # make sure don't keep nan
    s.update(dts[1])

    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 1
    assert 'c2' in selected

    # if specify include_no_data then 2
    algo = algos.SelectAll(include_no_data=True)

    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 2
    assert 'c1' in selected
    assert 'c2' in selected


def test_weight_equally():
    algo = algos.WeighEqually()

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)
    s.update(dts[0])
    s.temp['selected'] = ['c1', 'c2']

    assert algo(s)
    weights = s.temp['weights']
    assert len(weights) == 2
    assert 'c1' in weights
    assert weights['c1'] == 0.5
    assert 'c2' in weights
    assert weights['c2'] == 0.5


def test_weight_specified():
    algo = algos.WeighSpecified(c1=0.6, c2=0.4)

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    weights = s.temp['weights']
    assert len(weights) == 2
    assert 'c1' in weights
    assert weights['c1'] == 0.6
    assert 'c2' in weights
    assert weights['c2'] == 0.4


def test_select_has_data():
    algo = algos.SelectHasData(min_count=3, lookback=pd.DateOffset(days=3))

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=10)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)
    data['c1'].ix[dts[0]] = np.nan
    data['c1'].ix[dts[1]] = np.nan

    s.setup(data)
    s.update(dts[2])

    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 1
    assert 'c2' in selected


def test_select_has_data_preselected():
    algo = algos.SelectHasData(min_count=3, lookback=pd.DateOffset(days=3))

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)
    data['c1'].ix[dts[0]] = np.nan
    data['c1'].ix[dts[1]] = np.nan

    s.setup(data)
    s.update(dts[2])
    s.temp['selected'] = ['c1']

    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 0


@mock.patch('bt.ffn.calc_erc_weights')
def test_weigh_erc(mock_erc):
    algo = algos.WeighERC(lookback=pd.DateOffset(days=5))

    mock_erc.return_value = pd.Series({'c1': 0.3, 'c2': 0.7})

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=5)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)

    s.setup(data)
    s.update(dts[4])
    s.temp['selected'] = ['c1', 'c2']

    assert algo(s)
    assert mock_erc.called
    rets = mock_erc.call_args[0][0]
    assert len(rets) == 4
    assert 'c1' in rets
    assert 'c2' in rets

    weights = s.temp['weights']
    assert len(weights) == 2
    assert weights['c1'] == 0.3
    assert weights['c2'] == 0.7


def test_weigh_inv_vol():
    algo = algos.WeighInvVol(lookback=pd.DateOffset(days=5))

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=5)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)

    # high vol c1
    data['c1'].ix[dts[1]] = 105
    data['c1'].ix[dts[2]] = 95
    data['c1'].ix[dts[3]] = 105
    data['c1'].ix[dts[4]] = 95

    # low vol c2
    data['c2'].ix[dts[1]] = 100.1
    data['c2'].ix[dts[2]] = 99.9
    data['c2'].ix[dts[3]] = 100.1
    data['c2'].ix[dts[4]] = 99.9

    s.setup(data)
    s.update(dts[4])
    s.temp['selected'] = ['c1', 'c2']

    assert algo(s)
    weights = s.temp['weights']
    assert len(weights) == 2
    assert weights['c2'] > weights['c1']
    aae(weights['c1'], 0.020, 3)
    aae(weights['c2'], 0.980, 3)


@mock.patch('bt.ffn.calc_mean_var_weights')
def test_weigh_mean_var(mock_mv):
    algo = algos.WeighMeanVar(lookback=pd.DateOffset(days=5))

    mock_mv.return_value = pd.Series({'c1': 0.3, 'c2': 0.7})

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=5)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)

    s.setup(data)
    s.update(dts[4])
    s.temp['selected'] = ['c1', 'c2']

    assert algo(s)
    assert mock_mv.called
    rets = mock_mv.call_args[0][0]
    assert len(rets) == 4
    assert 'c1' in rets
    assert 'c2' in rets

    weights = s.temp['weights']
    assert len(weights) == 2
    assert weights['c1'] == 0.3
    assert weights['c2'] == 0.7


def test_stat_total_return():
    algo = algos.StatTotalReturn(lookback=pd.DateOffset(days=3))

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)
    data['c1'].ix[dts[2]] = 105
    data['c2'].ix[dts[2]] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp['selected'] = ['c1', 'c2']

    assert algo(s)
    stat = s.temp['stat']
    assert len(stat) == 2
    assert stat['c1'] == 105.0 / 100 - 1
    assert stat['c2'] == 95.0 / 100 - 1


def test_select_n():
    algo = algos.SelectN(n=1, sort_descending=True)

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)
    data['c1'].ix[dts[2]] = 105
    data['c2'].ix[dts[2]] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp['stat'] = data.calc_total_return()

    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 1
    assert 'c1' in selected

    algo = algos.SelectN(n=1, sort_descending=False)
    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 1
    assert 'c2' in selected

    # return 2 we have if all_or_none false
    algo = algos.SelectN(n=3, sort_descending=False)
    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 2
    assert 'c1' in selected
    assert 'c2' in selected

    # return 0 we have if all_or_none true
    algo = algos.SelectN(n=3, sort_descending=False, all_or_none=True)
    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 0


def test_select_n_perc():
    algo = algos.SelectN(n=0.5, sort_descending=True)

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)
    data['c1'].ix[dts[2]] = 105
    data['c2'].ix[dts[2]] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp['stat'] = data.calc_total_return()

    assert algo(s)
    selected = s.temp['selected']
    assert len(selected) == 1
    assert 'c1' in selected


def test_select_momentum():
    algo = algos.SelectMomentum(n=1, lookback=pd.DateOffset(days=3))

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)
    data['c1'].ix[dts[2]] = 105
    data['c2'].ix[dts[2]] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp['selected'] = ['c1', 'c2']

    assert algo(s)
    actual = s.temp['selected']
    assert len(actual) == 1
    assert 'c1' in actual


def test_limit_deltas():
    algo = algos.LimitDeltas(0.1)

    s = bt.Strategy('s')
    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)

    s.setup(data)
    s.temp['weights'] = {'c1': 1}

    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp['weights']
    assert w['c1'] == 0.1

    s.temp['weights'] = {'c1': 0.05}
    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp['weights']
    assert w['c1'] == 0.05

    s.temp['weights'] = {'c1': 0.5, 'c2': 0.5}
    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp['weights']
    assert len(w) == 2
    assert w['c1'] == 0.1
    assert w['c2'] == 0.1

    s.temp['weights'] = {'c1': 0.5, 'c2': -0.5}
    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp['weights']
    assert len(w) == 2
    assert w['c1'] == 0.1
    assert w['c2'] == -0.1

    s.temp['weights'] = {'c1': 0.5, 'c2': -0.5}
    algo = algos.LimitDeltas({'c1': 0.1})
    assert algo(s)
    w = s.temp['weights']
    assert len(w) == 2
    assert w['c1'] == 0.1
    assert w['c2'] == -0.5

    s.temp['weights'] = {'c1': 0.5, 'c2': -0.5}
    algo = algos.LimitDeltas({'c1': 0.1, 'c2': 0.3})
    assert algo(s)
    w = s.temp['weights']
    assert len(w) == 2
    assert w['c1'] == 0.1
    assert w['c2'] == -0.3

    # set exisitng weight
    s.children['c1'] = bt.core.SecurityBase('c1')
    s.children['c1']._weight = 0.3
    s.children['c2'] = bt.core.SecurityBase('c2')
    s.children['c2']._weight = -0.7

    s.temp['weights'] = {'c1': 0.5, 'c2': -0.5}
    algo = algos.LimitDeltas(0.1)
    assert algo(s)
    w = s.temp['weights']
    assert len(w) == 2
    assert w['c1'] == 0.4
    assert w['c2'] == -0.6


def test_rebalance_over_time():
    target = mock.MagicMock()
    rb = mock.MagicMock()

    algo = algos.RebalanceOverTime(n=2)
    # patch in rb function
    algo._rb = rb

    target.temp = {}
    target.temp['weights'] = {'a': 1, 'b': 0}

    a = mock.MagicMock()
    a.weight = 0.
    b = mock.MagicMock()
    b.weight = 1.
    target.children = {'a': a, 'b': b}

    assert algo(target)
    w = target.temp['weights']
    assert len(w) == 2
    assert w['a'] == 0.5
    assert w['b'] == 0.5

    assert rb.called
    called_tgt = rb.call_args[0][0]
    called_tgt_w = called_tgt.temp['weights']
    assert len(called_tgt_w) == 2
    assert called_tgt_w['a'] == 0.5
    assert called_tgt_w['b'] == 0.5

    # update weights for next call
    a.weight = 0.5
    b.weight = 0.5

    # clear out temp - same as would Strategy
    target.temp = {}

    assert algo(target)
    w = target.temp['weights']
    assert len(w) == 2
    assert w['a'] == 1.
    assert w['b'] == 0.

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


def test_require():
    target = mock.MagicMock()
    target.temp = {}

    algo = algos.Require(lambda x: len(x) > 0, 'selected')
    assert not algo(target)

    target.temp['selected'] = []
    assert not algo(target)

    target.temp['selected'] = ['a', 'b']
    assert algo(target)


def test_run_every_n_periods():
    target = mock.MagicMock()
    target.temp = {}

    algo = algos.RunEveryNPeriods(n=3, offset=0)

    target.now = pd.to_datetime('2010-01-01')
    assert algo(target)
    # run again w/ no date change should not trigger
    assert not algo(target)

    target.now = pd.to_datetime('2010-01-02')
    assert not algo(target)

    target.now = pd.to_datetime('2010-01-03')
    assert not algo(target)

    target.now = pd.to_datetime('2010-01-04')
    assert algo(target)

    target.now = pd.to_datetime('2010-01-05')
    assert not algo(target)


def test_run_every_n_periods_offset():
    target = mock.MagicMock()
    target.temp = {}

    algo = algos.RunEveryNPeriods(n=3, offset=1)

    target.now = pd.to_datetime('2010-01-01')
    assert not algo(target)
    # run again w/ no date change should not trigger
    assert not algo(target)

    target.now = pd.to_datetime('2010-01-02')
    assert algo(target)

    target.now = pd.to_datetime('2010-01-03')
    assert not algo(target)

    target.now = pd.to_datetime('2010-01-04')
    assert not algo(target)

    target.now = pd.to_datetime('2010-01-05')
    assert algo(target)


def test_or():
    target = mock.MagicMock()
    target.temp = {}

    #run on the 1/2/18
    runOnDateAlgo = algos.RunOnDate(pd.to_datetime('2018-01-02'))
    runOnDateAlgo2 = algos.RunOnDate(pd.to_datetime('2018-01-03'))
    runOnDateAlgo3 = algos.RunOnDate(pd.to_datetime('2018-01-04'))
    runOnDateAlgo4 = algos.RunOnDate(pd.to_datetime('2018-01-04'))

    orAlgo = algos.Or([runOnDateAlgo, runOnDateAlgo2, runOnDateAlgo3, runOnDateAlgo4])

    #verify it returns false when neither is true
    target.now = pd.to_datetime('2018-01-01')
    assert not orAlgo(target)

    # verify it returns true when the first is true
    target.now = pd.to_datetime('2018-01-02')
    assert orAlgo(target)

    # verify it returns true when the second is true
    target.now = pd.to_datetime('2018-01-03')
    assert orAlgo(target)

    # verify it returns true when both algos return true
    target.now = pd.to_datetime('2018-01-04')
    assert orAlgo(target)

def test_TargetVol():

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=7)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)

    # high vol c1
    data.loc[dts[0],'c1'] = 95
    data.loc[dts[1],'c1'] = 105
    data.loc[dts[2],'c1'] = 95
    data.loc[dts[3],'c1'] = 105
    data.loc[dts[4],'c1'] = 95
    data.loc[dts[5],'c1'] = 105
    data.loc[dts[6],'c1'] = 95

    # low vol c2
    data.loc[dts[0], 'c2'] = 99
    data.loc[dts[1], 'c2'] = 101
    data.loc[dts[2], 'c2'] = 99
    data.loc[dts[3], 'c2'] = 101
    data.loc[dts[4], 'c2'] = 99
    data.loc[dts[5], 'c2'] = 101
    data.loc[dts[6], 'c2'] = 99

    targetVolAlgo = algos.TargetVol(
        0.1,
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=1),
        covar_method='standard',
        annualization_factor=1
    )

    s.setup(data)
    s.update(dts[6])
    s.temp['weights'] = {'c1':0.5, 'c2':0.5}

    assert targetVolAlgo(s)
    weights = s.temp['weights']
    assert len(weights) == 2
    assert np.isclose(weights['c2'],weights['c1'])

    unannualized_c2_weight = weights['c1']

    targetVolAlgo = algos.TargetVol(
        0.1*np.sqrt(252),
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=1),
        covar_method='standard',
        annualization_factor=252
    )

    s.setup(data)
    s.update(dts[6])
    s.temp['weights'] = {'c1': 0.5, 'c2': 0.5}

    assert targetVolAlgo(s)
    weights = s.temp['weights']
    assert len(weights) == 2
    assert np.isclose(weights['c2'], weights['c1'])

    assert np.isclose(unannualized_c2_weight, weights['c2'])


def test_PTE_Rebalance():

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=30*4)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)

    # high vol c1
    # low vol c2
    for i,dt in enumerate(dts[:-2]):
        if i % 2 == 0:
            data.loc[dt,'c1'] = 95
            data.loc[dt,'c2'] = 101
        else:
            data.loc[dt, 'c1'] = 105
            data.loc[dt, 'c2'] = 99

    dt = dts[-2]
    data.loc[dt,'c1'] = 115
    data.loc[dt,'c2'] = 97

    s.setup(data)
    s.update(dts[-2])
    s.adjust(1000000)
    s.rebalance(0.4,'c1')
    s.rebalance(0.6,'c2')

    wdf = pd.DataFrame(
        np.zeros(data.shape),
        columns=data.columns,
        index=data.index
    )

    wdf['c1'] = 0.5
    wdf['c2'] = 0.5


    PTE_rebalance_Algo = bt.algos.PTE_Rebalance(
        0.01,
        wdf,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=1),
        covar_method='standard',
        annualization_factor=252
    )

    assert PTE_rebalance_Algo(s)

    s.rebalance(0.5, 'c1')
    s.rebalance(0.5, 'c2')

    assert not PTE_rebalance_Algo(s)

