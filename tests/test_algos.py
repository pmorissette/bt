from datetime import datetime

import mock
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


def test_run_weekly():
    algo = algos.RunWeekly()

    target = mock.MagicMock()

    target.now = None
    assert not algo(target)

    target.now = datetime(2010, 1, 1)
    assert not algo(target)

    target.now = datetime(2010, 1, 15)
    assert algo(target)

    target.now = datetime(2010, 2, 15)
    assert algo(target)

    # sat
    target.now = datetime(2014, 1, 4)
    assert algo(target)

    # sun
    target.now = datetime(2014, 1, 5)
    assert not algo(target)

    # mon - week change
    target.now = datetime(2014, 1, 6)
    assert algo(target)


def test_run_monthly():
    algo = algos.RunMonthly()

    target = mock.MagicMock()

    target.now = None
    assert not algo(target)

    target.now = datetime(2010, 1, 1)
    assert not algo(target)

    target.now = datetime(2010, 1, 15)
    assert not algo(target)

    target.now = datetime(2010, 2, 15)
    assert algo(target)

    target.now = datetime(2010, 2, 25)
    assert not algo(target)

    target.now = datetime(2010, 12, 25)
    assert algo(target)

    target.now = datetime(2011, 1, 25)
    assert algo(target)


def test_run_yearly():
    algo = algos.RunYearly()

    target = mock.MagicMock()

    target.now = datetime(2010, 1, 1)
    actual = algo(target)
    assert not actual

    target.now = datetime(2010, 5, 1)
    actual = algo(target)
    assert not actual

    target.now = datetime(2011, 1, 1)
    actual = algo(target)
    assert actual


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

    s.algo_data['weights'] = {'c1': 1}

    assert algo(s)
    assert s.value == 999
    assert s.capital == -1
    c1 = s['c1']
    assert c1.value == 1000
    assert c1.position == 10
    assert c1.weight == 1000.0 / 999

    s.algo_data['weights'] = {'c2': 1}

    assert algo(s)
    assert s.value == 997
    assert s.capital == 97
    c2 = s['c2']
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 900
    assert c2.position == 9
    assert c2.weight == 900.0 / 997


def test_select_all():
    algo = algos.SelectAll()

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.algo_data['selected']
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
    s.algo_data['selected'] = ['c1', 'c2']

    assert algo(s)
    weights = s.algo_data['weights']
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
    weights = s.algo_data['weights']
    assert len(weights) == 2
    assert 'c1' in weights
    assert weights['c1'] == 0.6
    assert 'c2' in weights
    assert weights['c2'] == 0.4


def test_select_has_data():
    algo = algos.SelectHasData(min_count=3, lookback=pd.DateOffset(days=3))

    s = bt.Strategy('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100.)
    data['c1'].ix[dts[0]] = np.nan
    data['c1'].ix[dts[1]] = np.nan

    s.setup(data)
    s.update(dts[2])

    assert algo(s)
    selected = s.algo_data['selected']
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
    s.algo_data['selected'] = ['c1']

    assert algo(s)
    selected = s.algo_data['selected']
    assert len(selected) == 0


def test_weight_inv_vol():
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
    s.algo_data['selected'] = ['c1', 'c2']

    assert algo(s)
    weights = s.algo_data['weights']
    assert len(weights) == 2
    assert weights['c2'] > weights['c1']
    aae(weights['c1'], 0.020, 3)
    aae(weights['c2'], 0.980, 3)
