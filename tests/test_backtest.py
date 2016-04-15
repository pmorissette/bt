from __future__ import division
import bt
import pandas as pd
import mock


def test_backtest_copies_strategy():
    s = mock.MagicMock()
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data, progress_bar=False)

    assert id(s) != id(actual.strategy)


def test_backtest_dates_set():
    s = mock.MagicMock()
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data, progress_bar=False)

    assert len(actual.dates) == len(data.index)
    assert actual.dates[0] == data.index[0]
    assert actual.dates[-1] == data.index[-1]


def test_backtest_auto_name():
    s = mock.MagicMock()
    s.name = 's'
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data, progress_bar=False)

    assert actual.name == 's'


def test_initial_capital_set():
    s = mock.MagicMock()
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data, initial_capital=302, progress_bar=False)
    actual.run()

    s = actual.strategy

    s.adjust.assert_called_with(302)


def test_run_loop():
    s = mock.MagicMock()
    # run loop checks on this
    s.bankrupt = False
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data, initial_capital=302, progress_bar=False)
    actual.run()

    s = actual.strategy

    assert s.update.call_count == 10
    assert s.run.call_count == 5


def test_turnover():
    dts = pd.date_range('2010-01-01', periods=5)
    data = pd.DataFrame(index=dts, columns=['a', 'b'], data=100)

    data['a'][dts[1]] = 105
    data['b'][dts[1]] = 95

    data['a'][dts[2]] = 110
    data['b'][dts[2]] = 90

    data['a'][dts[3]] = 115
    data['b'][dts[3]] = 85

    s = bt.Strategy('s', [bt.algos.SelectAll(),
                          bt.algos.WeighEqually(),
                          bt.algos.Rebalance()])

    t = bt.Backtest(s, data, commissions=lambda x, y: 0, progress_bar=False)
    res = bt.run(t)

    t = res.backtests['s']

    # these numbers were (tediously) calculated in excel
    assert t.turnover[dts[0]] == 0. / 1000000
    assert t.turnover[dts[1]] == 24985. / 1000000
    assert t.turnover[dts[2]] == 24970. / 997490
    assert t.turnover[dts[3]] == 25160. / 992455
    assert t.turnover[dts[4]] == 76100. / 1015285
