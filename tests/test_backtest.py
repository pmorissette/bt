from __future__ import division
import bt
import pandas as pd
import mock


def test_backtest_copies_strategy():
    s = mock.MagicMock()
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data)

    assert id(s) != id(actual.strategy)


def test_backtest_dates_set():
    s = mock.MagicMock()
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data)

    assert len(actual.dates) == len(data.index)
    assert actual.dates[0] == data.index[0]
    assert actual.dates[-1] == data.index[-1]


def test_backtest_auto_name():
    s = mock.MagicMock()
    s.name = 's'
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data)

    assert actual.name == 's'


def test_initial_capital_set():
    s = mock.MagicMock()
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data, initial_capital=302)
    actual.run()

    s = actual.strategy

    s.adjust.assert_called_with(302)


def test_run_loop():
    s = mock.MagicMock()
    # run loop checks on this
    s.bankrupt = False
    data = pd.DataFrame(index=pd.date_range('2010-01-01', periods=5),
                        columns=['a', 'b'], data=100)

    actual = bt.Backtest(s, data, initial_capital=302)
    actual.run()

    s = actual.strategy

    assert s.update.call_count == 10
    assert s.run.call_count == 5
