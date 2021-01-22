from __future__ import division
import bt
import pandas as pd
import numpy as np
from nose.tools import assert_almost_equal as aae
import random
import sys
if sys.version_info < (3, 3):
    import mock
else:
    from unittest import mock


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

    # must account for 't0' addition
    assert len(actual.dates) == len(data.index) + 1
    assert actual.dates[1] == data.index[0]
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

    # account for first update call to 'setup' initial state
    assert s.update.call_count == 10 + 1
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
    assert np.allclose(t.turnover[dts[0]], 0. / 1000000)
    assert np.allclose(t.turnover[dts[1]], 24985. / 1000000)
    assert np.allclose(t.turnover[dts[2]], 24970. / 997490)
    assert np.allclose(t.turnover[dts[3]], 25160. / 992455)
    assert np.allclose(t.turnover[dts[4]], 76100. / 1015285)


def test_Results_helper_functions():

    names = ['foo', 'bar']
    dates = pd.date_range(start='2017-01-01', end='2017-12-31', freq=pd.tseries.offsets.BDay())
    n = len(dates)
    rdf = pd.DataFrame(
        np.zeros((n, len(names))),
        index=dates,
        columns=names
    )

    np.random.seed(1)
    rdf[names[0]] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)
    rdf[names[1]] = np.random.normal(loc=0.04 / n, scale=0.05 / np.sqrt(n), size=n)

    pdf = 100 * np.cumprod(1 + rdf)

    # algo to fire on the beginning of every month and to run on the first date
    runDailyAlgo = bt.algos.RunDaily(
        run_on_first_date=True
    )

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
    strat = bt.Strategy('static',
        [
            runDailyAlgo,
            weighSpecifiedAlgo,
            rebalAlgo
        ]
    )

    backtest = bt.Backtest(
        strat,
        pdf,
        integer_positions=False,
        progress_bar=False
    )

    res = bt.run(backtest)

    assert(type(res.get_security_weights()) is pd.DataFrame)

    assert (type(res.get_transactions()) is pd.DataFrame)

    assert (type(res.get_weights()) is pd.DataFrame)


def test_Results_helper_functions_fi():

    names = ['foo', 'bar']
    dates = pd.date_range(start='2017-01-01', end='2017-12-31', freq=pd.tseries.offsets.BDay())
    n = len(dates)
    rdf = pd.DataFrame(
        np.zeros((n, len(names))),
        index=dates,
        columns=names
    )

    np.random.seed(1)
    rdf[names[0]] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)
    rdf[names[1]] = np.random.normal(loc=0.04 / n, scale=0.05 / np.sqrt(n), size=n)

    pdf = 100 * np.cumprod(1 + rdf)
    notional = pd.Series( 1e6, index = pdf.index )
    
    # algo to fire on the beginning of every month and to run on the first date
    runDailyAlgo = bt.algos.RunDaily(
        run_on_first_date=True
    )

    # algo to select all securities
    selectAll = bt.algos.SelectAll()

    # algo to set the weights
    #  it will only run when runMonthlyAlgo returns true
    #  which only happens on the first of every month        
    weighRandomly = bt.algos.WeighRandomly()

    # algo to set the notional of the fixed income strategy
    setNotional = bt.algos.SetNotional( 'notional' )

    # algo to rebalance the current weights to weights set by weighSpecified
    #  will only run when weighSpecifiedAlgo returns true
    #  which happens every time it runs
    rebalAlgo = bt.algos.Rebalance()

    # a strategy that rebalances monthly to specified weights
    strat = bt.FixedIncomeStrategy('random',
        [
            runDailyAlgo,
            selectAll,
            weighRandomly,
            setNotional,
            rebalAlgo
        ]
    )

    backtest = bt.Backtest(
        strat,
        pdf,
        initial_capital = 0,
        integer_positions=False,
        progress_bar=False,
        additional_data = {'mydata':pdf, 'notional':notional}
    )
    bidoffer = 1.
    backtest2 = bt.Backtest(
        strat,
        pdf,
        initial_capital = 0,
        integer_positions=False,
        progress_bar=False,
        additional_data = {'mydata':pdf, 
                           'notional':notional,
                           'bidoffer': pd.DataFrame( bidoffer, pdf.index, pdf.columns )}
    )
    random.seed(1234)
    res = bt.run(backtest)
    random.seed(1234)
    res2 = bt.run(backtest2)

    assert(type(res.get_security_weights()) is pd.DataFrame)

    assert (type(res.get_transactions()) is pd.DataFrame)
    assert len(res.get_transactions()) > 0

    assert (type(res.get_weights()) is pd.DataFrame)

    # Make sure the insertion of the first row applies to additional data as well
    assert backtest.data.index.equals( backtest.additional_data['mydata'].index )

    # Check that bid/offer is accounted for
    transactions = res.get_transactions()        
    transactions['price'] = transactions['price'] + 0.5 * bidoffer
    assert (res2.get_transactions().price - res2.get_transactions().price).abs().sum() == 0
        
        
def test_30_min_data():
    names = ['foo']
    dates = pd.date_range(start='2017-01-01', end='2017-12-31', freq='30min')
    n = len(dates)
    rdf = pd.DataFrame(
        np.zeros((n, len(names))),
        index=dates,
        columns=names
    )

    np.random.seed(1)
    rdf[names[0]] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)

    pdf = 100 * np.cumprod(1 + rdf)

    sma50 = pdf.rolling(50).mean()
    sma200 = pdf.rolling(200).mean()

    tw = sma200.copy()
    tw[sma50 > sma200] = 1.0
    tw[sma50 <= sma200] = -1.0
    tw[sma200.isnull()] = 0.0

    ma_cross = bt.Strategy('ma_cross', [bt.algos.WeighTarget(tw), bt.algos.Rebalance()])
    t = bt.Backtest(ma_cross, pdf,progress_bar=False)
    res = bt.run(t)

    wait=1


def test_RenomalizedFixedIncomeResult():
    dts = pd.date_range('2010-01-01', periods=5)
    data = pd.DataFrame(index=dts, columns=['a'], data=1.)    
    data['a'][dts[0]] = 0.99
    data['a'][dts[1]] = 1.01
    data['a'][dts[2]] = 0.99
    data['a'][dts[3]] = 1.01
    data['a'][dts[4]] = 0.99
    
    weights = pd.DataFrame(index=dts, columns=['a'], data=1.)
    weights['a'][dts[0]] = 1.
    weights['a'][dts[1]] = 2.
    weights['a'][dts[2]] = 1.
    weights['a'][dts[3]] = 2.
    weights['a'][dts[4]] = 1.
    
    coupons = pd.DataFrame(index=dts, columns=['a'], data=0.)
    
    algos = [ bt.algos.SelectAll(),
              bt.algos.WeighTarget( weights ),
              bt.algos.SetNotional( 'notional' ),
              bt.algos.Rebalance()]
    children = [ bt.CouponPayingSecurity('a') ]
    
    s = bt.FixedIncomeStrategy('s', algos, children = children )

    t = bt.Backtest(s, data, initial_capital=0, 
                    additional_data = {'notional': pd.Series( 1e6, dts),
                                       'coupons': coupons},
                    progress_bar=False)
    res = bt.run(t)

    t = res.backtests['s']
    
    # Due to the relationship between the time varying notional and the prices,
    # the strategy has lost money, but price == 100, so "total return" is zero
    assert t.strategy.value < 0.
    assert t.strategy.price == 100.
    assert res.stats['s'].total_return == 0
    
    # Renormalizing results to a constant size "fixes" this
    norm_res = bt.backtest.RenormalizedFixedIncomeResult( 1e6, *res.backtest_list )    
    aae( norm_res.stats['s'].total_return, t.strategy.value / 1e6, 16 )
    
    # Check that using the lagged notional value series leads to the same results
    # as the original calculation. This proves that we can re-derive the price
    # series from the other data available on the strategy
    notl_values = t.strategy.notional_values.shift(1)
    notl_values[dts[0]] = 1e6 # The notional value *before* any trades are put on
    norm_res = bt.backtest.RenormalizedFixedIncomeResult( 
                                notl_values, 
                                *res.backtest_list )

    assert norm_res.stats['s'].total_return == res.stats['s'].total_return   
    assert norm_res.prices.equals( res.prices )    
    
    
