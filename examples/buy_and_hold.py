
if __name__ == "__main__":

    import numpy as np
    import pandas as pd

    import ffn
    import bt


    names = ['foo','bar','rf']
    dates = pd.date_range(start='2017-01-01',end='2017-12-31', freq=pd.tseries.offsets.BDay())
    n = len(dates)
    rdf = pd.DataFrame(
        np.zeros((n, len(names))),
        index = dates,
        columns = names
    )

    np.random.seed(1)
    rdf['foo'] = np.random.normal(loc = 0.1/n,scale=0.2/np.sqrt(n),size=n)
    rdf['bar'] = np.random.normal(loc = 0.04/n,scale=0.05/np.sqrt(n),size=n)
    rdf['rf'] = np.random.normal(loc=0.02/ n, scale=0.01 / np.sqrt(n), size=n)

    pdf = 100*np.cumprod(1+rdf)

    # algo to fire on the beginning of every month and to run on the first date
    runMonthlyAlgo = bt.algos.RunMonthly(
        run_on_first_date=True,
        run_on_end_of_period=True
    )

    # algo to set the weights in the temp dictionary\
    weights = pd.Series([0.6,0.4,0.],index = rdf.columns)
    weighSpecifiedAlgo = bt.algos.WeighSpecified(**weights)


    # algo to rebalance the current weights to weights set in temp dictionary
    rebalAlgo = bt.algos.Rebalance()

    # a strategy that rebalances monthly to specified weights
    s = 'monthly'
    strat = bt.Strategy(s,
                    [
                        runMonthlyAlgo,
                        weighSpecifiedAlgo,
                        rebalAlgo
                    ]
    )
    """
    runMonthlyAlgo will return True on the last day of the month.
    If runMonthlyAlgo returns True, then weighSpecifiedAlgo will set the weights and return True.
    If weighSpecifiedAlgo returns True, then rebalAlgo will rebalance the portfolio to match the
        target weights.
    """

    # set integer_positions=False when positions are not required to be integers(round numbers)
    backtest = bt.Backtest(
        strat,
        pdf,
        integer_positions=False
    )

    res = bt.run(backtest)

    # set riskfree as the rf index
    res.set_riskfree_rate(pdf['rf'])


    wait=1