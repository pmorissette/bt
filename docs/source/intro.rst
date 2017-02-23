
.. code:: python

    import bt


A Simple Strategy Backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create a simple strategy. We will create a monthly rebalanced, long-only strategy where we place equal weights on each asset in our universe of assets.

First, we will download some data. By default, :func:`bt.get (alias for ffn.get) <ffn.data.get>` downloads the Adjusted Close from Yahoo! Finance. We will download some data starting on January 1, 2010 for the purposes of this demo.

.. code:: python

    # fetch some data
    data = bt.get('spy,agg', start='2010-01-01')
    print data.head()


.. parsed-literal::
    :class: pynb-result

                      spy        agg
    Date                            
    2010-01-04  98.214371  84.963075
    2010-01-05  98.474354  85.349609
    2010-01-06  98.543685  85.300266
    2010-01-07  98.959667  85.201575
    2010-01-08  99.288981  85.250923



Once we have our data, we will create our strategy. The :class:`Strategy <bt.core.Strategy>` object contains the strategy logic by combining various :class:`Algos <bt.core.Algo>`. 

.. code:: python

    # create the strategy
    s = bt.Strategy('s1', [bt.algos.RunMonthly(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])


Finally, we will create a :class:`Backtest <bt.backtest.Backtest>`, which is the logical combination of a strategy with a data set.

Once this is done, we can run the backtest and analyze the results.

.. code:: python

    # create a backtest and run it
    test = bt.Backtest(s, data)
    res = bt.run(test)


.. parsed-literal::
    :class: pynb-result

    s1
    0%                          100%
    [############################# ] | ETA: 00:00:00


Now we can analyze the results of our backtest. The :class:`Result <bt.backtest.Result>` object is a thin wrapper around `ffn.GroupStats <http://pmorissette.github.io/ffn/ffn.html#ffn.core.GroupStats>`__ that adds some helper methods.

.. code:: python

    # first let's see an equity curve
    res.plot()



.. image:: _static/intro_8_0.png
    :class: pynb


.. code:: python

    # ok and what about some stats?
    res.display()


.. parsed-literal::
    :class: pynb-result

    Stat                 s1
    -------------------  ----------
    Start                2010-01-03
    End                  2017-02-22
    Risk-free rate       0.00%
    
    Total Return         81.30%
    Daily Sharpe         1.19
    Daily Sortino        1.57
    CAGR                 8.69%
    Max Drawdown         -7.83%
    Calmar Ratio         1.11
    
    MTD                  2.08%
    3m                   4.08%
    6m                   3.26%
    YTD                  3.11%
    1Y                   12.04%
    3Y (ann.)            6.82%
    5Y (ann.)            8.12%
    10Y (ann.)           8.69%
    Since Incep. (ann.)  8.69%
    
    Daily Sharpe         1.19
    Daily Sortino        1.57
    Daily Mean (ann.)    8.61%
    Daily Vol (ann.)     7.23%
    Daily Skew           -0.35
    Daily Kurt           3.80
    Best Day             2.48%
    Worst Day            -3.11%
    
    Monthly Sharpe       1.41
    Monthly Sortino      2.61
    Monthly Mean (ann.)  8.61%
    Monthly Vol (ann.)   6.10%
    Monthly Skew         0.01
    Monthly Kurt         0.18
    Best Month           5.69%
    Worst Month          -3.39%
    
    Yearly Sharpe        1.62
    Yearly Sortino       -
    Yearly Mean          7.25%
    Yearly Vol           4.46%
    Yearly Skew          0.15
    Yearly Kurt          -0.71
    Best Year            14.10%
    Worst Year           1.17%
    
    Avg. Drawdown        -0.79%
    Avg. Drawdown Days   13.31
    Avg. Up Month        1.64%
    Avg. Down Month      -1.27%
    Win Year %           100.00%
    Win 12m %            96.00%


.. code:: python

    # ok and how does the return distribution look like?
    res.plot_histogram()



.. image:: _static/intro_10_0.png
    :class: pynb


.. code:: python

    # and just to make sure everything went along as planned, let's plot the security weights over time
    res.plot_security_weights()



.. image:: _static/intro_11_0.png
    :class: pynb



Modifying a Strategy
~~~~~~~~~~~~~~~~~~~~

Now what if we ran this strategy weekly and also used some risk parity style approach by using weights that are proportional to the inverse of each asset's volatility? Well, all we have to do is plug in some different algos. See below:

.. code:: python

    # create our new strategy
    s2 = bt.Strategy('s2', [bt.algos.RunWeekly(),
                            bt.algos.SelectAll(),
                            bt.algos.WeighInvVol(),
                            bt.algos.Rebalance()])
    
    # now let's test it with the same data set. We will also compare it with our first backtest.
    test2 = bt.Backtest(s2, data)
    # we include test here to see the results side-by-side
    res2 = bt.run(test, test2)
    
    res2.plot()


.. parsed-literal::
    :class: pynb-result

    s2
    0%                          100%
    [############################# ] | ETA: 00:00:00


.. image:: _static/intro_13_1.png
    :class: pynb


.. code:: python

    res2.display()


.. parsed-literal::
    :class: pynb-result

    Stat                 s1          s2
    -------------------  ----------  ----------
    Start                2010-01-03  2010-01-03
    End                  2017-02-22  2017-02-22
    Risk-free rate       0.00%       0.00%
    
    Total Return         81.30%      40.79%
    Daily Sharpe         1.19        1.45
    Daily Sortino        1.57        2.00
    CAGR                 8.69%       4.91%
    Max Drawdown         -7.83%      -4.07%
    Calmar Ratio         1.11        1.21
    
    MTD                  2.08%       1.56%
    3m                   4.08%       2.66%
    6m                   3.26%       0.47%
    YTD                  3.11%       2.27%
    1Y                   12.04%      5.49%
    3Y (ann.)            6.82%       3.97%
    5Y (ann.)            8.12%       4.02%
    10Y (ann.)           8.69%       4.91%
    Since Incep. (ann.)  8.69%       4.91%
    
    Daily Sharpe         1.19        1.45
    Daily Sortino        1.57        2.00
    Daily Mean (ann.)    8.61%       4.85%
    Daily Vol (ann.)     7.23%       3.34%
    Daily Skew           -0.35       -0.29
    Daily Kurt           3.80        2.87
    Best Day             2.48%       1.20%
    Worst Day            -3.11%      -1.13%
    
    Monthly Sharpe       1.41        1.68
    Monthly Sortino      2.61        2.61
    Monthly Mean (ann.)  8.61%       5.04%
    Monthly Vol (ann.)   6.10%       3.00%
    Monthly Skew         0.01        -0.59
    Monthly Kurt         0.18        0.03
    Best Month           5.69%       1.91%
    Worst Month          -3.39%      -2.09%
    
    Yearly Sharpe        1.62        1.61
    Yearly Sortino       -           -
    Yearly Mean          7.25%       4.08%
    Yearly Vol           4.46%       2.53%
    Yearly Skew          0.15        -0.45
    Yearly Kurt          -0.71       -0.03
    Best Year            14.10%      7.02%
    Worst Year           1.17%       -0.13%
    
    Avg. Drawdown        -0.79%      -0.40%
    Avg. Drawdown Days   13.31       13.28
    Avg. Up Month        1.64%       0.83%
    Avg. Down Month      -1.27%      -0.69%
    Win Year %           100.00%     85.71%
    Win 12m %            96.00%      94.67%

