.. code:: ipython3

    import bt

.. code:: ipython3

    %matplotlib inline


A Simple Strategy Backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create a simple strategy. We will create a monthly rebalanced, long-only strategy where we place equal weights on each asset in our universe of assets.

First, we will download some data. By default, :func:`bt.get (alias for ffn.get) <ffn.data.get>` downloads the Adjusted Close from Yahoo! Finance. We will download some data starting on January 1, 2010 for the purposes of this demo.

.. code:: ipython3

    # fetch some data
    data = bt.get('spy,agg', start='2010-01-01')
    print(data.head())


.. parsed-literal::
   :class: pynb-result

                      spy        agg
    Date                            
    2010-01-04  89.225433  74.942795
    2010-01-05  89.461586  75.283775
    2010-01-06  89.524582  75.240242
    2010-01-07  89.902512  75.153168
    2010-01-08  90.201645  75.196709



Once we have our data, we will create our strategy. The :class:`Strategy <bt.core.Strategy>` object contains the strategy logic by combining various :class:`Algos <bt.core.Algo>`. 

.. code:: ipython3

    # create the strategy
    s = bt.Strategy('s1', [bt.algos.RunMonthly(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])


Finally, we will create a :class:`Backtest <bt.backtest.Backtest>`, which is the logical combination of a strategy with a data set.

Once this is done, we can run the backtest and analyze the results.

.. code:: ipython3

    # create a backtest and run it
    test = bt.Backtest(s, data)
    res = bt.run(test)


Now we can analyze the results of our backtest. The :class:`Result <bt.backtest.Result>` object is a thin wrapper around `ffn.GroupStats <http://pmorissette.github.io/ffn/ffn.html#ffn.core.GroupStats>`__ that adds some helper methods.

.. code:: ipython3

    # first let's see an equity curve
    res.plot();



.. image:: _static/intro_9_0.png
   :class: pynb
   :width: 877px
   :height: 302px


.. code:: ipython3

    # ok and what about some stats?
    res.display()


.. parsed-literal::
   :class: pynb-result

    Stat                 s1
    -------------------  ----------
    Start                2010-01-03
    End                  2022-06-30
    Risk-free rate       0.00%
    
    Total Return         151.12%
    Daily Sharpe         0.90
    Daily Sortino        1.36
    CAGR                 7.65%
    Max Drawdown         -18.42%
    Calmar Ratio         0.42
    
    MTD                  -4.58%
    3m                   -10.89%
    6m                   -14.83%
    YTD                  -14.71%
    1Y                   -9.78%
    3Y (ann.)            5.18%
    5Y (ann.)            6.47%
    10Y (ann.)           7.38%
    Since Incep. (ann.)  7.65%
    
    Daily Sharpe         0.90
    Daily Sortino        1.36
    Daily Mean (ann.)    7.75%
    Daily Vol (ann.)     8.62%
    Daily Skew           -0.98
    Daily Kurt           16.56
    Best Day             4.77%
    Worst Day            -6.63%
    
    Monthly Sharpe       1.07
    Monthly Sortino      1.93
    Monthly Mean (ann.)  7.87%
    Monthly Vol (ann.)   7.37%
    Monthly Skew         -0.37
    Monthly Kurt         1.54
    Best Month           7.57%
    Worst Month          -6.44%
    
    Yearly Sharpe        0.82
    Yearly Sortino       1.77
    Yearly Mean          7.49%
    Yearly Vol           9.15%
    Yearly Skew          -1.33
    Yearly Kurt          2.24
    Best Year            19.64%
    Worst Year           -14.71%
    
    Avg. Drawdown        -0.84%
    Avg. Drawdown Days   13.23
    Avg. Up Month        1.71%
    Avg. Down Month      -1.79%
    Win Year %           83.33%
    Win 12m %            94.24%


.. code:: ipython3

    # ok and how does the return distribution look like?
    res.plot_histogram()



.. image:: _static/intro_11_0.png
   :class: pynb
   :width: 891px
   :height: 318px


.. code:: ipython3

    # and just to make sure everything went along as planned, let's plot the security weights over time
    res.plot_security_weights()



.. image:: _static/intro_12_0.png
   :class: pynb
   :width: 874px
   :height: 290px



Modifying a Strategy
~~~~~~~~~~~~~~~~~~~~

Now what if we ran this strategy weekly and also used some risk parity style approach by using weights that are proportional to the inverse of each asset's volatility? Well, all we have to do is plug in some different algos. See below:

.. code:: ipython3

    # create our new strategy
    s2 = bt.Strategy('s2', [bt.algos.RunWeekly(),
                            bt.algos.SelectAll(),
                            bt.algos.WeighInvVol(),
                            bt.algos.Rebalance()])
    
    # now let's test it with the same data set. We will also compare it with our first backtest.
    test2 = bt.Backtest(s2, data)
    # we include test here to see the results side-by-side
    res2 = bt.run(test, test2)
    
    res2.plot();



.. image:: _static/intro_14_0.png
   :class: pynb
   :width: 877px
   :height: 302px


.. code:: ipython3

    res2.display()


.. parsed-literal::
   :class: pynb-result

    Stat                 s1          s2
    -------------------  ----------  ----------
    Start                2010-01-03  2010-01-03
    End                  2022-06-30  2022-06-30
    Risk-free rate       0.00%       0.00%
    
    Total Return         151.12%     69.32%
    Daily Sharpe         0.90        0.96
    Daily Sortino        1.36        1.41
    CAGR                 7.65%       4.31%
    Max Drawdown         -18.42%     -14.62%
    Calmar Ratio         0.42        0.29
    
    MTD                  -4.58%      -2.76%
    3m                   -10.89%     -7.46%
    6m                   -14.83%     -12.20%
    YTD                  -14.71%     -12.13%
    1Y                   -9.78%      -10.10%
    3Y (ann.)            5.18%       1.79%
    5Y (ann.)            6.47%       3.30%
    10Y (ann.)           7.38%       3.75%
    Since Incep. (ann.)  7.65%       4.31%
    
    Daily Sharpe         0.90        0.96
    Daily Sortino        1.36        1.41
    Daily Mean (ann.)    7.75%       4.32%
    Daily Vol (ann.)     8.62%       4.50%
    Daily Skew           -0.98       -2.21
    Daily Kurt           16.56       46.11
    Best Day             4.77%       2.84%
    Worst Day            -6.63%      -4.66%
    
    Monthly Sharpe       1.07        1.14
    Monthly Sortino      1.93        1.89
    Monthly Mean (ann.)  7.87%       4.41%
    Monthly Vol (ann.)   7.37%       3.89%
    Monthly Skew         -0.37       -1.03
    Monthly Kurt         1.54        3.86
    Best Month           7.57%       4.05%
    Worst Month          -6.44%      -5.04%
    
    Yearly Sharpe        0.82        0.65
    Yearly Sortino       1.77        1.18
    Yearly Mean          7.49%       4.12%
    Yearly Vol           9.15%       6.34%
    Yearly Skew          -1.33       -1.50
    Yearly Kurt          2.24        3.43
    Best Year            19.64%      11.71%
    Worst Year           -14.71%     -12.13%
    
    Avg. Drawdown        -0.84%      -0.48%
    Avg. Drawdown Days   13.23       13.67
    Avg. Up Month        1.71%       0.91%
    Avg. Down Month      -1.79%      -0.92%
    Win Year %           83.33%      83.33%
    Win 12m %            94.24%      92.09%

