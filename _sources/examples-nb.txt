
                
SMA Strategy
------------

Let's start off with a Simple Moving Average (SMA) strategy. We will start with a simple version of the strategy, namely:

* **Select** the securities that are currently above their 50 day moving average
* **Weigh** each selected security equally
* **Rebalance** the portfolio to reflect the target weights

This should be pretty simple to build. The only thing missing above is the calculation of the simple moving average. When should this take place? 

Given the flexibility of **bt**, there is no strict rule. The average calculation could be performed in an Algo, but that would be pretty inefficient. A better way would be to calculate the moving average at the beginning - before starting the backtest. After all, all the data is known in advance. 

Now that we know what we have to do, let's get started. First we will download some data and calculate the simple moving average.
                
.. code:: python

    import bt
    #%pylab inline
.. code:: python

    # download data
    data = bt.get('aapl,msft,c,gs,ge', start='2010-01-01')
    
    # calculate moving average DataFrame using pandas' rolling_mean
    import pandas as pd
    # a rolling mean is a moving average, right?
    sma = pd.rolling_mean(data, 50)
                
It's always a good idea to plot your data to make sure it looks ok. So let's see how the data + sma plot looks like.
                
.. code:: python

    # let's see what the data looks like - this is by no means a pretty chart, but it does the job
    plot = bt.merge(data, sma).plot(figsize=(15, 5))


.. image:: _static/examples-nb_4_0.png
    :class: pynb


                
Looks legit.

Now that we have our data, we will need to create our security selection logic. Let's create a basic Algo that will select the securities that are above their moving average.

Before we do that, let's think about how we will code it. We could pass the SMA data and then extract the row (from the sma DataFrame) on the current date, compare the values to the current prices, and then keep a list of those securities where the price is above the SMA. This is the most straightforward approach. However, this is not very re-usable because the logic within the Algo will be quite specific to the task at hand and if we wish to change the logic, we will have to write a new algo. 

For example, what if we wanted to select securities that were below their sma? Or what if we only wanted securities that were 5% above their sma?

What we could do instead is pre-calculate the selection logic DataFrame (a fast, vectorized operation) and write a generic Algo that takes in this boolean DataFrame and returns the securities where the value is True on a given date. This will be must faster and much more reusable. Let's see how the implementation looks like.
                
.. code:: python

    class SelectWhere(bt.Algo):
        
        """
        Selects securities based on an indicator DataFrame.
        
        Selects securities where the value is True on the current date (target.now).
        
        Args:
            * signal (DataFrame): DataFrame containing the signal (boolean DataFrame)
        
        Sets:
            * selected
        
        """
        def __init__(self, signal):
            self.signal = signal
            
        def __call__(self, target):
            # get signal on target.now
            if target.now in self.signal.index:
                sig = self.signal.ix[target.now]
    
                # get indices where true as list
                selected = list(sig.index[sig])
    
                # save in temp - this will be used by the weighing algo
                target.temp['selected'] = selected
            
            # return True because we want to keep on moving down the stack
            return True
                
So there we have it. Our selection Algo. 

.. note:: 

    By the way, this Algo already exists - I just wanted to show you how you would code it from scratch. 
    :class:`Here is the code <bt.algos.SelectWhere>`.

All we have to do now is pass in a signal matrix. In our case, it's quite easy::

    signal = data > sma

Simple, concise and more importantly, fast! Let's move on and test the strategy. 
                
.. code:: python

    # first we create the Strategy
    s = bt.Strategy('above50sma', [SelectWhere(data > sma),
                                   bt.algos.WeighEqually(),
                                   bt.algos.Rebalance()])
    
    # now we create the Backtest
    t = bt.Backtest(s, data)
    
    # and let's run it!
    res = bt.run(t)
                
So just to recap, we created the strategy, created the backtest by joining Strategy+Data, and ran the backtest. Let's see the results.
                
.. code:: python

    # what does the equity curve look like?
    res.plot('d')


.. image:: _static/examples-nb_10_0.png
    :class: pynb


.. code:: python

    # and some performance stats
    res.display()

.. parsed-literal::
    :class: pynb-result

    Stat                 above50sma
    -------------------  ------------
    Start                2010-01-04
    End                  2014-07-17
    
    Total Return         19.92%
    Daily Sharpe         0.30
    CAGR                 4.09%
    Max Drawdown         -33.75%
    
    MTD                  1.92%
    3m                   7.17%
    6m                   -4.39%
    YTD                  -7.68%
    1Y                   -4.46%
    3Y (ann.)            12.30%
    5Y (ann.)            4.09%
    10Y (ann.)           4.09%
    Since Incep. (ann.)  4.09%
    
    Daily Sharpe         0.30
    Daily Mean (ann.)    5.91%
    Daily Vol (ann.)     19.43%
    Daily Skew           -0.70
    Daily Kurt           4.42
    Best Day             5.77%
    Worst Day            -8.00%
    
    Monthly Sharpe       0.29
    Monthly Mean (ann.)  6.66%
    Monthly Vol (ann.)   22.72%
    Monthly Skew         -0.78
    Monthly Kurt         0.36
    Best Month           12.57%
    Worst Month          -16.28%
    
    Yearly Sharpe        0.35
    Yearly Mean          8.33%
    Yearly Vol           23.68%
    Yearly Skew          0.06
    Yearly Kurt          -4.59
    Best Year            33.39%
    Worst Year           -15.66%
    
    Avg. Drawdown        -4.68%
    Avg. Drawdown Days   84.83
    Avg. Up Month        4.68%
    Avg. Down Month      -5.93%
    Win Year %           50.00%
    Win 12m %            61.36%


                
Nothing stellar but at least you learnt something along the way (I hope). 

Oh, and one more thing. If you were to write your own "library" of backtests, you might want to write yourself a helper function that would allow you to test different parameters and securities. That function might look something like this:
                
.. code:: python

    def above_sma(tickers, sma_per=50, start='2010-01-01', name='above_sma'):
        """
        Long securities that are above their n period 
        Simple Moving Averages with equal weights.
        """
        # download data
        data = bt.get(tickers, start=start)
        # calc sma
        sma = pd.rolling_mean(data, sma_per)
    
        # create strategy
        s = bt.Strategy(name, [SelectWhere(data > sma),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])    
    
        # now we create the backtest
        return bt.Backtest(s, data)
                
This function allows us to easily generate backtests. We could easily compare a few different SMA periods. Also, let's see if we can beat a long-only allocation to the SPY.
                
.. code:: python

    # simple backtest to test long-only allocation
    def long_only_ew(tickers, start='2010-01-01', name='long_only_ew'):
        s = bt.Strategy(name, [bt.algos.RunOnce(),
                               bt.algos.SelectAll(),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])
        data = bt.get(tickers, start=start)
        return bt.Backtest(s, data)
    
    # create the backtests
    tickers = 'aapl,msft,c,gs,ge'
    sma10 = above_sma(tickers, sma_per=10, name='sma10')
    sma20 = above_sma(tickers, sma_per=20, name='sma20')
    sma40 = above_sma(tickers, sma_per=40, name='sma40')
    benchmark = long_only_ew('spy', name='spy')
    
    # run all the backtests!
    res2 = bt.run(sma10, sma20, sma40, benchmark)
.. code:: python

    res2.plot()


.. image:: _static/examples-nb_16_0.png
    :class: pynb


.. code:: python

    res2.display()

.. parsed-literal::
    :class: pynb-result

    Stat                 sma10       sma20       sma40       spy
    -------------------  ----------  ----------  ----------  ----------
    Start                2010-01-04  2010-01-04  2010-01-04  2010-01-04
    End                  2014-07-17  2014-07-17  2014-07-17  2014-07-17
    
    Total Return         28.92%      44.81%      33.33%      89.21%
    Daily Sharpe         0.38        0.53        0.43        0.95
    CAGR                 5.77%       8.51%       6.55%       15.11%
    Max Drawdown         -24.99%     -27.91%     -35.40%     -18.61%
    
    MTD                  1.86%       0.89%       1.94%       -0.01%
    3m                   5.48%       3.96%       6.79%       5.50%
    6m                   -1.27%      -7.90%      4.78%       7.56%
    YTD                  -3.57%      -10.33%     1.50%       6.95%
    1Y                   4.38%       0.77%       3.11%       18.82%
    3Y (ann.)            8.39%       12.72%      10.78%      16.86%
    5Y (ann.)            5.77%       8.51%       6.55%       15.11%
    10Y (ann.)           5.77%       8.51%       6.55%       15.11%
    Since Incep. (ann.)  5.77%       8.51%       6.55%       15.11%
    
    Daily Sharpe         0.38        0.53        0.43        0.95
    Daily Mean (ann.)    7.58%       9.99%       8.15%       15.40%
    Daily Vol (ann.)     19.82%      19.00%      18.91%      16.20%
    Daily Skew           -0.41       -0.61       -0.51       -0.41
    Daily Kurt           6.90        4.37        3.10        4.30
    Best Day             9.54%       5.77%       5.77%       4.65%
    Worst Day            -8.00%      -8.00%      -5.68%      -6.51%
    
    Monthly Sharpe       0.45        0.50        0.40        1.23
    Monthly Mean (ann.)  9.51%       10.39%      8.84%       16.33%
    Monthly Vol (ann.)   21.09%      20.80%      22.17%      13.30%
    Monthly Skew         -0.38       -0.20       -0.30       -0.30
    Monthly Kurt         1.77        -0.15       -0.29       0.40
    Best Month           18.24%      12.48%      12.90%      10.92%
    Worst Month          -17.36%     -14.74%     -15.31%     -7.94%
    
    Yearly Sharpe        0.32        0.33        0.32        1.07
    Yearly Mean          6.48%       9.34%       7.18%       14.28%
    Yearly Vol           20.57%      28.31%      22.42%      13.35%
    Yearly Skew          -0.09       0.64        -0.73       1.00
    Yearly Kurt          -2.83       -2.09       -0.96       0.33
    Best Year            28.84%      44.96%      28.64%      32.30%
    Worst Year           -16.88%     -16.26%     -21.80%     1.90%
    
    Avg. Drawdown        -6.41%      -4.64%      -5.23%      -1.76%
    Avg. Drawdown Days   87.56       49.45       140.82      17.28
    Avg. Up Month        4.34%       5.01%       5.01%       3.38%
    Avg. Down Month      -4.79%      -4.72%      -5.48%      -3.04%
    Win Year %           50.00%      50.00%      75.00%      100.00%
    Win 12m %            77.27%      75.00%      61.36%      95.45%


                
And there you have it. Beating the market ain't that easy!
                
                
SMA Crossover Strategy
----------------------

Let's build on the last section to test a moving average crossover strategy. The easiest way to achieve this is to build an Algo similar to SelectWhere, but for the purpose of setting target weights. Let's call this algo WeighTarget. This algo will take a DataFrame of target weights that we will pre-calculate. 

Basically, when the 50 day moving average will be above the 200-day moving average, we will be long (+1 target weight). Conversely, when the 50 is below the 200, we will be short (-1 target weight). 

Here's the WeighTarget implementation (this Algo also already exists in the algos module):
                
.. code:: python

    class WeighTarget(bt.Algo):
        """
        Sets target weights based on a target weight DataFrame.
        
        Args:
            * target_weights (DataFrame): DataFrame containing the target weights
        
        Sets:
            * weights
        
        """
        
        def __init__(self, target_weights):
            self.tw = target_weights
        
        def __call__(self, target):
            # get target weights on date target.now
            if target.now in self.tw.index:
                w = self.tw.ix[target.now]                
    
                # save in temp - this will be used by the weighing algo
                # also dropping any na's just in case they pop up
                target.temp['weights'] = w.dropna()
            
            # return True because we want to keep on moving down the stack
            return True
                
So let's start with a simple 50-200 day sma crossover for a single security.
                
.. code:: python

    ## download some data & calc SMAs
    data = bt.get('spy', start='2010-01-01')
    sma50 = pd.rolling_mean(data, 50)
    sma200 = pd.rolling_mean(data, 200)
    
    ## now we need to calculate our target weight DataFrame
    # first we will copy the sma200 DataFrame since our weights will have the same strucutre
    tw = sma200.copy()
    # set appropriate target weights
    tw[sma50 > sma200] = 1.0
    tw[sma50 <= sma200] = -1.0
    # here we will set the weight to 0 - this is because the sma200 needs 200 data points before
    # calculating its first point. Therefore, it will start with a bunch of nulls (NaNs).
    tw[sma200.isnull()] = 0.0
                
Ok so we downloaded our data, calculated the simple moving averages, and then we setup our target weight (tw) DataFrame. Let's take a look at our target weights to see if they make any sense.
                
.. code:: python

    # plot the target weights + chart of price & SMAs
    tmp = bt.merge(tw, data, sma50, sma200)
    tmp.columns = ['tw', 'price', 'sma50', 'sma200']
    ax = tmp.plot(figsize=(15,5), secondary_y=['tw'])


.. image:: _static/examples-nb_24_0.png
    :class: pynb


                
As mentioned earlier, it's always a good idea to plot your strategy data. It is usually easier to spot logic/programming errors this way, especially when dealing with lots of data. 

Now let's move on with the Strategy & Backtest. 
                
.. code:: python

    ma_cross = bt.Strategy('ma_cross', [WeighTarget(tw),
                                        bt.algos.Rebalance()])
    
    t = bt.Backtest(ma_cross, data)
    res = bt.run(t)
.. code:: python

    res.plot()


.. image:: _static/examples-nb_27_0.png
    :class: pynb


                
Ok great so there we have our basic moving average crossover strategy. 

Exploring the Tree Structure
----------------------------

So far, we have explored strategies that allocate capital to securities. But what if we wanted to test a strategy that allocated capital to sub-strategies?

The most straightforward way would be to test the different sub-strategies, extract their equity curves and create "synthetic securities" that would basically just represent the returns achieved from allocating capital to the different sub-strategies.

Let's see how this looks:
                
.. code:: python

    # first let's create a helper function to create a ma cross backtest
    def ma_cross(ticker, start='2010-01-01', 
                 short_ma=50, long_ma=200, name='ma_cross'):
        # these are all the same steps as above
        data = bt.get(ticker, start=start)
        short_sma = pd.rolling_mean(data, short_ma)
        long_sma  = pd.rolling_mean(data, long_ma)
    
        # target weights
        tw = long_sma.copy()
        tw[short_sma > long_sma] = 1.0
        tw[short_sma <= long_sma] = -1.0    
        tw[long_sma.isnull()] = 0.0
        
        # here we specify the children (3rd) arguemnt to make sure the strategy
        # has the proper universe. This is necessary in strategies of strategies
        s = bt.Strategy(name, [WeighTarget(tw), bt.algos.Rebalance()], [ticker])
    
        return bt.Backtest(s, data)
    
    # ok now let's create a few backtests and gather the results.
    # these will later become our "synthetic securities"
    t1 = ma_cross('aapl', name='aapl_ma_cross')
    t2 = ma_cross('msft', name='msft_ma_cross')
    
    # let's run these strategies now
    res = bt.run(t1, t2)
    
    # now that we have run the strategies, let's extract
    # the data to create "synthetic securities"
    data = bt.merge(res['aapl_ma_cross'].prices, res['msft_ma_cross'].prices)
    
    # now we have our new data. This data is basically the equity
    # curves of both backtested strategies. Now we can just use this
    # to test any old strategy, just like before.
    s = bt.Strategy('s', [bt.algos.SelectAll(),
                          bt.algos.WeighInvVol(),
                          bt.algos.Rebalance()])
    
    # create and run
    t = bt.Backtest(s, data)
    res = bt.run(t)
.. code:: python

    res.plot()


.. image:: _static/examples-nb_30_0.png
    :class: pynb


.. code:: python

    res.plot_weights()


.. image:: _static/examples-nb_31_0.png
    :class: pynb


                
As we can see above, the process is a bit more involved, but it works. It is not very elegant though, and obtaining security-level allocation information is problematic. 

Luckily, bt has built-in functionality for dealing with strategies of strategies. It uses the same general principal as demonstrated above but does it seamlessly. Basically, when a strategy is a child of another strategy, it will create a "paper trade" version of itself internally. As we run our strategy, it will run its internal "paper version" and use the returns from that strategy to populate the **price** property.

This means that the parent strategy can use the price information (which reflects the returns of the strategy had it been employed) to determine the appropriate allocation. Again, this is basically the same process as above, just packed into 1 step.

Perhaps some code will help:
                
.. code:: python

    # once again, we will create a few backtests
    # these will be the child strategies
    t1 = ma_cross('aapl', name='aapl_ma_cross')
    t2 = ma_cross('msft', name='msft_ma_cross')
    
    # let's extract the data object
    data = bt.merge(t1.data, t2.data)
    
    # now we create the parent strategy
    # we specify the children to be the two 
    # strategies created above
    s = bt.Strategy('s', [bt.algos.SelectAll(),
                          bt.algos.WeighInvVol(),
                          bt.algos.Rebalance()],
                    [t1.strategy, t2.strategy])
    
    # create and run
    t = bt.Backtest(s, data)
    res = bt.run(t)
.. code:: python

    res.plot()


.. image:: _static/examples-nb_34_0.png
    :class: pynb


.. code:: python

    res.plot_weights()


.. image:: _static/examples-nb_35_0.png
    :class: pynb


                
So there you have it. Simpler, and more complete. 
                