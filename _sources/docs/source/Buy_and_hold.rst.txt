Buy and Hold Strategy
---------------------

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    import ffn
    import bt
    
    %matplotlib inline

Create Fake Index Data
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

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
    rdf['rf'] = 0.
    
    pdf = 100*np.cumprod(1+rdf)
    pdf.plot();



.. image:: _static/Buy_and_hold_3_0.png
   :class: pynb
   :width: 377px
   :height: 262px


Build Strategy
~~~~~~~~~~~~~~

.. code:: ipython3

    # algo to fire on the beginning of every month and to run on the first date
    runMonthlyAlgo = bt.algos.RunMonthly(
        run_on_first_date=True
    )
    
    # algo to set the weights
    #  it will only run when runMonthlyAlgo returns true
    #  which only happens on the first of every month
    weights = pd.Series([0.6,0.4,0.],index = rdf.columns)
    weighSpecifiedAlgo = bt.algos.WeighSpecified(**weights)
    
    # algo to rebalance the current weights to weights set by weighSpecified
    #  will only run when weighSpecifiedAlgo returns true
    #  which happens every time it runs
    rebalAlgo = bt.algos.Rebalance()
    
    # a strategy that rebalances monthly to specified weights
    strat = bt.Strategy('static',
        [
            runMonthlyAlgo,
            weighSpecifiedAlgo,
            rebalAlgo
        ]
    )

Run Backtest
~~~~~~~~~~~~

Note: The logic of the strategy is seperate from the data used in the
backtest.

.. code:: ipython3

    # set integer_positions=False when positions are not required to be integers(round numbers)
    backtest = bt.Backtest(
        strat,
        pdf,
        integer_positions=False
    )
    
    res = bt.run(backtest)

.. code:: ipython3

    res.stats




.. raw:: html

    <div class="pynb-result">
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>static</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>start</th>
          <td>2017-01-01 00:00:00</td>
        </tr>
        <tr>
          <th>end</th>
          <td>2017-12-29 00:00:00</td>
        </tr>
        <tr>
          <th>rf</th>
          <td>0.0</td>
        </tr>
        <tr>
          <th>total_return</th>
          <td>0.229372</td>
        </tr>
        <tr>
          <th>cagr</th>
          <td>0.231653</td>
        </tr>
        <tr>
          <th>max_drawdown</th>
          <td>-0.069257</td>
        </tr>
        <tr>
          <th>calmar</th>
          <td>3.344851</td>
        </tr>
        <tr>
          <th>mtd</th>
          <td>-0.000906</td>
        </tr>
        <tr>
          <th>three_month</th>
          <td>0.005975</td>
        </tr>
        <tr>
          <th>six_month</th>
          <td>0.142562</td>
        </tr>
        <tr>
          <th>ytd</th>
          <td>0.229372</td>
        </tr>
        <tr>
          <th>one_year</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>three_year</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>five_year</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>ten_year</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>incep</th>
          <td>0.231653</td>
        </tr>
        <tr>
          <th>daily_sharpe</th>
          <td>1.804549</td>
        </tr>
        <tr>
          <th>daily_sortino</th>
          <td>3.306154</td>
        </tr>
        <tr>
          <th>daily_mean</th>
          <td>0.206762</td>
        </tr>
        <tr>
          <th>daily_vol</th>
          <td>0.114578</td>
        </tr>
        <tr>
          <th>daily_skew</th>
          <td>0.012208</td>
        </tr>
        <tr>
          <th>daily_kurt</th>
          <td>-0.04456</td>
        </tr>
        <tr>
          <th>best_day</th>
          <td>0.020402</td>
        </tr>
        <tr>
          <th>worst_day</th>
          <td>-0.0201</td>
        </tr>
        <tr>
          <th>monthly_sharpe</th>
          <td>2.806444</td>
        </tr>
        <tr>
          <th>monthly_sortino</th>
          <td>15.352486</td>
        </tr>
        <tr>
          <th>monthly_mean</th>
          <td>0.257101</td>
        </tr>
        <tr>
          <th>monthly_vol</th>
          <td>0.091611</td>
        </tr>
        <tr>
          <th>monthly_skew</th>
          <td>0.753881</td>
        </tr>
        <tr>
          <th>monthly_kurt</th>
          <td>0.456278</td>
        </tr>
        <tr>
          <th>best_month</th>
          <td>0.073657</td>
        </tr>
        <tr>
          <th>worst_month</th>
          <td>-0.014592</td>
        </tr>
        <tr>
          <th>yearly_sharpe</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>yearly_sortino</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>yearly_mean</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>yearly_vol</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>yearly_skew</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>yearly_kurt</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>best_year</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>worst_year</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>avg_drawdown</th>
          <td>-0.016052</td>
        </tr>
        <tr>
          <th>avg_drawdown_days</th>
          <td>12.695652</td>
        </tr>
        <tr>
          <th>avg_up_month</th>
          <td>0.03246</td>
        </tr>
        <tr>
          <th>avg_down_month</th>
          <td>-0.008001</td>
        </tr>
        <tr>
          <th>win_year_perc</th>
          <td>NaN</td>
        </tr>
        <tr>
          <th>twelve_month_win_perc</th>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    res.prices.head()




.. raw:: html

    <div class="pynb-result">
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>static</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2017-01-01</th>
          <td>100.000000</td>
        </tr>
        <tr>
          <th>2017-01-02</th>
          <td>100.000000</td>
        </tr>
        <tr>
          <th>2017-01-03</th>
          <td>99.384719</td>
        </tr>
        <tr>
          <th>2017-01-04</th>
          <td>99.121677</td>
        </tr>
        <tr>
          <th>2017-01-05</th>
          <td>98.316364</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    res.plot_security_weights()



.. image:: _static/Buy_and_hold_10_0.png
   :class: pynb
   :width: 876px
   :height: 297px


Strategy value over time

.. code:: ipython3

    performanceStats = res['static']
    #performance stats is an ffn object
    res.backtest_list[0].strategy.values.plot();



.. image:: _static/Buy_and_hold_12_0.png
   :class: pynb
   :width: 380px
   :height: 259px


Strategy Outlays

Outlays are the total dollar amount spent(gained) by a purchase(sale) of
securities.

.. code:: ipython3

    res.backtest_list[0].strategy.outlays.plot();



.. image:: _static/Buy_and_hold_14_0.png
   :class: pynb
   :width: 395px
   :height: 248px


You can get the change in number of shares purchased a

.. code:: ipython3

    security_names = res.backtest_list[0].strategy.outlays.columns
    
    
    res.backtest_list[0].strategy.outlays/pdf.loc[:,security_names]
    res.backtest_list[0].positions.diff(1)
    res.backtest_list[0].positions




.. raw:: html

    <div class="pynb-result">
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>foo</th>
          <th>bar</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2017-01-01</th>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>2017-01-02</th>
          <td>5879.285683</td>
          <td>3998.068018</td>
        </tr>
        <tr>
          <th>2017-01-03</th>
          <td>5879.285683</td>
          <td>3998.068018</td>
        </tr>
        <tr>
          <th>2017-01-04</th>
          <td>5879.285683</td>
          <td>3998.068018</td>
        </tr>
        <tr>
          <th>2017-01-05</th>
          <td>5879.285683</td>
          <td>3998.068018</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2017-12-25</th>
          <td>5324.589093</td>
          <td>4673.239436</td>
        </tr>
        <tr>
          <th>2017-12-26</th>
          <td>5324.589093</td>
          <td>4673.239436</td>
        </tr>
        <tr>
          <th>2017-12-27</th>
          <td>5324.589093</td>
          <td>4673.239436</td>
        </tr>
        <tr>
          <th>2017-12-28</th>
          <td>5324.589093</td>
          <td>4673.239436</td>
        </tr>
        <tr>
          <th>2017-12-29</th>
          <td>5324.589093</td>
          <td>4673.239436</td>
        </tr>
      </tbody>
    </table>
    <p>261 rows × 2 columns</p>
    </div>



