Trend Example 2
---------------

.. code:: ipython3

    import numpy as np
    import pandas as pd
    
    import bt
    import matplotlib.pyplot as plt
    
    %matplotlib inline

.. code:: ipython3

    np.random.seed(0)
    returns =  np.random.normal(0.08/12,0.2/np.sqrt(12),12*10)
    pdf = pd.DataFrame(
        np.cumprod(1+returns),
        index = pd.date_range(start="2008-01-01",periods=12*10,freq="m"),
        columns=['foo']
    )
    
    pdf.plot();



.. image:: _static/Trend_2_2_0.png
   :class: pynb
   :width: 373px
   :height: 251px


.. code:: ipython3

    runMonthlyAlgo = bt.algos.RunMonthly()
    rebalAlgo = bt.algos.Rebalance()
    
    class Signal(bt.Algo):
    
        """
        
        Mostly copied from StatTotalReturn
        
        Sets temp['Signal'] with total returns over a given period.
    
        Sets the 'Signal' based on the total return of each
        over a given lookback period.
    
        Args:
            * lookback (DateOffset): lookback period.
            * lag (DateOffset): Lag interval. Total return is calculated in
                the inteval [now - lookback - lag, now - lag]
    
        Sets:
            * stat
    
        Requires:
            * selected
    
        """
    
        def __init__(self, lookback=pd.DateOffset(months=3),
                     lag=pd.DateOffset(days=0)):
            super(Signal, self).__init__()
            self.lookback = lookback
            self.lag = lag
    
        def __call__(self, target):
            selected = 'foo'
            t0 = target.now - self.lag
            
            if target.universe[selected].index[0] > t0:
                return False
            prc = target.universe[selected].loc[t0 - self.lookback:t0]
            
            
            trend = prc.iloc[-1]/prc.iloc[0] - 1
            signal = trend > 0.
            
            if signal:
                target.temp['Signal'] = 1.
            else:
                target.temp['Signal'] = 0.
                
            return True
    
    signalAlgo = Signal(pd.DateOffset(months=12),pd.DateOffset(months=1))
        
    class WeighFromSignal(bt.Algo):
    
        """
        Sets temp['weights'] from the signal.
        Sets:
            * weights
    
        Requires:
            * selected
    
        """
    
        def __init__(self):
            super(WeighFromSignal, self).__init__()
    
        def __call__(self, target):
            selected = 'foo'
            if target.temp['Signal'] is None:
                raise(Exception('No Signal!'))
            
            target.temp['weights'] = {selected : target.temp['Signal']}
            return True
        
    weighFromSignalAlgo = WeighFromSignal()

.. code:: ipython3

    s = bt.Strategy(
        'example1',
        [
            runMonthlyAlgo,
            signalAlgo,
            weighFromSignalAlgo,
            rebalAlgo
        ]
    )
    
    t = bt.Backtest(s, pdf, integer_positions=False, progress_bar=True)
    res = bt.run(t)



.. parsed-literal::
   :class: pynb-result

    example1
    0% [############################# ] 100% | ETA: 00:00:00

.. code:: ipython3

    res.plot_security_weights();



.. image:: _static/Trend_2_5_0.png
   :class: pynb
   :width: 876px
   :height: 289px


.. code:: ipython3

    t.positions




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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2008-01-30</th>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>2008-01-31</th>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>2008-02-29</th>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>2008-03-31</th>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>2008-04-30</th>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
        </tr>
        <tr>
          <th>2017-08-31</th>
          <td>631321.251898</td>
        </tr>
        <tr>
          <th>2017-09-30</th>
          <td>631321.251898</td>
        </tr>
        <tr>
          <th>2017-10-31</th>
          <td>631321.251898</td>
        </tr>
        <tr>
          <th>2017-11-30</th>
          <td>631321.251898</td>
        </tr>
        <tr>
          <th>2017-12-31</th>
          <td>631321.251898</td>
        </tr>
      </tbody>
    </table>
    <p>121 rows × 1 columns</p>
    </div>



.. code:: ipython3

    res.prices.tail()




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
          <th>example1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2017-08-31</th>
          <td>240.302579</td>
        </tr>
        <tr>
          <th>2017-09-30</th>
          <td>255.046653</td>
        </tr>
        <tr>
          <th>2017-10-31</th>
          <td>254.464421</td>
        </tr>
        <tr>
          <th>2017-11-30</th>
          <td>265.182603</td>
        </tr>
        <tr>
          <th>2017-12-31</th>
          <td>281.069771</td>
        </tr>
      </tbody>
    </table>
    </div>



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
          <th>example1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>start</th>
          <td>2008-01-30 00:00:00</td>
        </tr>
        <tr>
          <th>end</th>
          <td>2017-12-31 00:00:00</td>
        </tr>
        <tr>
          <th>rf</th>
          <td>0.0</td>
        </tr>
        <tr>
          <th>total_return</th>
          <td>1.810698</td>
        </tr>
        <tr>
          <th>cagr</th>
          <td>0.109805</td>
        </tr>
        <tr>
          <th>max_drawdown</th>
          <td>-0.267046</td>
        </tr>
        <tr>
          <th>calmar</th>
          <td>0.411186</td>
        </tr>
        <tr>
          <th>mtd</th>
          <td>0.05991</td>
        </tr>
        <tr>
          <th>three_month</th>
          <td>0.102033</td>
        </tr>
        <tr>
          <th>six_month</th>
          <td>0.22079</td>
        </tr>
        <tr>
          <th>ytd</th>
          <td>0.879847</td>
        </tr>
        <tr>
          <th>one_year</th>
          <td>0.879847</td>
        </tr>
        <tr>
          <th>three_year</th>
          <td>0.406395</td>
        </tr>
        <tr>
          <th>five_year</th>
          <td>0.227148</td>
        </tr>
        <tr>
          <th>ten_year</th>
          <td>0.109805</td>
        </tr>
        <tr>
          <th>incep</th>
          <td>0.109805</td>
        </tr>
        <tr>
          <th>daily_sharpe</th>
          <td>3.299555</td>
        </tr>
        <tr>
          <th>daily_sortino</th>
          <td>6.352869</td>
        </tr>
        <tr>
          <th>daily_mean</th>
          <td>2.448589</td>
        </tr>
        <tr>
          <th>daily_vol</th>
          <td>0.742097</td>
        </tr>
        <tr>
          <th>daily_skew</th>
          <td>0.307861</td>
        </tr>
        <tr>
          <th>daily_kurt</th>
          <td>1.414455</td>
        </tr>
        <tr>
          <th>best_day</th>
          <td>0.137711</td>
        </tr>
        <tr>
          <th>worst_day</th>
          <td>-0.14073</td>
        </tr>
        <tr>
          <th>monthly_sharpe</th>
          <td>0.723148</td>
        </tr>
        <tr>
          <th>monthly_sortino</th>
          <td>1.392893</td>
        </tr>
        <tr>
          <th>monthly_mean</th>
          <td>0.117579</td>
        </tr>
        <tr>
          <th>monthly_vol</th>
          <td>0.162594</td>
        </tr>
        <tr>
          <th>monthly_skew</th>
          <td>0.301545</td>
        </tr>
        <tr>
          <th>monthly_kurt</th>
          <td>1.379006</td>
        </tr>
        <tr>
          <th>best_month</th>
          <td>0.137711</td>
        </tr>
        <tr>
          <th>worst_month</th>
          <td>-0.14073</td>
        </tr>
        <tr>
          <th>yearly_sharpe</th>
          <td>0.503939</td>
        </tr>
        <tr>
          <th>yearly_sortino</th>
          <td>5.019272</td>
        </tr>
        <tr>
          <th>yearly_mean</th>
          <td>0.14814</td>
        </tr>
        <tr>
          <th>yearly_vol</th>
          <td>0.293964</td>
        </tr>
        <tr>
          <th>yearly_skew</th>
          <td>2.317496</td>
        </tr>
        <tr>
          <th>yearly_kurt</th>
          <td>5.894955</td>
        </tr>
        <tr>
          <th>best_year</th>
          <td>0.879847</td>
        </tr>
        <tr>
          <th>worst_year</th>
          <td>-0.088543</td>
        </tr>
        <tr>
          <th>avg_drawdown</th>
          <td>-0.091255</td>
        </tr>
        <tr>
          <th>avg_drawdown_days</th>
          <td>369.714286</td>
        </tr>
        <tr>
          <th>avg_up_month</th>
          <td>0.064341</td>
        </tr>
        <tr>
          <th>avg_down_month</th>
          <td>-0.012928</td>
        </tr>
        <tr>
          <th>win_year_perc</th>
          <td>0.555556</td>
        </tr>
        <tr>
          <th>twelve_month_win_perc</th>
          <td>0.46789</td>
        </tr>
      </tbody>
    </table>
    </div>



