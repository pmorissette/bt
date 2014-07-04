
.. code:: python

    import pandas as pd
    import numpy as np
    import ffn
    #%pylab inline
.. code:: python

    print 'this is a printed line'

.. parsed-literal::
    :class: pynb-result

    this is a printed line


.. code:: python

    data = ffn.get('aapl,msft,yhoo', start='2010-01-01')
    print data.head()

.. parsed-literal::
    :class: pynb-result

                 aapl   msft   yhoo
    Date                           
    2010-01-04  29.22  27.48  17.10
    2010-01-05  29.27  27.49  17.23
    2010-01-06  28.81  27.32  17.17
    2010-01-07  28.75  27.03  16.70
    2010-01-08  28.94  27.22  16.70
    
    [5 rows x 3 columns]


.. code:: python

    data.head()



.. raw:: html

    <div class="pynb-result" style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>aapl</th>
          <th>msft</th>
          <th>yhoo</th>
        </tr>
        <tr>
          <th>Date</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2010-01-04</th>
          <td> 29.22</td>
          <td> 27.48</td>
          <td> 17.10</td>
        </tr>
        <tr>
          <th>2010-01-05</th>
          <td> 29.27</td>
          <td> 27.49</td>
          <td> 17.23</td>
        </tr>
        <tr>
          <th>2010-01-06</th>
          <td> 28.81</td>
          <td> 27.32</td>
          <td> 17.17</td>
        </tr>
        <tr>
          <th>2010-01-07</th>
          <td> 28.75</td>
          <td> 27.03</td>
          <td> 16.70</td>
        </tr>
        <tr>
          <th>2010-01-08</th>
          <td> 28.94</td>
          <td> 27.22</td>
          <td> 16.70</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 3 columns</p>
    </div>



.. code:: python

    data.plot()



.. parsed-literal::
    :class: pynb-result

    <matplotlib.axes.AxesSubplot at 0x7fbae88b19d0>




.. image:: _static/nb-examples_4_1.png
    :class: pynb


.. code:: python

    # this is a comment
    data.to_returns().dropna().corr().as_format('.2f')



.. raw:: html

    <div class="pynb-result" style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>aapl</th>
          <th>msft</th>
          <th>yhoo</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>aapl</th>
          <td> 1.00</td>
          <td> 0.35</td>
          <td> 0.28</td>
        </tr>
        <tr>
          <th>msft</th>
          <td> 0.35</td>
          <td> 1.00</td>
          <td> 0.37</td>
        </tr>
        <tr>
          <th>yhoo</th>
          <td> 0.28</td>
          <td> 0.37</td>
          <td> 1.00</td>
        </tr>
      </tbody>
    </table>
    <p>3 rows × 3 columns</p>
    </div>


