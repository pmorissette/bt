FXMacroData Forex Prices
------------------------

The ``examples/fxmacrodata.py`` script shows how to fetch daily FX spot history
from FXMacroData and adapt it into the price ``DataFrame`` shape expected by
``bt.Backtest``. The executable example uses static rows, so it can run without
network access or an API key.

For live data, call ``fetch_fxmacrodata_prices`` with a currency pair such as
``"EURUSD"`` and a date range. The helper reads an optional API key from
``FXMACRODATA_API_KEY`` or ``FXMD_API_KEY``.

.. literalinclude:: ../../examples/fxmacrodata.py
   :language: python
