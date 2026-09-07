Nonlinear Cost Models: Almgren-Chriss vs. Default
-------------------------------------------------

By default :class:`bt.Backtest <bt.backtest.Backtest>` charges no transaction
cost, and even when the legacy ``commissions=fn(q, p)`` callable is supplied
the cost depends only on quantity and price. Real trades are more expensive
than that: large orders move the market against the trader, and that move
scales with order size relative to the available liquidity, and with the
instrument's volatility.

To support nonlinear, size-aware costs, ``Backtest`` accepts a
:class:`CostModel <bt.core.CostModel>` instance in place of the callable. Two
implementations ship with bt:

* :class:`SqrtCostModel <bt.core.SqrtCostModel>` — the empirical square-root
  law of impact (Tóth et al. 2011).
* :class:`AlmgrenChrissCostModel <bt.core.AlmgrenChrissCostModel>` — the
  three-component Almgren & Chriss (2001) decomposition.

When a ``CostModel`` is passed as ``commissions``, ``Backtest`` also expects
``volume`` and ``volatility`` DataFrames aligned with ``data``. Existing usage
with ``commissions=None`` or ``commissions=fn(q, p)`` is unaffected. This
example uses the AC model on a $1 billion notional book to make impact costs
visible. The design of the cost-model interface follows Abbade & Costa (2026)
[#AbbadeCosta2026]_.

The cost decomposition (Almgren-Chriss):

.. math::

    \text{cost} = \underbrace{\tfrac{1}{2} \alpha\, \sigma\, \tfrac{|q|}{V}\, |q|\, P}_{\text{permanent (triangular)}}
                + \underbrace{\epsilon\, |q|\, P}_{\text{linear: spread / fees}}
                + \underbrace{\beta\, \sigma\, \tfrac{|q|}{V}\, |q|\, P}_{\text{depth depletion}}

with defaults ``alpha=1.0``, ``beta=1.0``, ``epsilon=0.0005`` (5 bps
half-spread, typical for liquid US large-caps).

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import bt
    from bt.core import AlmgrenChrissCostModel

    %matplotlib inline
    np.random.seed(42)

Synthetic Universe
~~~~~~~~~~~~~~~~~~

Thirty securities, five years of business-day prices on a geometric random
walk. We also fabricate per-security daily volume (~30M shares average,
typical for a large-cap universe) and a 20-day rolling realised volatility
series — these are the inputs the cost model needs that a vanilla
``Backtest`` ignores.

.. code:: ipython3

    n_securities = 30
    dates = pd.date_range('2018-01-01', '2022-12-31', freq='B')
    n = len(dates)
    tickers = [f'S{i:02d}' for i in range(n_securities)]

    mu = np.random.uniform(0.04, 0.12, n_securities) / 252
    sig = np.random.uniform(0.15, 0.45, n_securities) / np.sqrt(252)
    returns = np.random.randn(n, n_securities) * sig + mu
    prices = pd.DataFrame(100 * np.exp(returns.cumsum(axis=0)),
                          index=dates, columns=tickers)

    log_ret = np.log(prices / prices.shift()).fillna(0.0)
    volatility = log_ret.rolling(20, min_periods=1).std().fillna(0.02)
    volume = pd.DataFrame(
        np.random.lognormal(mean=np.log(30_000_000), sigma=0.3,
                            size=(n, n_securities)),
        index=dates, columns=tickers,
    )

Strategy
~~~~~~~~

Monthly rebalance into the top decile by trailing 6-month return — a
classic high-turnover demo that gives the cost model real trades to score.

.. code:: ipython3

    def momentum_strategy(name):
        return bt.Strategy(
            name,
            [
                bt.algos.RunMonthly(),
                bt.algos.SelectAll(),
                bt.algos.SelectMomentum(n=10, lookback=pd.DateOffset(months=6)),
                bt.algos.WeighEqually(),
                bt.algos.Rebalance(),
            ],
        )

Run Both Backtests
~~~~~~~~~~~~~~~~~~

``baseline`` uses ``bt.Backtest`` with no commissions, the default
zero-cost assumption. ``realistic`` uses the same ``bt.Backtest`` but with a
default Almgren-Chriss cost model passed as ``commissions``, plus the
volume and volatility frames the model needs.

.. code:: ipython3

    INITIAL_CAPITAL = 1_000_000_000.0  # 1 billion USD

    baseline = bt.Backtest(
        momentum_strategy('baseline (no cost)'),
        prices,
        initial_capital=INITIAL_CAPITAL,
        integer_positions=False,
        progress_bar=False,
    )

    realistic = bt.Backtest(
        momentum_strategy('realistic (AC)'),
        prices,
        commissions=AlmgrenChrissCostModel(alpha=1.0, beta=1.0, epsilon=0.0005),
        volume=volume,
        volatility=volatility,
        initial_capital=INITIAL_CAPITAL,
        integer_positions=False,
        progress_bar=False,
    )

    result = bt.run(baseline, realistic)
    result.display()

.. parsed-literal::

    Stat                 baseline (no cost)    realistic (AC)
    -------------------  --------------------  ----------------
    Start                2017-12-31            2017-12-31
    End                  2022-12-30            2022-12-30
    Risk-free rate       0.00%                 0.00%

    Total Return         26.53%                21.59%
    Daily Sharpe         0.51                  0.43
    Daily Sortino        0.76                  ...

Equity Curves
~~~~~~~~~~~~~

The drag from realistic frictions on a $1B book at this turnover is
visually obvious — the realistic curve sits below the cost-free baseline
by roughly 5 percentage points of total return over five years.

.. code:: ipython3

    fig, ax = plt.subplots(figsize=(11, 5))
    result.prices.plot(ax=ax, lw=1.5)
    ax.set_title('Equity curves: cost-free vs Almgren-Chriss frictions ($1B book)')
    ax.set_ylabel('Index level (start = 100)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

Cost Attribution
~~~~~~~~~~~~~~~~

Inspecting fees paid against gross traded notional gives an effective bps
per dollar traded — a useful sanity check that the calibration matches
realistic expectations for a large-cap book.

.. code:: ipython3

    def summarise(bkt, label):
        fees_total = float(bkt.strategy.fees.sum())
        final_value = float(bkt.strategy.values.iloc[-1])
        traded_notional = sum(abs(sec.outlays).sum()
                              for sec in bkt.strategy.securities)
        return {
            'strategy'             : label,
            'final equity ($M)'    : final_value / 1e6,
            'total return (%)'     : 100 * (final_value / INITIAL_CAPITAL - 1),
            'gross traded ($B)'    : traded_notional / 1e9,
            'total fees ($M)'      : fees_total / 1e6,
            'fees / traded (bps)'  : 1e4 * fees_total / max(traded_notional, 1.0),
        }

    pd.DataFrame([summarise(baseline, 'baseline'),
                  summarise(realistic, 'realistic')]).set_index('strategy')

.. parsed-literal::

               final equity ($M)  total return (%)  gross traded ($B)  total fees ($M)  fees / traded (bps)
    strategy
    baseline         1265.262743         26.526274          35.432102         0.000000             0.000000
    realistic        1215.902289         21.590229          34.717752        46.394005            13.363194

The AC cost works out to ~13 bps per dollar traded — in line with
realistic large-cap impact at a few percent of average daily volume.

Sensitivity to Participation Rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The permanent and depth components scale linearly in ``|q|/V``. We can
see this directly by sweeping the available volume — a thinner market
charges more for the same trades.

.. code:: ipython3

    def run_with_volume_scale(scale):
        bkt = bt.Backtest(
            momentum_strategy(f'V x {scale}'),
            prices,
            commissions=AlmgrenChrissCostModel(alpha=1.0, beta=1.0, epsilon=0.0005),
            volume=volume * scale,
            volatility=volatility,
            initial_capital=INITIAL_CAPITAL,
            integer_positions=False,
            progress_bar=False,
        )
        bt.run(bkt)
        return float(bkt.strategy.fees.sum())

    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    fees = [run_with_volume_scale(s) for s in scales]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(scales, np.array(fees) / 1e6, marker='o')
    ax.set_xscale('log')
    ax.set_xlabel('Volume scale (1.0 = baseline)')
    ax.set_ylabel('Total fees ($M)')
    ax.set_title('Fees vs. available volume (lower V = higher participation)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

Takeaway
~~~~~~~~

Picking the cost model is a first-order modeling choice for any strategy
that trades non-trivial notional relative to available liquidity. A flat
commissions function flatters such strategies by charging a fixed bps
regardless of order size or market depth. Passing a ``CostModel`` to
``Backtest`` removes that bias by routing trade quantities through a cost
model that is sensitive to size, volume, and volatility, while leaving
existing callable-based usage of the ``commissions`` argument unchanged.

For a flat-bps degenerate behaviour, the AC model with
``alpha=0, beta=0, epsilon=bps/1e4`` reproduces the legacy commissions
path bit-for-bit.

References
~~~~~~~~~~

.. [#AbbadeCosta2026] Abbade, L. R., & Costa, A. H. R. (2026). *Realistic
   Market Impact Modeling for Reinforcement Learning Trading
   Environments.* arXiv:2603.29086. https://arxiv.org/abs/2603.29086

* Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio
  transactions. *Journal of Risk* 3(2), 5-39.
* Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren,
  J., & Bouchaud, J.-P. (2011). Anomalous Price Impact and the Critical
  Nature of Liquidity in Financial Markets. *Physical Review X* 1(2),
  021006.
* Gatheral, J. (2010). No-Dynamic-Arbitrage and Market Impact.
  *Quantitative Finance* 10(7), 749-759.
