# bt — Flexible Backtesting for Python

[![Build Status](https://github.com/pmorissette/bt/workflows/Build%20Status/badge.svg)](https://github.com/pmorissette/bt/actions/)
[![PyPI Version](https://img.shields.io/pypi/v/bt)](https://pypi.org/project/bt/)
[![PyPI License](https://img.shields.io/pypi/l/bt)](https://pypi.org/project/bt/)

<a id="what-is-bt"></a>

Build, test, and compare investment strategies from reusable Python components.
bt combines strategy logic with historical price data, tracks portfolio positions
and transactions, and provides performance statistics and charts through
[ffn](https://github.com/pmorissette/ffn).

<a id="features"></a>

- **Compose strategy logic:** combine algorithms for scheduling, security selection, weighting, and rebalancing.
- **Build portfolios of strategies:** nest strategies and securities in a common tree.
- **Model trading costs:** configure commissions and transaction cost models.
- **Compare results:** inspect returns, weights, transactions, drawdowns, and other statistics.

## Install

```bash
pip install bt
```

See the [installation guide](docs/source/install.md) for additional details.

<a id="a-quick-example"></a>
<a id="a-simple-strategy-backtest"></a>

## A first backtest

This example uses synthetic prices, so it runs without downloading market data:

```python
import numpy as np
import pandas as pd

import bt

prices = pd.DataFrame(
    {
        "asset_a": np.linspace(100, 120, 252),
        "asset_b": np.linspace(100, 110, 252),
    },
    index=pd.bdate_range("2020-01-01", periods=252),
)

strategy = bt.Strategy(
    "equal_weight",
    [
        bt.algos.RunMonthly(),
        bt.algos.SelectAll(),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance(),
    ],
)
result = bt.run(bt.Backtest(strategy, prices))
result.display()
```

The strategy selects both assets, gives each equal weight, and rebalances monthly.
Replace the synthetic prices with your own data to explore a strategy. Backtest
results depend on data quality and modeling assumptions; they do not predict
future performance.

<a id="modifying-a-strategy"></a>

## Explore the documentation

- [First strategy tutorial](docs/source/intro.md): walk through a backtest and inspect its results.
- [Algorithms](docs/source/algos.md): compose and customize strategy logic.
- [Portfolio trees](docs/source/tree.md): combine securities and nested strategies.
- [Examples](docs/source/examples.md): explore momentum, risk allocation, and fixed-income strategies.
- [API overview](docs/source/overview.md): find strategy, algorithm, and backtest interfaces.

The published documentation is at <https://pmorissette.github.io/bt/>.

<a id="roadmap"></a>

## Contribute

See the [development guide](docs/development.md) for environment setup, tests,
documentation builds, and Copier template updates. Report bugs and propose
improvements through [GitHub issues](https://github.com/pmorissette/bt/issues).

bt is released under the [MIT license](LICENSE).
