"""
Performance benchmarks
"""
import numpy as np
import pandas as pd
import bt
import cProfile


def benchmark_1():
    x = np.random.randn(10000, 1000) * 0.01
    idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())

    s = bt.Strategy(
        "s",
        [
            bt.algos.RunMonthly(),
            bt.algos.SelectRandomly(len(data.columns) / 2),
            bt.algos.WeighRandomly(),
            bt.algos.Rebalance(),
        ],
    )

    t = bt.Backtest(s, data)
    return bt.run(t)


def benchmark_2():
    x = np.random.randn(10000, 1000) * 0.01
    idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())
    bidoffer = data * 0.01
    coupons = data * 0.0
    s = bt.FixedIncomeStrategy(
        "s",
        algos=[
            bt.algos.RunMonthly(),
            bt.algos.SelectRandomly(len(data.columns) / 2),
            bt.algos.WeighRandomly(),
            bt.algos.Rebalance(),
        ],
        children=[bt.CouponPayingSecurity(c) for c in data],
    )

    t = bt.Backtest(s, data, additional_data={"bidoffer": bidoffer, "coupons": coupons})
    return bt.run(t)


def benchmark_3():
    # Similar to benchmark_1, but with trading in only a small subset of assets
    # However, because the "multipier" is used, we can't just pass the string
    # names to the constructor, and so the solution is to use the lazy_add flag.
    # Changing lazy_add to False demonstrates the performance gain.
    # i.e. on Win32, it went from 4.3s with the flag to 10.9s without.

    x = np.random.randn(10000, 1000) * 0.01
    idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())
    children = [bt.Security(name=i, multiplier=10, lazy_add=False) for i in range(1000)]
    s = bt.Strategy(
        "s",
        [
            bt.algos.RunMonthly(),
            bt.algos.SelectThese([0, 1]),
            bt.algos.WeighRandomly(),
            bt.algos.Rebalance(),
        ],
        children=children,
    )

    t = bt.Backtest(s, data)
    return bt.run(t)


if __name__ == "__main__":
    print("\n\n\n================= Benchmark 1 =======================\n")
    cProfile.run("benchmark_1()", sort="tottime")
    print("\n----------------- Benchmark 1 -----------------------\n\n\n")

    print("\n\n\n================= Benchmark 2 =======================\n")
    cProfile.run("benchmark_2()", sort="tottime")
    print("\n----------------- Benchmark 2 -----------------------\n\n\n")

    print("\n\n\n================= Benchmark 3 =======================\n")
    cProfile.run("benchmark_3()", sort="cumtime")
    print("\n----------------- Benchmark 3 -----------------------\n\n\n")
