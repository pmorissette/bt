"""
Performance benchmarks
"""
import numpy as np
import pandas as pd
import bt
import cProfile


def benchmark_1():
    x = np.random.randn(10000, 100) * 0.01
    idx = pd.date_range('1990-01-01', freq='B', periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())

    s = bt.Strategy('s', [bt.algos.RunMonthly(),
                          bt.algos.SelectAll(),
                          bt.algos.SelectRandomly(len(data.columns) / 2),
                          bt.algos.WeighRandomly(),
                          bt.algos.Rebalance()])

    t = bt.Backtest(s, data)
    return bt.run(t)


if __name__ == '__main__':
    print('\n\n\n================= Benchmark 1 =======================\n')
    cProfile.run('benchmark_1()', sort='tottime')
    print('\n----------------- Benchmark 1 -----------------------\n\n\n')
