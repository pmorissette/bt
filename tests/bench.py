"""
Performance benchmarks
"""
import numpy as np
import pandas as pd
import bt
import cProfile


def benchmark_1():
    x = np.random.randn(10000, 1000) * 0.01
    idx = pd.date_range('1990-01-01', freq='B', periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())

    s = bt.Strategy('s', [bt.algos.RunMonthly(),
                          bt.algos.SelectRandomly(len(data.columns) / 2),
                          bt.algos.WeighRandomly(),
                          bt.algos.Rebalance()])

    t = bt.Backtest(s, data)
    return bt.run(t)
    
def benchmark_2():
    x = np.random.randn(10000, 1000) * 0.01
    idx = pd.date_range('1990-01-01', freq='B', periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())
    bidoffer = data * 0.01
    coupons = data * 0.
    s = bt.FixedIncomeStrategy('s', 
                            algos = [bt.algos.RunMonthly(),
                              bt.algos.SelectRandomly(len(data.columns) / 2),
                              bt.algos.WeighRandomly(),
                              bt.algos.Rebalance()],
                              children = [ bt.CouponPayingSecurity(c) 
                                            for c in data ]
                              )

    t = bt.Backtest(s, data, additional_data = {'bidoffer':bidoffer, 
                                                'coupons':coupons})
    return bt.run(t)


if __name__ == '__main__':
    print('\n\n\n================= Benchmark 1 =======================\n')
    cProfile.run('benchmark_1()', sort='tottime')
    print('\n----------------- Benchmark 1 -----------------------\n\n\n')
    
    print('\n\n\n================= Benchmark 2 =======================\n')
    cProfile.run('benchmark_2()', sort='tottime')
    print('\n----------------- Benchmark 2 -----------------------\n\n\n')
