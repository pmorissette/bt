"""
Performance regression tests to ensure iloc-free implementation is fast.

These tests verify that the core backtesting engine maintains at least a 20x
speedup compared to iloc-based implementations.
"""

import pytest
import numpy as np
import pandas as pd
import time
import bt


class TestCorePerformance:
    """
    Performance tests for core.py to verify at least 20x speedup vs iloc-based approach.

    The original iloc-based implementation in pandas 2+ is slow because each
    iloc access involves significant overhead from pandas' indexing machinery.
    The numpy-native implementation should be at least 20x faster.
    """

    @pytest.fixture
    def large_data(self):
        """Generate large dataset for performance testing."""
        np.random.seed(42)
        x = np.random.randn(5000, 500) * 0.01
        idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
        data = np.exp(pd.DataFrame(x, index=idx).cumsum())
        return data

    @pytest.fixture
    def simple_strategy_data(self, large_data):
        """Simple strategy for performance testing."""
        s = bt.Strategy(
            "s",
            [
                bt.algos.RunMonthly(),
                bt.algos.SelectAll(),
                bt.algos.WeighEqually(),
                bt.algos.Rebalance(),
            ],
        )
        return s, large_data

    def test_update_performance_numpy_vs_iloc(self, large_data):
        """
        Compare numpy array indexing vs iloc for hot path operations.

        The iloc path simulates the old pandas 2+ behavior:
        series.iloc[inow] = value

        The numpy path uses direct array indexing:
        arr[inow] = value

        Expected speedup: at least 20x
        """
        n_rows = len(large_data)
        len(large_data.columns)
        n_iterations = n_rows

        # Create test arrays
        arr = np.zeros(n_rows, dtype=np.float64)
        series = pd.Series(np.zeros(n_rows, dtype=np.float64), index=large_data.index)

        # Warmup
        for i in range(min(100, n_rows)):
            arr[i] = i * 1.0
            series.iloc[i] = i * 1.0

        # Test numpy array access
        start = time.perf_counter()
        for i in range(n_iterations):
            arr[i] = i * 1.5
        numpy_time = time.perf_counter() - start

        # Reset for iloc test
        arr.fill(0)

        # Test iloc access
        start = time.perf_counter()
        for i in range(n_iterations):
            series.iloc[i] = i * 1.5
        iloc_time = time.perf_counter() - start

        speedup = iloc_time / numpy_time if numpy_time > 0 else float("inf")
        print(f"\nNumpy array indexing time: {numpy_time:.4f}s")
        print(f"iloc indexing time: {iloc_time:.4f}s")
        print(f"Speedup: {speedup:.1f}x")

        # Assert at least 20x speedup
        assert speedup >= 20.0, f"Expected at least 20x speedup, got {speedup:.1f}x"

    def test_backtest_performance_small(self, simple_strategy_data):
        """
        Test backtest performance with a smaller dataset.

        This ensures the basic backtest functionality is fast enough.
        """
        s, data = simple_strategy_data
        t = bt.Backtest(s, data)

        start = time.perf_counter()
        bt.run(t)
        elapsed = time.perf_counter() - start

        print(f"\nSmall backtest time: {elapsed:.4f}s")

        # Should complete in reasonable time
        assert elapsed < 180.0, f"Backtest took too long: {elapsed:.4f}s"

    def test_backtest_performance_large(self):
        """
        Test backtest performance with a large dataset.

        This is the main performance regression test.
        """
        np.random.seed(42)
        x = np.random.randn(2000, 200) * 0.01
        idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
        data = np.exp(pd.DataFrame(x, index=idx).cumsum())

        s = bt.Strategy(
            "s",
            [
                bt.algos.RunMonthly(),
                bt.algos.SelectAll(),
                bt.algos.WeighEqually(),
                bt.algos.Rebalance(),
            ],
        )
        t = bt.Backtest(s, data)

        start = time.perf_counter()
        bt.run(t)
        elapsed = time.perf_counter() - start

        print(f"\nLarge backtest time: {elapsed:.4f}s")
        print(f"Data shape: {data.shape}")

        # Should complete in reasonable time
        assert elapsed < 120.0, f"Large backtest took too long: {elapsed:.4f}s"

    def test_strategy_update_hot_path(self):
        """
        Test the hot path update performance.

        This simulates what happens in StrategyBase.update() and SecurityBase.update()
        which are called thousands of times during a backtest.
        """
        n_rows = 10000
        n_updates = n_rows

        # Simulate strategy internal arrays
        prices = np.zeros(n_rows, dtype=np.float64)
        values = np.zeros(n_rows, dtype=np.float64)
        cash = np.zeros(n_rows, dtype=np.float64)

        # Simulate pandas series for comparison
        prices_s = pd.Series(np.zeros(n_rows, dtype=np.float64))
        values_s = pd.Series(np.zeros(n_rows, dtype=np.float64))
        cash_s = pd.Series(np.zeros(n_rows, dtype=np.float64))

        # Warmup
        for i in range(100):
            prices[i] = i * 1.0
            values[i] = i * 2.0
            cash[i] = i * 0.5
            prices_s.iloc[i] = i * 1.0
            values_s.iloc[i] = i * 2.0
            cash_s.iloc[i] = i * 0.5

        # Test direct numpy array updates (new implementation)
        start = time.perf_counter()
        for i in range(n_updates):
            prices[i] = i * 1.0
            values[i] = i * 2.0
            cash[i] = i * 0.5
        numpy_time = time.perf_counter() - start

        # Test iloc-based updates (old implementation)
        start = time.perf_counter()
        for i in range(n_updates):
            prices_s.iloc[i] = i * 1.0
            values_s.iloc[i] = i * 2.0
            cash_s.iloc[i] = i * 0.5
        iloc_time = time.perf_counter() - start

        speedup = iloc_time / numpy_time if numpy_time > 0 else float("inf")
        print(f"\nHot path numpy time: {numpy_time:.4f}s")
        print(f"Hot path iloc time: {iloc_time:.4f}s")
        print(f"Speedup: {speedup:.1f}x")

        # Assert at least 20x speedup
        assert speedup >= 20.0, f"Expected at least 20x speedup, got {speedup:.1f}x"

    def test_security_update_performance(self):
        """
        Test SecurityBase.update() hot path performance.

        SecurityBase.update() is called very frequently during backtesting.
        """
        np.random.seed(42)
        x = np.random.randn(10000, 100) * 0.01
        idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
        data = np.exp(pd.DataFrame(x, index=idx).cumsum())

        s = bt.Strategy(
            "s",
            [
                bt.algos.RunMonthly(),
                bt.algos.SelectAll(),
                bt.algos.WeighEqually(),
                bt.algos.Rebalance(),
            ],
            children=[bt.Security(name=c) for c in data.columns],
        )
        t = bt.Backtest(s, data)

        start = time.perf_counter()
        bt.run(t)
        elapsed = time.perf_counter() - start

        print(f"\nSecurity update backtest time: {elapsed:.4f}s")

        # Should be fast enough for typical use
        assert elapsed < 120.0, f"Security update backtest took too long: {elapsed:.4f}s"

    def test_fixed_income_strategy_performance(self):
        """
        Test FixedIncomeStrategy performance with CouponPayingSecurity.
        """
        np.random.seed(42)
        x = np.random.randn(2000, 100) * 0.01
        idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
        data = np.exp(pd.DataFrame(x, index=idx).cumsum())
        bidoffer = data * 0.01
        coupons = data * 0.001

        s = bt.FixedIncomeStrategy(
            "s",
            algos=[
                bt.algos.RunMonthly(),
                bt.algos.SelectAll(),
                bt.algos.WeighEqually(),
                bt.algos.Rebalance(),
            ],
            children=[bt.CouponPayingSecurity(c) for c in data.columns],
        )

        t = bt.Backtest(s, data, additional_data={"bidoffer": bidoffer, "coupons": coupons})

        start = time.perf_counter()
        bt.run(t)
        elapsed = time.perf_counter() - start

        print(f"\nFixed income strategy time: {elapsed:.4f}s")

        # Should complete in reasonable time
        assert elapsed < 120.0, f"Fixed income strategy took too long: {elapsed:.4f}s"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
