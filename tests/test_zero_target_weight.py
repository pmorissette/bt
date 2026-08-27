import pandas as pd
import pytest

import bt


@pytest.mark.parametrize("current_weight", [0.0, 0.2, -0.2])
def test_run_if_out_of_bounds_zero_target(current_weight):
    strategy = bt.Strategy("test")
    dates = pd.date_range("2024-01-01", periods=2)
    strategy.setup(pd.DataFrame({"asset": [100.0, 100.0]}, index=dates))
    strategy.update(dates[0])
    strategy.children["asset"] = bt.core.SecurityBase("asset")
    strategy.children["asset"]._weight = current_weight
    strategy.temp["weights"] = {"asset": 0.0}

    result = bt.algos.RunIfOutOfBounds(0.5)(strategy)

    assert result == (current_weight != 0.0)
    assert strategy.temp["weights"] == {"asset": 0.0}


def test_zero_target_does_not_skip_other_children():
    strategy = bt.Strategy("test")
    dates = pd.date_range("2024-01-01", periods=2)
    strategy.setup(pd.DataFrame({"zero": [100.0, 100.0], "other": [100.0, 100.0]}, index=dates))
    strategy.update(dates[0])
    for name, weight in (("zero", 0.0), ("other", 0.9)):
        strategy.children[name] = bt.core.SecurityBase(name)
        strategy.children[name]._weight = weight
    strategy.temp["weights"] = {"zero": 0.0, "other": 0.5}

    assert bt.algos.RunIfOutOfBounds(0.5)(strategy)
