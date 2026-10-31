import pandas as pd
import pytest

import bt
from bt import algos


@pytest.mark.parametrize("selected", [[], ["bond"], ["bond", "bond"]])
def test_resolve_on_the_run_without_aliases_on_unmapped_date(selected):
    dates = pd.date_range("2024-01-01", periods=2)
    prices = pd.DataFrame({"bond": [100.0, 101.0]}, index=dates)
    aliases = pd.DataFrame({"otr": ["bond"]}, index=dates[:1])
    strategy = bt.Strategy("strategy")
    strategy.setup(prices, on_the_run=aliases)
    strategy.update(dates[1])
    strategy.temp["selected"] = selected.copy()

    assert algos.ResolveOnTheRun("on_the_run")(strategy)
    assert strategy.temp["selected"] == selected


def test_resolve_on_the_run_requires_mapping_for_selected_alias():
    dates = pd.date_range("2024-01-01", periods=2)
    prices = pd.DataFrame({"bond": [100.0, 101.0]}, index=dates)
    aliases = pd.DataFrame({"otr": ["bond"]}, index=dates[:1])
    strategy = bt.Strategy("strategy")
    strategy.setup(prices, on_the_run=aliases)
    strategy.update(dates[1])
    strategy.temp["selected"] = ["otr"]

    with pytest.raises(KeyError):
        algos.ResolveOnTheRun("on_the_run")(strategy)
