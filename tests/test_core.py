from __future__ import division

import copy
import numpy as np
import pandas as pd
import pytest

from unittest import mock

import bt
from bt.core import Node, StrategyBase, SecurityBase, AlgoStack, Strategy
from bt.core import FixedIncomeStrategy, HedgeSecurity, FixedIncomeSecurity
from bt.core import CouponPayingSecurity, CouponPayingHedgeSecurity
from bt.core import is_zero


def test_node_tree1():
    # Create a regular strategy
    c1 = Node("c1")
    c2 = Node("c2")
    p = Node("p", children=[c1, c2, "c3", "c4"])

    assert "c1" in p.children
    assert "c2" in p.children
    assert p["c1"] != c1
    assert p["c1"] != c2
    c1 = p["c1"]
    c2 = p["c2"]

    assert len(p.children) == 2
    assert p == c1.parent
    assert p == c2.parent
    assert p == c1.root
    assert p == c2.root

    # Create a new parent strategy with a child sub-strategy
    m = Node("m", children=[p, c1])
    p = m["p"]
    mc1 = m["c1"]
    c1 = p["c1"]
    c2 = p["c2"]

    assert len(m.children) == 2
    assert "p" in m.children
    assert "c1" in m.children
    assert mc1 != c1

    assert p.parent == m
    assert len(p.children) == 2
    assert "c1" in p.children
    assert "c2" in p.children
    assert p == c1.parent
    assert p == c2.parent
    assert m == p.root
    assert m == c1.root
    assert m == c2.root

    # Add a new node into the strategy
    c0 = Node("c0", parent=p)
    c0 = p["c0"]
    assert "c0" in p.children
    assert p == c0.parent
    assert m == c0.root
    assert len(p.children) == 3

    # Add a new sub-strategy into the parent strategy
    p2 = Node("p2", children=[c0, c1], parent=m)
    p2 = m["p2"]
    c0 = p2["c0"]
    c1 = p2["c1"]
    assert "p2" in m.children
    assert p2.parent == m
    assert len(p2.children) == 2
    assert "c0" in p2.children
    assert "c1" in p2.children
    assert c0 != p["c0"]
    assert c1 != p["c1"]
    assert p2 == c0.parent
    assert p2 == c1.parent
    assert m == p2.root
    assert m == c0.root
    assert m == c1.root


def test_node_tree2():
    # Just like test_node_tree1, but using the dictionary constructor
    c = Node("template")
    p = Node("p", children={"c1": c, "c2": c, "c3": "", "c4": ""})
    assert "c1" in p.children
    assert "c2" in p.children
    assert p["c1"] != c
    assert p["c1"] != c
    c1 = p["c1"]
    c2 = p["c2"]

    assert len(p.children) == 2
    assert c1.name == "c1"
    assert c2.name == "c2"
    assert p == c1.parent
    assert p == c2.parent
    assert p == c1.root
    assert p == c2.root


def test_node_tree3():
    c1 = Node("c1")
    c2 = Node("c1")  # Same name!
    raised = False
    try:
        p = Node("p", children=[c1, c2, "c3", "c4"])
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        p = Node("p", children=["c1", "c1"])
    except ValueError:
        raised = True
    assert raised

    c1 = Node("c1")
    c2 = Node("c2")
    p = Node("p", children=[c1, c2, "c3", "c4"])
    raised = False
    try:
        Node("c1", parent=p)
    except ValueError:
        raised = True
    assert raised

    # This does not raise, as it's just providing an implementation of 'c3',
    # which had been declared earlier
    c3 = Node("c3", parent=p)
    assert "c3" in p.children


def test_integer_positions():
    c1 = Node("c1")
    c2 = Node("c2")
    c1.integer_positions = False
    p = Node("p", children=[c1, c2])
    c1 = p["c1"]
    c2 = p["c2"]
    assert p.integer_positions
    assert c1.integer_positions
    assert c2.integer_positions

    p.use_integer_positions(False)
    assert not p.integer_positions
    assert not c1.integer_positions
    assert not c2.integer_positions

    c3 = Node("c3", parent=p)
    c3 = p["c3"]
    assert not c3.integer_positions

    p2 = Node("p2", children=[p])
    p = p2["p"]
    c1 = p["c1"]
    c2 = p["c2"]
    assert p2.integer_positions
    assert p.integer_positions
    assert c1.integer_positions
    assert c2.integer_positions


def test_strategybase_tree():
    s1 = SecurityBase("s1")
    s2 = SecurityBase("s2")
    s = StrategyBase("p", [s1, s2])

    s1 = s["s1"]
    s2 = s["s2"]

    assert len(s.children) == 2
    assert "s1" in s.children
    assert "s2" in s.children
    assert s == s1.parent
    assert s == s2.parent


def test_node_members():
    s1 = SecurityBase("s1")
    s2 = SecurityBase("s2")
    s = StrategyBase("p", [s1, s2])

    s1 = s["s1"]
    s2 = s["s2"]

    actual = s.members
    assert len(actual) == 3
    assert s1 in actual
    assert s2 in actual
    assert s in actual

    actual = s1.members
    assert len(actual) == 1
    assert s1 in actual

    actual = s2.members
    assert len(actual) == 1
    assert s2 in actual


def test_node_full_name():
    s1 = SecurityBase("s1")
    s2 = SecurityBase("s2")
    s = StrategyBase("p", [s1, s2])

    # we cannot access s1 and s2 directly since they are copied
    # we must therefore access through s
    assert s.full_name == "p"
    assert s["s1"].full_name == "p>s1"
    assert s["s2"].full_name == "p>s2"


def test_security_setup_prices():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105
    data.loc[dts[0], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    assert c1.price == 105
    assert len(c1.prices) == 1
    assert c1.prices.iloc[0] == 105

    assert c2.price == 95
    assert len(c2.prices) == 1
    assert c2.prices.iloc[0] == 95

    # now with setup
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])
    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105
    data.loc[dts[0], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    assert c1.price == 105
    assert len(c1.prices) == 1
    assert c1.prices.iloc[0] == 105

    assert c2.price == 95
    assert len(c2.prices) == 1
    assert c2.prices.iloc[0] == 95


def test_strategybase_tree_setup():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    assert len(s.data) == 3
    assert len(c1.data) == 3
    assert len(c2.data) == 3

    assert len(s._prices) == 3
    assert len(c1._prices) == 3
    assert len(c2._prices) == 3

    assert len(s._values) == 3
    assert len(c1._values) == 3
    assert len(c2._values) == 3


def test_strategybase_tree_adjust():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    s.adjust(1000)

    assert s.capital == 1000
    assert s.value == 1000
    assert c1.value == 0
    assert c2.value == 0
    assert c1.weight == 0
    assert c2.weight == 0

    s.update(dts[0])
    assert s.flows[dts[0]] == 1000


def test_strategybase_tree_update():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])
    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    assert c1.price == 100
    assert c2.price == 100

    i = 1
    s.update(dts[i], data.loc[dts[i]])

    assert c1.price == 105
    assert c2.price == 95

    i = 2
    s.update(dts[i], data.loc[dts[i]])

    assert c1.price == 100
    assert c2.price == 100


def test_update_fails_if_price_is_nan_and_position_open():
    c1 = SecurityBase("c1")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100)
    data.loc[dts[1], "c1"] = np.nan

    c1.setup(data)

    i = 0
    # mock in position
    c1._position = 100
    c1.update(dts[i], data.loc[dts[i]])

    # test normal case - position & non-nan price
    assert c1._value == 100 * 100

    i = 1
    # this should fail, because we have non-zero position, and price is nan, so
    # bt has no way of updating the _value
    try:
        c1.update(dts[i], data.loc[dts[i]])
        assert False
    except Exception as e:
        assert str(e).startswith("Position is open")

    # on the other hand, if position was 0, this should be fine, and update
    # value to 0
    c1._position = 0
    c1.update(dts[i], data.loc[dts[i]])
    assert c1._value == 0


def test_strategybase_tree_allocate():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)
    # since children have w == 0 this should stay in s
    s.allocate(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now allocate directly to child
    c1.allocate(500)

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 500
    assert s.value == 1000
    assert c1.weight == pytest.approx(500.0 / 1000)
    assert c2.weight == 0


def test_strategybase_tree_allocate_child_from_strategy():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])
    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)
    # since children have w == 0 this should stay in s
    s.allocate(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now allocate to c1
    s.allocate(500, "c1")

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 500
    assert s.value == 1000
    assert c1.weight == pytest.approx(500.0 / 1000)
    assert c2.weight == 0


def test_strategybase_tree_allocate_level2():
    c1 = SecurityBase("c1")
    c12 = copy.deepcopy(c1)
    c2 = SecurityBase("c2")
    c22 = copy.deepcopy(c2)
    s1 = StrategyBase("s1", [c1, c2])
    s2 = StrategyBase("s2", [c12, c22])
    m = StrategyBase("m", [s1, s2])

    s1 = m["s1"]
    s2 = m["s2"]

    c1 = s1["c1"]
    c2 = s1["c2"]
    c12 = s2["c1"]
    c22 = s2["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    m.setup(data)

    i = 0
    m.update(dts[i], data.loc[dts[i]])

    m.adjust(1000)
    # since children have w == 0 this should stay in s
    m.allocate(1000)

    assert m.value == 1000
    assert m.capital == 1000
    assert s1.value == 0
    assert s2.value == 0
    assert c1.value == 0
    assert c2.value == 0

    # now allocate directly to child
    s1.allocate(500)

    assert s1.value == 500
    assert m.capital == 1000 - 500
    assert m.value == 1000
    assert s1.weight == pytest.approx(500.0 / 1000)
    assert s2.weight == 0

    # now allocate directly to child of child
    c1.allocate(200)

    assert s1.value == 500
    assert s1.capital == 500 - 200
    assert c1.value == 200
    assert c1.weight == pytest.approx(200.0 / 500)
    assert c1.position == 2

    assert m.capital == 1000 - 500
    assert m.value == 1000
    assert s1.weight == pytest.approx(500.0 / 1000)
    assert s2.weight == 0

    assert c12.value == 0


def test_strategybase_tree_allocate_long_short():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)
    c1.allocate(500)

    assert c1.position == 5
    assert c1.value == 500
    assert c1.weight == pytest.approx(500.0 / 1000)
    assert s.capital == 1000 - 500
    assert s.value == 1000

    c1.allocate(-200)

    assert c1.position == 3
    assert c1.value == 300
    assert c1.weight == pytest.approx(300.0 / 1000)
    assert s.capital == 1000 - 500 + 200
    assert s.value == 1000

    c1.allocate(-400)

    assert c1.position == -1
    assert c1.value == -100
    assert c1.weight == pytest.approx(-100.0 / 1000)
    assert s.capital == 1000 - 500 + 200 + 400
    assert s.value == 1000

    # close up
    c1.allocate(-c1.value)

    assert c1.position == 0
    assert c1.value == 0
    assert c1.weight == 0
    assert s.capital == 1000 - 500 + 200 + 400 - 100
    assert s.value == 1000


def test_strategybase_tree_allocate_update():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])
    assert s.price == 100

    s.adjust(1000)

    assert s.price == 100
    assert s.value == 1000
    assert s._value == 1000

    c1.allocate(500)

    assert c1.position == 5
    assert c1.value == 500
    assert c1.weight == pytest.approx(500.0 / 1000)
    assert s.capital == 1000 - 500
    assert s.value == 1000
    assert s.price == 100

    i = 1
    s.update(dts[i], data.loc[dts[i]])

    assert c1.position == 5
    assert c1.value == 525
    assert c1.weight == pytest.approx(525.0 / 1025)
    assert s.capital == 1000 - 500
    assert s.value == 1025
    assert np.allclose(s.price, 102.5)


def test_strategybase_universe():
    s = StrategyBase("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105
    data.loc[dts[0], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i])

    assert len(s.universe) == 1
    assert "c1" in s.universe
    assert "c2" in s.universe
    assert s.universe["c1"][dts[i]] == 105
    assert s.universe["c2"][dts[i]] == 95

    # should not have children unless allocated
    assert len(s.children) == 0


def test_strategybase_allocate():
    s = StrategyBase("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 100
    data.loc[dts[0], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i])

    s.adjust(1000)
    s.allocate(100, "c1")
    c1 = s["c1"]

    assert c1.position == 1
    assert c1.value == 100
    assert s.value == 1000


def test_strategybase_lazy():
    # A mix of test_strategybase_universe and test_strategybase_allocate
    # to make sure that assets with lazy_add work correctly.
    c1 = SecurityBase(
        "c1",
        multiplier=2,
        lazy_add=True,
    )
    c2 = FixedIncomeSecurity("c2", lazy_add=True)
    s = StrategyBase("s", [c1, c2])

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105
    data.loc[dts[0], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i])

    assert len(s.universe) == 1
    assert "c1" in s.universe
    assert "c2" in s.universe
    assert s.universe["c1"][dts[i]] == 105
    assert s.universe["c2"][dts[i]] == 95

    # should not have children unless allocated
    assert len(s.children) == 0

    s.adjust(1000)
    s.allocate(100, "c1")
    s.allocate(100, "c2")
    c1 = s["c1"]
    c2 = s["c2"]
    assert c1.multiplier == 2
    assert isinstance(c2, FixedIncomeSecurity)


def test_strategybase_close():
    s = StrategyBase("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)

    i = 0
    s.update(dts[i])

    s.adjust(1000)
    s.allocate(100, "c1")
    c1 = s["c1"]

    assert c1.position == 1
    assert c1.value == 100
    assert s.value == 1000

    s.close("c1")

    assert c1.position == 0
    assert c1.value == 0
    assert s.value == 1000


def test_strategybase_flatten():
    s = StrategyBase("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)

    i = 0
    s.update(dts[i])

    s.adjust(1000)
    s.allocate(100, "c1")
    c1 = s["c1"]
    s.allocate(100, "c2")
    c2 = s["c2"]

    assert c1.position == 1
    assert c1.value == 100
    assert c2.position == 1
    assert c2.value == 100
    assert s.value == 1000

    s.flatten()

    assert c1.position == 0
    assert c1.value == 0
    assert s.value == 1000


def test_strategybase_multiple_calls():
    s = StrategyBase("s")

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    data.loc[dts[0], 'c2'] = 95
    data.loc[dts[1], 'c1'] = 95
    data.loc[dts[2], 'c2'] = 95
    data.loc[dts[3], 'c2'] = 95
    data.loc[dts[4], 'c2'] = 95
    data.loc[dts[4], 'c1'] = 105

    s.setup(data)

    # define strategy logic
    def algo(target):
        # close out any open positions
        target.flatten()

        # get stock w/ lowest price
        c = target.universe.loc[target.now].idxmin()

        # allocate all capital to that stock
        target.allocate(target.value, c)

    # replace run logic
    s.run = algo

    # start w/ 1000
    s.adjust(1000)

    # loop through dates manually
    i = 0

    # update t0
    s.update(dts[i])

    assert len(s.children) == 0
    assert s.value == 1000

    # run t0
    s.run(s)

    assert len(s.children) == 1
    assert s.value == 1000
    assert s.capital == 50

    c2 = s["c2"]
    assert c2.value == 950
    assert c2.weight == pytest.approx(950.0 / 1000)
    assert c2.price == 95

    # update out t0
    s.update(dts[i])

    c2 = s["c2"]
    assert len(s.children) == 1
    assert s.value == 1000
    assert s.capital == 50

    assert c2.value == 950
    assert c2.weight == pytest.approx(950.0 / 1000)
    assert c2.price == 95

    # update t1
    i = 1
    s.update(dts[i])

    assert s.value == 1050
    assert s.capital == 50
    assert len(s.children) == 1

    assert "c2" in s.children
    c2 = s["c2"]
    assert c2.value == 1000
    assert c2.weight == pytest.approx(1000.0 / 1050.0)
    assert c2.price == 100

    # run t1 - close out c2, open c1
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1050
    assert s.capital == 5

    c1 = s["c1"]
    assert c1.value == 1045
    assert c1.weight == pytest.approx(1045.0 / 1050)
    assert c1.price == 95

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 100

    # update out t1
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1050
    assert s.capital == 5

    assert c1 == s["c1"]
    assert c1.value == 1045
    assert c1.weight == pytest.approx(1045.0 / 1050)
    assert c1.price == 95

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 100

    # update t2
    i = 2
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 5

    assert c1.value == 1100
    assert c1.weight == pytest.approx(1100.0 / 1105)
    assert c1.price == 100

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 95

    # run t2
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update out t2
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update t3
    i = 3
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # run t3
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update out t3
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update t4
    i = 4
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    # accessing price should refresh - this child has been idle for a while -
    # must make sure we can still have a fresh prices
    assert c1.price == 105
    assert len(c1.prices) == 5

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # run t4
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 105

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update out t4
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 105

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95


def test_strategybase_multiple_calls_preset_secs():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("s", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    data.loc[dts[0], 'c2'] = 95
    data.loc[dts[1], 'c1'] = 95
    data.loc[dts[2], 'c2'] = 95
    data.loc[dts[3], 'c2'] = 95
    data.loc[dts[4], 'c2'] = 95
    data.loc[dts[4], 'c1'] = 105

    s.setup(data)

    # define strategy logic
    def algo(target):
        # close out any open positions
        target.flatten()

        # get stock w/ lowest price
        c = target.universe.loc[target.now].idxmin()

        # allocate all capital to that stock
        target.allocate(target.value, c)

    # replace run logic
    s.run = algo

    # start w/ 1000
    s.adjust(1000)

    # loop through dates manually
    i = 0

    # update t0
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1000

    # run t0
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1000
    assert s.capital == 50

    assert c2.value == 950
    assert c2.weight == pytest.approx(950.0 / 1000)
    assert c2.price == 95

    # update out t0
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1000
    assert s.capital == 50

    assert c2.value == 950
    assert c2.weight == pytest.approx(950.0 / 1000)
    assert c2.price == 95

    # update t1
    i = 1
    s.update(dts[i])

    assert s.value == 1050
    assert s.capital == 50
    assert len(s.children) == 2

    assert c2.value == 1000
    assert c2.weight == pytest.approx(1000.0 / 1050.0)
    assert c2.price == 100

    # run t1 - close out c2, open c1
    s.run(s)

    assert c1.value == 1045
    assert c1.weight == pytest.approx(1045.0 / 1050)
    assert c1.price == 95

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 100

    assert len(s.children) == 2
    assert s.value == 1050
    assert s.capital == 5

    # update out t1
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1050
    assert s.capital == 5

    assert c1.value == 1045
    assert c1.weight == pytest.approx(1045.0 / 1050)
    assert c1.price == 95

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 100

    # update t2
    i = 2
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 5

    assert c1.value == 1100
    assert c1.weight == pytest.approx(1100.0 / 1105)
    assert c1.price == 100

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 95

    # run t2
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update out t2
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update t3
    i = 3
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # run t3
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update out t3
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update t4
    i = 4
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    # accessing price should refresh - this child has been idle for a while -
    # must make sure we can still have a fresh prices
    assert c1.price == 105
    assert len(c1.prices) == 5

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # run t4
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 105

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95

    # update out t4
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1105
    assert s.capital == 60

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 105

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1105)
    assert c2.price == 95


def test_strategybase_multiple_calls_no_post_update():
    s = StrategyBase("s")
    s.set_commissions(lambda q, p: 1)

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    data.loc[dts[0], 'c2'] = 95
    data.loc[dts[1], 'c1'] = 95
    data.loc[dts[2], 'c2'] = 95
    data.loc[dts[3], 'c2'] = 95
    data.loc[dts[4], 'c2'] = 95
    data.loc[dts[4], 'c1'] = 105

    s.setup(data)

    # define strategy logic
    def algo(target):
        # close out any open positions
        target.flatten()

        # get stock w/ lowest price
        c = target.universe.loc[target.now].idxmin()

        # allocate all capital to that stock
        target.allocate(target.value, c)

    # replace run logic
    s.run = algo

    # start w/ 1000
    s.adjust(1000)

    # loop through dates manually
    i = 0

    # update t0
    s.update(dts[i])

    assert len(s.children) == 0
    assert s.value == 1000

    # run t0
    s.run(s)

    assert len(s.children) == 1
    assert s.value == 999
    assert s.capital == 49

    c2 = s["c2"]
    assert c2.value == 950
    assert c2.weight == pytest.approx(950.0 / 999)
    assert c2.price == 95

    # update t1
    i = 1
    s.update(dts[i])

    assert s.value == 1049
    assert s.capital == 49
    assert len(s.children) == 1

    assert "c2" in s.children
    c2 = s["c2"]
    assert c2.value == 1000
    assert c2.weight == pytest.approx(1000.0 / 1049.0)
    assert c2.price == 100

    # run t1 - close out c2, open c1
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1047
    assert s.capital == 2

    c1 = s["c1"]
    assert c1.value == 1045
    assert c1.weight == pytest.approx(1045.0 / 1047)
    assert c1.price == 95

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 100

    # update t2
    i = 2
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1102
    assert s.capital == 2

    assert c1.value == 1100
    assert c1.weight == pytest.approx(1100.0 / 1102)
    assert c1.price == 100

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 95

    # run t2
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1100
    assert s.capital == 55

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1100)
    assert c2.price == 95

    # update t3
    i = 3
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1100
    assert s.capital == 55

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1100)
    assert c2.price == 95

    # run t3
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1098
    assert s.capital == 53

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1098)
    assert c2.price == 95

    # update t4
    i = 4
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1098
    assert s.capital == 53

    assert c1.value == 0
    assert c1.weight == 0
    # accessing price should refresh - this child has been idle for a while -
    # must make sure we can still have a fresh prices
    assert c1.price == 105
    assert len(c1.prices) == 5

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1098)
    assert c2.price == 95

    # run t4
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1096
    assert s.capital == 51

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 105

    assert c2.value == 1045
    assert c2.weight == pytest.approx(1045.0 / 1096)
    assert c2.price == 95


def test_strategybase_prices():
    dts = pd.date_range("2010-01-01", periods=21)
    rawd = [
        13.555,
        13.75,
        14.16,
        13.915,
        13.655,
        13.765,
        14.02,
        13.465,
        13.32,
        14.65,
        14.59,
        14.175,
        13.865,
        13.865,
        13.89,
        13.85,
        13.565,
        13.47,
        13.225,
        13.385,
        12.89,
    ]
    data = pd.DataFrame(index=dts, data=rawd, columns=["a"])

    s = StrategyBase("s")
    s.set_commissions(lambda q, p: 1)
    s.setup(data)

    # buy 100 shares on day 1 - hold until end
    # just enough to buy 100 shares + 1$ commission
    s.adjust(1356.50)

    s.update(dts[0])
    # allocate all capital to child a
    # a should be dynamically created and should have
    # 100 shares allocated. s.capital should be 0
    s.allocate(s.value, "a")

    assert s.capital == 0
    assert s.value == pytest.approx(1355.50)
    assert len(s.children) == 1
    assert s.price == pytest.approx(99.92628, 5)

    a = s["a"]
    assert a.position == 100
    assert a.value == pytest.approx(1355.50)
    assert a.weight == 1
    assert a.price == pytest.approx(13.555)
    assert len(a.prices) == 1

    # update through all dates and make sure price is ok
    s.update(dts[1])
    assert s.price == pytest.approx(101.3638, 4)

    s.update(dts[2])
    assert s.price == pytest.approx(104.3863, 4)

    s.update(dts[3])
    assert s.price == pytest.approx(102.5802, 4)

    # finish updates and make sure ok at end
    for i in range(4, 21):
        s.update(dts[i])

    assert len(s.prices) == 21
    assert s.prices.iloc[-1] == pytest.approx(95.02396, 5)
    assert s.prices.iloc[-2] == pytest.approx(98.67306, 5)


def test_fail_if_root_value_negative():
    s = StrategyBase("s")
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 100
    data.loc[dts[0], "c2"] = 95
    s.setup(data)

    s.adjust(-100)
    # trigger update
    s.update(dts[0])

    assert s.bankrupt

    # make sure only triggered if root negative
    c1 = StrategyBase("c1")
    s = StrategyBase("s", children=[c1])
    c1 = s["c1"]

    s.setup(data)

    s.adjust(1000)
    c1.adjust(-100)
    s.update(dts[0])

    # now make it trigger
    c1.adjust(-1000)
    # trigger update
    s.update(dts[0])

    assert s.bankrupt


def test_fail_if_0_base_in_return_calc():
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 100
    data.loc[dts[0], "c2"] = 95

    # must setup tree because if not negative root error pops up first
    c1 = StrategyBase("c1")
    s = StrategyBase("s", children=[c1])
    c1 = s["c1"]
    s.setup(data)

    s.adjust(1000)
    c1.adjust(100)
    s.update(dts[0])

    c1.adjust(-100)
    s.update(dts[1])

    try:
        c1.adjust(-100)
        s.update(dts[1])
        assert False
    except ZeroDivisionError as e:
        if "Could not update" not in str(e):
            assert False


def test_strategybase_tree_rebalance():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])
    s.set_commissions(lambda q, p: 1)

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now rebalance c1
    s.rebalance(0.5, "c1", update=True)
    assert s.root.stale == True

    assert c1.position == 4
    assert c1.value == 400
    assert s.capital == 1000 - 401
    assert s.value == 999
    assert c1.weight == pytest.approx(400.0 / 999)
    assert c2.weight == 0

    # Check that rebalance with update=False
    # does not mark the node as stale
    s.rebalance(0.6, "c1", update=False)
    assert s.root.stale == False


def test_strategybase_tree_decimal_position_rebalance():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])
    s.use_integer_positions(False)

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000.2)
    s.rebalance(0.42, "c1")
    s.rebalance(0.58, "c2")

    assert c1.value == pytest.approx(420.084)
    assert c2.value == pytest.approx(580.116)
    assert c1.value + c2.value == pytest.approx(1000.2)


def test_rebalance_child_not_in_tree():
    s = StrategyBase("p")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i])
    s.adjust(1000)

    # rebalance to 0 w/ child that is not present - should ignore
    s.rebalance(0, "c2")

    assert s.value == 1000
    assert s.capital == 1000
    assert len(s.children) == 0


def test_strategybase_tree_rebalance_to_0():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now rebalance c1
    s.rebalance(0.5, "c1")

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 500
    assert s.value == 1000
    assert c1.weight == pytest.approx(500.0 / 1000)
    assert c2.weight == 0

    # now rebalance c1
    s.rebalance(0, "c1")

    assert c1.position == 0
    assert c1.value == 0
    assert s.capital == 1000
    assert s.value == 1000
    assert c1.weight == 0
    assert c2.weight == 0


def test_strategybase_tree_rebalance_level2():
    c1 = SecurityBase("c1")
    c12 = copy.deepcopy(c1)
    c2 = SecurityBase("c2")
    c22 = copy.deepcopy(c2)
    s1 = StrategyBase("s1", [c1, c2])
    s2 = StrategyBase("s2", [c12, c22])
    m = StrategyBase("m", [s1, s2])

    s1 = m["s1"]
    s2 = m["s2"]

    c1 = s1["c1"]
    c2 = s1["c2"]

    c12 = s2["c1"]
    c22 = s2["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    m.setup(data)

    i = 0
    m.update(dts[i], data.loc[dts[i]])

    m.adjust(1000)

    assert m.value == 1000
    assert m.capital == 1000
    assert s1.value == 0
    assert s2.value == 0
    assert c1.value == 0
    assert c2.value == 0

    # now rebalance child s1 - since its children are 0, no waterfall alloc
    m.rebalance(0.5, "s1")

    assert s1.value == 500
    assert m.capital == 1000 - 500
    assert m.value == 1000
    assert s1.weight == pytest.approx(500.0 / 1000)
    assert s2.weight == 0

    # now allocate directly to child of child
    s1.rebalance(0.4, "c1")

    assert s1.value == 500
    assert s1.capital == 500 - 200
    assert c1.value == 200
    assert c1.weight == pytest.approx(200.0 / 500)
    assert c1.position == 2

    assert m.capital == 1000 - 500
    assert m.value == 1000
    assert s1.weight == pytest.approx(500.0 / 1000)
    assert s2.weight == 0

    assert c12.value == 0

    # now rebalance child s1 again and make sure c1 also gets proportional
    # increase
    m.rebalance(0.8, "s1")
    assert s1.value == 800
    assert m.capital == pytest.approx(200, 1)
    assert m.value == 1000
    assert s1.weight == 800 / 1000
    assert s2.weight == 0
    assert c1.value == pytest.approx(300.0)
    assert c1.weight == pytest.approx(300.0 / 800)
    assert c1.position == 3

    # now rebalance child s1 to 0 - should close out s1 and c1 as well
    m.rebalance(0, "s1")

    assert s1.value == 0
    assert m.capital == 1000
    assert m.value == 1000
    assert s1.weight == 0
    assert s2.weight == 0
    assert c1.weight == 0


def test_strategybase_tree_rebalance_base():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])
    s.set_commissions(lambda q, p: 1)

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # check that 2 rebalances of equal weight lead to two different allocs
    # since value changes after first call
    s.rebalance(0.5, "c1")

    assert c1.position == 4
    assert c1.value == 400
    assert s.capital == 1000 - 401
    assert s.value == 999
    assert c1.weight == pytest.approx(400.0 / 999)
    assert c2.weight == 0

    s.rebalance(0.5, "c2")

    assert c2.position == 4
    assert c2.value == 400
    assert s.capital == 1000 - 401 - 401
    assert s.value == 998
    assert c2.weight == pytest.approx(400.0 / 998)
    assert c1.weight == pytest.approx(400.0 / 998)

    # close out everything
    s.flatten()

    # adjust to get back to 1000
    s.adjust(4)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now rebalance but set fixed base
    base = s.value
    s.rebalance(0.5, "c1", base=base)

    assert c1.position == 4
    assert c1.value == 400
    assert s.capital == 1000 - 401
    assert s.value == 999
    assert c1.weight == pytest.approx(400.0 / 999)
    assert c2.weight == 0

    s.rebalance(0.5, "c2", base=base)

    assert c2.position == 4
    assert c2.value == 400
    assert s.capital == 1000 - 401 - 401
    assert s.value == 998
    assert c2.weight == pytest.approx(400.0 / 998)
    assert c1.weight == pytest.approx(400.0 / 998)


def test_algo_stack():
    a1 = mock.MagicMock(return_value=True)
    a2 = mock.MagicMock(return_value=False)
    a3 = mock.MagicMock(return_value=True)

    # no run_always for now
    del a1.run_always
    del a2.run_always
    del a3.run_always

    stack = AlgoStack(a1, a2, a3)
    target = mock.MagicMock()
    assert not stack(target)
    assert a1.called
    assert a2.called
    assert not a3.called

    # now test that run_always marked are run

    a1 = mock.MagicMock(return_value=True)
    a2 = mock.MagicMock(return_value=False)
    a3 = mock.MagicMock(return_value=True)

    # a3 will have run_always
    del a1.run_always
    del a2.run_always

    stack = AlgoStack(a1, a2, a3)
    target = mock.MagicMock()
    assert not stack(target)
    assert a1.called
    assert a2.called
    assert a3.called


def test_set_commissions():
    s = StrategyBase("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.set_commissions(lambda x, y: 1.0)
    s.setup(data)
    s.update(dts[0])
    s.adjust(1000)

    s.allocate(500, "c1")
    assert s.capital == 599

    s.set_commissions(lambda x, y: 0.0)
    s.allocate(-400, "c1")
    assert s.capital == 999


def test_strategy_tree_proper_return_calcs():
    s1 = StrategyBase("s1")
    s2 = StrategyBase("s2")
    m = StrategyBase("m", [s1, s2])

    s1 = m["s1"]
    s2 = m["s2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc["c1", dts[1]] = 105
    data.loc["c2", dts[1]] = 95

    m.setup(data)

    i = 0
    m.update(dts[i], data.loc[dts[i]])

    m.adjust(1000)
    # since children have w == 0 this should stay in s
    m.allocate(1000)

    assert m.value == 1000
    assert m.capital == 1000
    assert m.price == 100
    assert s1.value == 0
    assert s2.value == 0

    # now allocate directly to child
    s1.allocate(500)

    assert m.capital == 500
    assert m.value == 1000
    assert m.price == 100
    assert s1.value == 500
    assert s1.weight == pytest.approx(500.0 / 1000)
    assert s1.price == 100
    assert s2.weight == 0

    # allocate to child2 via parent method
    m.allocate(500, "s2")

    assert m.capital == 0
    assert m.value == 1000
    assert m.price == 100
    assert s1.value == 500
    assert s1.weight == pytest.approx(500.0 / 1000)
    assert s1.price == 100
    assert s2.value == 500
    assert s2.weight == pytest.approx(500.0 / 1000)
    assert s2.price == 100

    # now allocate and incur commission fee
    s1.allocate(500, "c1")

    assert m.capital == 0
    assert m.value == 1000
    assert m.price == 100
    assert s1.value == 500
    assert s1.weight == pytest.approx(500.0 / 1000)
    assert s1.price == 100
    assert s2.value == 500
    assert s2.weight == pytest.approx(500.0 / 1000.0)
    assert s2.price == 100


def test_strategy_tree_proper_universes():
    def do_nothing(x):
        return True

    child1 = Strategy("c1", [do_nothing], ["b", "c"])
    parent = Strategy("m", [do_nothing], [child1, "a"])

    child1 = parent["c1"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(
        {
            "a": pd.Series(data=1, index=dts, name="a"),
            "b": pd.Series(data=2, index=dts, name="b"),
            "c": pd.Series(data=3, index=dts, name="c"),
        }
    )

    parent.setup(data, test_data1="test1")

    assert len(parent.children) == 1
    assert "c1" in parent.children
    assert len(parent._universe.columns) == 2
    assert "c1" in parent._universe.columns
    assert "a" in parent._universe.columns

    assert len(child1._universe.columns) == 2
    assert "b" in child1._universe.columns
    assert "c" in child1._universe.columns

    assert parent._has_strat_children
    assert len(parent._strat_children) == 1

    assert parent.get_data("test_data1") == "test1"

    # New child strategy with parent (and using dictionary notation}
    child2 = Strategy(
        "c2", [do_nothing], {"a": SecurityBase(""), "b": ""}, parent=parent
    )
    # Setup the child from the parent, but pass in some additional data
    child2.setup_from_parent(test_data2="test2")
    assert "a" in child2._universe.columns
    assert "b" in child2._universe.columns
    assert "c2" in parent._universe.columns
    # Make sure child has data from the parent and the additional data
    assert child2.get_data("test_data1") == "test1"
    assert child2.get_data("test_data2") == "test2"

    assert len(parent._strat_children) == 2


def test_strategy_tree_paper():
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["a"], data=100.0)
    data.loc[dts[1], "a"] = 101
    data.loc[dts[2], "a"] = 102

    s = Strategy(
        "s",
        [
            bt.algos.SelectWhere(data > 100),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )

    m = Strategy("m", [], [s])
    s = m["s"]

    m.setup(data)
    m.update(dts[0])
    m.run()

    assert m.price == 100
    assert s.price == 100
    assert s._paper_trade
    assert s._paper.price == 100

    s.update(dts[1])
    m.run()

    assert m.price == 100
    assert m.value == 0
    assert s.value == 0
    assert s.price == 100

    s.update(dts[2])
    m.run()

    assert m.price == 100
    assert m.value == 0
    assert s.value == 0
    assert np.allclose(s.price, 100.0 * (102 / 101.0))


def test_dynamic_strategy():
    def do_nothing(x):
        return True

    # Start with an empty parent
    parent = Strategy("p", [do_nothing], [])
    dts = pd.date_range("2010-01-01", periods=4)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    data.loc[dts[2], "c1"] = 105.0
    data.loc[dts[2], "c2"] = 95.0

    parent.setup(data)

    # NOTE: Price of the sub-strategy won't be correct in this example because
    # we are not using the algo stack to impact weights, and so the paper
    # trading strategy does not see the same actions as we are doing.
    i = 0
    parent.adjust(1e6)
    parent.update(dts[i])
    assert parent.price == pytest.approx(100.0)
    assert parent.value == 1e6

    i = 1
    parent.update(dts[i])
    # On this step, we decide to put a trade on c1 vs c2 and track it as a strategy
    trade = Strategy("c1_vs_c2", [], children=["c1", "c2"], parent=parent)
    trade.setup_from_parent()
    trade.update(parent.now)

    assert trade.price == pytest.approx(100.0)
    assert trade.value == 0

    # Allocate capital to the trade
    parent.allocate(1e5, trade.name)
    assert trade.value == 1e5
    assert trade.price == pytest.approx(100.0)

    # Go long 'c1' and short 'c2'
    trade.rebalance(1.0, "c1")
    trade.rebalance(-1.0, "c2")

    assert parent.universe[trade.name][dts[i]] == pytest.approx(100.0)
    assert parent.positions["c1"][dts[i]] == 1e3
    assert parent.positions["c2"][dts[i]] == -1e3

    i = 2
    parent.update(dts[i])
    assert trade.value == 1e5 + 10 * 1e3
    assert parent.value == 1e6 + 10 * 1e3

    # On this step, we close the trade, and allocate capital back to the parent
    trade.flatten()
    trade.update(trade.now)  # Need to update after flattening (for now)
    parent.allocate(-trade.capital, trade.name)
    assert trade.value == 0
    assert trade.capital == 0
    assert parent.value == 1e6 + 10 * 1e3
    assert parent.capital == parent.value
    assert parent.positions["c1"][dts[i]] == pytest.approx(0.0)
    assert parent.positions["c2"][dts[i]] == pytest.approx(0.0)

    i = 3
    parent.update(dts[i])
    # Just make sure we can update one step beyond closing

    # Note that "trade" is still a child of parent, and it also has children,
    # so it will keep getting updated (and paper trading will still happen).
    assert trade.value == 0
    assert trade.capital == 0
    assert trade.values[dts[i]] == pytest.approx(0.0)


def test_dynamic_strategy2():

    # Start with an empty parent
    parent = Strategy("p", [], [])

    dts = pd.date_range("2010-01-01", periods=4)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    data.loc[dts[2], "c1"] = 105.0
    data.loc[dts[2], "c2"] = 95.0
    data.loc[dts[3], "c1"] = 101.0
    data.loc[dts[3], "c2"] = 99.0
    parent.setup(data)

    i = 0
    parent.adjust(1e6)
    parent.update(dts[i])
    assert parent.price == pytest.approx(100.0)
    assert parent.value == 1e6

    i = 1
    parent.update(dts[i])
    # On this step, we decide to put a trade on c1 vs c2 and track it as a strategy
    def trade_c1_vs_c2(strategy):
        if strategy.now == dts[1]:
            strategy.rebalance(1.0, "c1")
            strategy.rebalance(-1.0, "c2")

    trade = Strategy("c1_vs_c2", [trade_c1_vs_c2], children=["c1", "c2"], parent=parent)
    trade.setup_from_parent()
    trade.update(parent.now)

    assert trade.price == pytest.approx(100.0)
    assert trade.value == 0

    # Allocate capital to the trade
    parent.allocate(1e5, trade.name)
    assert trade.value == 1e5
    assert trade.price == pytest.approx(100.0)

    # Run the strategy for the timestep
    parent.run()

    assert parent.universe[trade.name][dts[i]] == pytest.approx(100.0)
    assert np.isnan(parent.universe[trade.name][dts[0]])
    assert parent.positions["c1"][dts[i]] == 1e3
    assert parent.positions["c2"][dts[i]] == -1e3

    i = 2
    parent.update(dts[i])
    trade = parent[trade.name]
    assert trade.value == 1e5 + 10 * 1e3
    assert parent.value == 1e6 + 10 * 1e3
    assert trade.price == pytest.approx(110.0)

    # Next we close the trade by flattening positions
    trade.flatten()
    trade.update(trade.now)  # Need to update after flattening (for now)
    assert trade.price == pytest.approx(110.0)

    # Finally we allocate capital back to the parent to be re-deployed
    parent.allocate(-trade.capital, trade.name)
    assert trade.value == 0
    assert trade.capital == 0

    assert trade.price == pytest.approx(
        110.0
    )  # Price stays the same even after capital de-allocated
    assert parent.value == 1e6 + 10 * 1e3
    assert parent.capital == parent.value
    assert parent.positions["c1"][dts[i]] == pytest.approx(0.0)
    assert parent.positions["c2"][dts[i]] == pytest.approx(0.0)

    i = 3
    parent.update(dts[i])
    # Just make sure we can update one step beyond closing

    assert parent.value == 1e6 + 10 * 1e3

    # Note that "trade" is still a child of parent, and it also has children,
    # so it will keep getting updated (and paper trading will still happen).
    assert trade.value == 0
    assert trade.capital == 0
    assert trade.values[dts[i]] == pytest.approx(0.0)

    # Paper trading price, as asset prices have moved, paper trading price
    # keeps updating. Note that if the flattening of the position was part
    # of the definition of trade_c1_vs_c2, then the paper trading price
    # would be fixed after flattening, as it would apply to both real and paper.
    assert trade.price == pytest.approx(102.0)
    assert parent.universe[trade.name][dts[i]] == pytest.approx(102.0)


def test_outlays():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105
    data.loc[dts[0], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    # allocate 1000 to strategy
    s.adjust(1000)

    # now let's see what happens when we allocate 500 to each child
    c1.allocate(500)
    c2.allocate(500)

    # calling outlays should automatically update the strategy, since stale
    assert c1.outlays[dts[0]] == (4 * 105)
    assert c2.outlays[dts[0]] == (5 * 95)

    assert c1.data["outlay"][dts[0]] == (4 * 105)
    assert c2.data["outlay"][dts[0]] == (5 * 95)

    i = 1
    s.update(dts[i], data.loc[dts[i]])

    c1.allocate(-400)
    c2.allocate(100)

    # out update
    assert c1.outlays[dts[1]] == (-4 * 100)
    assert c2.outlays[dts[1]] == 100

    assert c1.data["outlay"][dts[1]] == (-4 * 100)
    assert c2.data["outlay"][dts[1]] == 100


def test_child_weight_above_1():
    # check for child weights not exceeding 1
    s = StrategyBase("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(np.random.randn(3, 2) + 100, index=dts, columns=["c1", "c2"])
    s.setup(data)

    i = 0
    s.update(dts[i])

    s.adjust(1e6)
    s.allocate(1e6, "c1")
    c1 = s["c1"]

    assert c1.weight <= 1


def test_fixed_commissions():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])
    # fixed $1 commission per transaction
    s.set_commissions(lambda q, p: 1)

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    # allocate 1000 to strategy
    s.adjust(1000)

    # now let's see what happens when we allocate 500 to each child
    c1.allocate(500)
    c2.allocate(500)

    # out update
    s.update(dts[i])

    assert c1.value == 400
    assert c2.value == 400
    assert s.capital == 198

    # de-alloc 100 from c1. This should force c1 to sell 2 units to raise at
    # least 100 (because of commissions)
    c1.allocate(-100)
    s.update(dts[i])

    assert c1.value == 200
    assert s.capital == 198 + 199

    # allocate 100 to c2. This should leave things unchaged, since c2 cannot
    # buy one unit since the commission will cause total outlay to exceed
    # allocation
    c2.allocate(100)
    s.update(dts[i])

    assert c2.value == 400
    assert s.capital == 198 + 199

    # ok try again w/ 101 allocation. This time, it should work
    c2.allocate(101)
    s.update(dts[i])

    assert c2.value == 500
    assert s.capital == 198 + 199 - 101

    # ok now let's close the whole position. Since we are closing, we expect
    # the allocation to go through, even though the outlay > amount
    c2.allocate(-500)
    s.update(dts[i])

    assert c2.value == 0
    assert s.capital == 198 + 199 - 101 + 499

    # now we are going to go short c2
    # we want to 'raise' 100 dollars. Since we need at a minimum 100, but we
    # also have commissions, we will actually short 2 units in order to raise
    # at least 100
    c2.allocate(-100)
    s.update(dts[i])

    assert c2.value == -200
    assert s.capital == 198 + 199 - 101 + 499 + 199


def test_degenerate_shorting():
    # can have situation where you short infinitely if commission/share > share
    # price
    c1 = SecurityBase("c1")
    s = StrategyBase("p", [c1])
    # $1/share commission
    s.set_commissions(lambda q, p: abs(q) * 1)

    c1 = s["c1"]

    dts = pd.date_range("2010-01-01", periods=3)
    # c1 trades at 0.01
    data = pd.DataFrame(index=dts, columns=["c1"], data=0.01)

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)

    try:
        c1.allocate(-10)
        assert False
    except Exception as e:
        assert "full_outlay should always be approaching amount" in str(e)


def test_securitybase_allocate():
    c1 = SecurityBase("c1")
    s = StrategyBase("p", [c1])

    c1 = s["c1"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    # set the price
    data.loc[dts[0], "c1"] = 91.40246706608193
    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    # allocate 100000 to strategy
    original_capital = 100000.0
    s.adjust(original_capital)
    # not integer positions
    c1.integer_positions = False
    # set the full_outlay and amount
    full_outlay = 1999.693706988672
    amount = 1999.6937069886717

    c1.allocate(amount)

    # the results that we want to be true
    assert np.isclose(full_outlay, amount, rtol=0.0)

    # check that the quantity wasn't decreased and the full_outlay == amount
    # we can get the full_outlay that was calculated by
    # original capital - current capital
    assert np.isclose(full_outlay, original_capital - s._capital, rtol=0.0)


def test_securitybase_allocate_commisions():

    date_span = pd.date_range(start="10/1/2017", end="10/11/2017", freq="B")
    numper = len(date_span.values)
    comms = 0.01

    data = [
        [10, 15, 20, 25, 30, 35, 40, 45],
        [10, 10, 10, 10, 20, 20, 20, 20],
        [20, 20, 20, 30, 30, 30, 40, 40],
        [20, 10, 20, 10, 20, 10, 20, 10],
    ]
    data = [[row[i] for row in data] for i in range(len(data[0]))]  # Transpose
    price = pd.DataFrame(data=data, index=date_span)
    price.columns = ["a", "b", "c", "d"]
    # price = price[['a', 'b']]

    sig1 = pd.DataFrame(price["a"] >= price["b"] + 10, columns=["a"])
    sig2 = pd.DataFrame(price["a"] < price["b"] + 10, columns=["b"])
    signal = sig1.join(sig2)

    signal1 = price.diff(1) > 0
    signal2 = price.diff(1) < 0

    tw = price.copy()
    tw.loc[:, :] = 0  # Initialize Set everything to 0

    tw[signal1] = -1.0
    tw[signal2] = 1.0

    s1 = bt.Strategy(
        "long_short",
        [bt.algos.WeighTarget(tw), bt.algos.RunDaily(), bt.algos.Rebalance()],
    )

    ####now we create the Backtest , commissions=(lambda q, p: abs(p * q) * comms)
    t = bt.Backtest(
        s1,
        price,
        initial_capital=1000000,
        commissions=(lambda q, p: abs(p * q) * comms),
        progress_bar=False,
    )

    ####and let's run it!
    res = bt.run(t)
    ########################


def test_strategybase_tree_transact():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)
    # since children have w == 0 this should stay in s
    s.transact(1)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now allocate directly to child
    c1.transact(5)

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 500
    assert s.value == 1000
    assert c1.weight == pytest.approx(500.0 / 1000)
    assert c2.weight == 0

    # now transact the parent since weights are nonzero
    s.transact(2)

    assert c1.position == 6
    assert c1.value == 600
    assert s.capital == 1000 - 600
    assert s.value == 1000
    assert c1.weight == pytest.approx(600.0 / 1000)
    assert c2.weight == 0


def test_strategybase_tree_transact_child_from_strategy():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])
    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)
    # since children have w == 0 this should stay in s
    s.transact(1)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now transact in c1
    s.transact(5, "c1")

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 500
    assert s.value == 1000
    assert c1.weight == pytest.approx(500.0 / 1000)
    assert c2.weight == 0


def test_strategybase_tree_transact_level2():
    c1 = SecurityBase("c1")
    c12 = copy.deepcopy(c1)
    c2 = SecurityBase("c2")
    c22 = copy.deepcopy(c2)
    s1 = StrategyBase("s1", [c1, c2])
    s2 = StrategyBase("s2", [c12, c22])
    m = StrategyBase("m", [s1, s2])

    s1 = m["s1"]
    s2 = m["s2"]

    c1 = s1["c1"]
    c2 = s1["c2"]
    c12 = s2["c1"]
    c22 = s2["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    m.setup(data)

    i = 0
    m.update(dts[i], data.loc[dts[i]])

    m.adjust(1000)
    # since children have w == 0 this should stay in s
    m.transact(1)

    assert m.value == 1000
    assert m.capital == 1000
    assert s1.value == 0
    assert s2.value == 0
    assert c1.value == 0
    assert c2.value == 0

    # now transact directly in child. No weights, so nothing happens
    s1.transact(1)

    assert m.value == 1000
    assert m.capital == 1000
    assert s1.value == 0
    assert s2.value == 0
    assert c1.value == 0
    assert c2.value == 0

    # now transact directly in child of child
    s1.allocate(500)
    c1.transact(2)

    assert s1.value == 500
    assert s1.capital == 500 - 200
    assert c1.value == 200
    assert c1.weight == pytest.approx(200.0 / 500)
    assert c1.position == 2

    assert m.capital == 1000 - 500
    assert m.value == 1000
    assert s1.weight == pytest.approx(500.0 / 1000)
    assert s2.weight == 0

    assert c12.value == 0

    # now transact directly in child again
    s1.transact(5)

    assert s1.value == 500
    assert s1.capital == 500 - 400
    assert c1.value == 400
    assert c1.weight == pytest.approx(400.0 / 500)
    assert c1.position == 4

    assert m.capital == 1000 - 500
    assert m.value == 1000
    assert s1.weight == pytest.approx(500.0 / 1000)
    assert s2.weight == 0

    assert c12.value == 0


def test_strategybase_precision():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    c3 = SecurityBase("c3")
    s = StrategyBase("p", [c1, c2, c3])
    s.use_integer_positions(False)

    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=1.0)

    s.setup(data)

    i = 0
    s.update(dts[i])

    s.adjust(1.0)
    s.rebalance(0.1, "c1")
    s.rebalance(0.1, "c2")
    s.rebalance(0.1, "c3")
    s.adjust(-0.7)

    assert s.capital == pytest.approx(0.0)
    assert s.value == pytest.approx(0.3)
    assert s.price == pytest.approx(100.0)

    assert s.capital != 0  # Due to numerical precision
    assert s.value != 0.3  # Created non-zero value out of numerical precision errors
    assert s.price != 100.0

    # Make sure we can still update and calculate return
    i = 1
    s.update(dts[i])

    assert s.price == pytest.approx(100.0)
    assert s.value == pytest.approx(0.3)

    assert s.price != 100.0
    assert s.value != 0.3


def test_securitybase_transact():
    c1 = SecurityBase("c1")
    s = StrategyBase("p", [c1])

    c1 = s["c1"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    # set the price
    price = 91.40246706608193
    data.loc[dts[0], "c1"] = 91.40246706608193
    s.setup(data)

    i = 0
    s.update(dts[i])

    # allocate 100000 to strategy
    original_capital = 100000.0
    s.adjust(original_capital)
    # not integer positions
    c1.integer_positions = False
    # set the full_outlay and amount
    q = 1000.0
    amount = q * price

    c1.transact(q)

    assert np.isclose(c1.value, amount, rtol=0.0)
    assert np.isclose(c1.weight, amount / original_capital, rtol=0.0)
    assert c1.position == q
    assert np.isclose(c1.outlays.iloc[0], amount, rtol=0.0)

    assert np.isclose(s.capital, (original_capital - amount))
    assert s.weight == 1
    assert s.value == original_capital
    assert np.isclose(s.outlays[c1.name].iloc[0], amount, rtol=0.0)

    # Call again on the same step (and again) to make sure all updates are working
    c1.transact(q)
    c1.transact(q)
    assert c1.position == 3 * q
    assert np.isclose(c1.outlays.iloc[0], 3 * amount, rtol=0.0)
    assert np.isclose(c1.value, 3 * amount, rtol=0.0)

    assert np.isclose(s.capital, (original_capital - 3 * amount))
    assert s.weight == 1
    assert s.value == original_capital
    assert np.isclose(s.outlays[c1.name].iloc[0], 3 * amount, rtol=0.0)


def test_security_setup_positions():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)

    i = 0
    s.update(dts[i])

    assert c1.position == 0
    assert len(c1.positions) == 1
    assert c1.positions.iloc[0] == 0

    assert c2.position == 0
    assert len(c2.positions) == 1
    assert c2.positions.iloc[0] == 0


def test_couponpayingsecurity_setup():
    c1 = CouponPayingSecurity("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105
    data.loc[dts[0], "c2"] = 95

    coupons = pd.DataFrame(index=dts, columns=["c1"], data=0.1)

    s.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i])

    assert "coupon" in c1.data
    assert c1.coupon == pytest.approx(0.0)
    assert len(c1.coupons) == 1
    assert c1.coupons.iloc[0] == pytest.approx(0.0)

    assert "holding_cost" in c1.data
    assert c1.holding_cost == pytest.approx(0.0)
    assert len(c1.holding_costs) == 1
    assert c1.holding_costs.iloc[0] == pytest.approx(0.0)

    assert c1.price == 105
    assert len(c1.prices) == 1
    assert c1.prices.iloc[0] == 105

    assert c2.price == 95
    assert len(c2.prices) == 1
    assert c2.prices.iloc[0] == 95


def test_couponpayingsecurity_setup_costs():
    c1 = CouponPayingSecurity("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105
    data.loc[dts[0], "c2"] = 95

    coupons = pd.DataFrame(index=dts, columns=["c1"], data=0.0)
    cost_long = pd.DataFrame(index=dts, columns=["c1"], data=0.01)
    cost_short = pd.DataFrame(index=dts, columns=["c1"], data=0.05)

    s.setup(data, coupons=coupons, cost_long=cost_long, cost_short=cost_short)

    i = 0
    s.update(dts[i])

    assert "coupon" in c1.data
    assert c1.coupon == pytest.approx(0.0)
    assert len(c1.coupons) == 1
    assert c1.coupons.iloc[0] == pytest.approx(0.0)

    assert "holding_cost" in c1.data
    assert c1.holding_cost == pytest.approx(0.0)
    assert len(c1.holding_costs) == 1
    assert c1.holding_costs.iloc[0] == pytest.approx(0.0)

    assert c1.price == 105
    assert len(c1.prices) == 1
    assert c1.prices.iloc[0] == 105

    assert c2.price == 95
    assert len(c2.prices) == 1
    assert c2.prices.iloc[0] == 95


def test_couponpayingsecurity_carry():
    c1 = CouponPayingSecurity("c1")
    s = StrategyBase("p", [c1])

    c1 = s["c1"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1"], data=1.0)

    coupons = pd.DataFrame(index=dts, columns=["c1"], data=0.0)
    coupons.loc[dts[0], "c1"] = 0.1
    cost_long = pd.DataFrame(index=dts, columns=["c1"], data=0.0)
    cost_long.loc[dts[0], "c1"] = 0.01
    cost_short = pd.DataFrame(index=dts, columns=["c1"], data=0.05)

    s.setup(data, coupons=coupons, cost_long=cost_long, cost_short=cost_short)

    i = 0
    s.update(dts[i])

    # allocate 1000 to strategy
    original_capital = 1000.0
    s.adjust(original_capital)
    # set the full_outlay and amount
    q = 1000.0
    c1.transact(q)

    assert c1.coupon == pytest.approx(100.0)
    assert len(c1.coupons) == 1
    assert c1.coupons.iloc[0] == pytest.approx(100.0)
    assert c1.holding_cost == pytest.approx(10.0)
    assert len(c1.holding_costs) == 1
    assert c1.holding_costs.iloc[0] == pytest.approx(10.0)

    assert s.capital == pytest.approx(0.0)
    assert s.cash.iloc[0] == pytest.approx(0.0)

    # On this step, the coupon/costs will be accounted for from the last holding
    i = 1
    s.update(dts[i])

    assert c1.coupon == pytest.approx(0.0)
    assert len(c1.coupons) == 2
    assert c1.coupons.iloc[1] == pytest.approx(0.0)
    assert c1.holding_cost == pytest.approx(0.0)
    assert len(c1.holding_costs) == 2
    assert c1.holding_costs.iloc[1] == pytest.approx(0.0)

    assert s.capital == pytest.approx(100.0 - 10.0)
    assert s.cash.iloc[0] == pytest.approx(0.0)
    assert s.cash.iloc[1] == pytest.approx(100.0 - 10.0)

    # Go short q
    c1.transact(-2 * q)
    # Note cost is positive even though we are short.
    assert c1.holding_cost == pytest.approx(50.0)
    assert len(c1.holding_costs) == 2
    assert c1.holding_costs.iloc[1] == pytest.approx(50.0)


def test_couponpayingsecurity_transact():
    c1 = CouponPayingSecurity("c1")
    s = StrategyBase("p", [c1])

    c1 = s["c1"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    # set the price
    price = 91.40246706608193
    data.loc[dts[0], "c1"] = 91.40246706608193
    data.loc[dts[1], "c1"] = 91.40246706608193

    coupon = 0.1
    coupons = pd.DataFrame(index=dts, columns=["c1"], data=0.0)
    coupons.loc[dts[0], "c1"] = coupon

    s.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i])

    # allocate 100000 to strategy
    original_capital = 100000.0
    s.adjust(original_capital)
    # set the full_outlay and amount
    q = 1000.0
    amount = q * price
    c1.transact(q)

    # The coupon is nonzero, but will only be counted in "value" the next day
    assert c1.coupon == coupon * q
    assert len(c1.coupons) == 1
    assert c1.coupons.iloc[0] == coupon * q

    assert np.isclose(c1.value, amount, rtol=0.0)
    assert np.isclose(c1.weight, amount / original_capital, rtol=0.0)
    assert c1.position == q

    assert s.capital == (original_capital - amount)
    assert s.cash.iloc[0] == (original_capital - amount)
    assert s.weight == 1
    assert s.value == original_capital

    assert c1._capital == coupon * q

    # On this step, the coupon will be paid
    i = 1
    s.update(dts[i])
    new_capital = original_capital + coupon * q
    assert c1.coupon == 0
    assert len(c1.coupons) == 2
    assert c1.coupons.iloc[0] == coupon * q
    assert c1.coupons.iloc[1] == 0

    assert np.isclose(c1.value, amount, rtol=0.0)
    assert np.isclose(c1.weight, amount / new_capital, rtol=0.0)
    assert c1.position == q

    assert s.capital == (new_capital - amount)
    assert s.weight == 1
    assert s.value == new_capital
    assert s.cash.iloc[0] == (original_capital - amount)
    assert s.cash.iloc[1] == (new_capital - amount)

    assert c1._capital == 0

    # Close the position
    c1.transact(-q)

    assert c1.coupon == 0
    assert len(c1.coupons) == 2
    assert c1.coupons.iloc[0] == coupon * q
    assert c1.coupons.iloc[1] == 0

    assert np.isclose(c1.value, 0.0, rtol=0.0)
    assert np.isclose(c1.weight, 0.0 / new_capital, rtol=0.0)
    assert c1.position == 0

    assert s.capital == new_capital
    assert s.weight == 1
    assert s.value == new_capital
    assert s.cash.iloc[0] == (original_capital - amount)
    assert s.cash.iloc[1] == new_capital

    assert c1._capital == 0


def test_bidoffer():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105
    data.loc[dts[0], "c2"] = 95

    bidoffer = pd.DataFrame(index=dts, columns=["c1", "c2"], data=1.0)
    bidoffer.loc[dts[0], "c1"] = 2
    bidoffer.loc[dts[0], "c2"] = 1.5

    s.setup(data, bidoffer=bidoffer)
    s.adjust(100000)
    i = 0
    s.update(dts[i])

    assert c1.bidoffer == 2
    assert len(c1.bidoffers) == 1
    assert c1.bidoffers.iloc[0] == 2

    assert c2.bidoffer == pytest.approx(1.5)
    assert len(c2.bidoffers) == 1
    assert c2.bidoffers.iloc[0] == pytest.approx(1.5)

    # Check the outlays are adjusted for bid/offer
    s.set_commissions(lambda q, p: 0.1)

    total, outlay, fee, bidoffer = c1.outlay(100)
    assert bidoffer == 100 * 1
    assert fee == pytest.approx(0.1)
    assert outlay == 100 * (105 + 1)
    assert total == outlay + fee

    total, outlay, fee, bidoffer = c1.outlay(-100)
    assert bidoffer == 100 * 1
    assert fee == pytest.approx(0.1)
    assert outlay == -100 * (105 - 1)
    assert total == outlay + fee

    total, outlay, fee, bidoffer = c2.outlay(100)
    assert bidoffer == pytest.approx(100 * 0.75)
    assert fee == pytest.approx(0.1)
    assert outlay == pytest.approx(100 * (95 + 0.75))
    assert total == outlay + fee

    total, outlay, fee, bidoffer = c2.outlay(-100)
    assert bidoffer == pytest.approx(100 * 0.75)
    assert fee == pytest.approx(0.1)
    assert outlay == pytest.approx(-100 * (95 - 0.75))
    assert total == outlay + fee

    # Do some transactions, and check that bidoffer_paid is updated
    c1.transact(100)
    assert c1.bidoffer_paid == 100 * 1
    assert c1.bidoffers_paid.iloc[i] == c1.bidoffer_paid
    c1.transact(100)
    assert c1.bidoffer_paid == 200 * 1
    assert c1.bidoffers_paid.iloc[i] == c1.bidoffer_paid

    c2.transact(-100)
    assert c2.bidoffer_paid == pytest.approx(100 * 0.75)
    assert c2.bidoffers_paid.iloc[i] == c2.bidoffer_paid
    assert s.bidoffer_paid == pytest.approx(100 * 0.75 + 200 * 1)
    assert s.bidoffers_paid.iloc[i] == s.bidoffer_paid

    assert s.fees.iloc[i] == pytest.approx(3 * 0.1)

    i = 1
    s.update(dts[i])
    assert c1.bidoffer_paid == pytest.approx(0.0)
    assert c1.bidoffers_paid.iloc[i] == c1.bidoffer_paid
    assert c2.bidoffer_paid == pytest.approx(0.0)
    assert c2.bidoffers_paid.iloc[i] == c2.bidoffer_paid
    assert s.bidoffer_paid == pytest.approx(0.0)
    assert s.bidoffers_paid.iloc[i] == s.bidoffer_paid
    assert s.fees.iloc[i] == pytest.approx(0.0)


def test_outlay_custom():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105

    s.setup(data)
    s.adjust(100000)
    i = 0
    s.update(dts[i])

    # Check the outlays are adjusted for custom prices
    s.set_commissions(lambda q, p: 0.1 * p)

    total, outlay, fee, bidoffer = c1.outlay(100, 106)
    assert bidoffer == 100 * 1
    assert fee == pytest.approx(0.1 * 106)
    assert outlay == 100 * (106)
    assert total == outlay + fee

    total, outlay, fee, bidoffer = c1.outlay(-100, 106)
    assert bidoffer == -100 * 1
    assert fee == pytest.approx(0.1 * 106)
    assert outlay == -100 * 106
    assert total == outlay + fee


def test_bidoffer_custom():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = StrategyBase("p", [c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105

    # Note: In order to access bidoffer_paid,
    # need to pass bidoffer kwarg during setup
    s.setup(data, bidoffer={})
    s.adjust(100000)
    i = 0
    s.update(dts[i])

    c1.transact(100, price=106)
    assert c1.bidoffer_paid == 100 * 1
    assert s.bidoffer_paid == c1.bidoffer_paid
    assert s.capital == 100000 - 100 * 106
    assert c1.bidoffers_paid.iloc[i] == c1.bidoffer_paid
    assert s.bidoffers_paid.iloc[i] == s.bidoffer_paid

    c1.transact(100, price=106)
    assert c1.bidoffer_paid == 200 * 1
    assert s.bidoffer_paid == c1.bidoffer_paid
    assert s.capital == 100000 - 100 * 106 - 100 * 106
    assert c1.bidoffers_paid.iloc[i] == c1.bidoffer_paid
    assert s.bidoffers_paid.iloc[i] == s.bidoffer_paid

    c1.transact(-100, price=107)
    assert c1.bidoffer_paid == 0
    assert s.bidoffer_paid == c1.bidoffer_paid
    assert s.capital == 100000 - 100 * 106 - 100 * 106 + 100 * 107
    assert c1.bidoffers_paid.iloc[i] == c1.bidoffer_paid
    assert s.bidoffers_paid.iloc[i] == s.bidoffer_paid


def test_security_notional_value():
    c1 = SecurityBase("c1")
    c2 = CouponPayingSecurity("c2")
    c3 = HedgeSecurity("c3")
    c4 = CouponPayingHedgeSecurity("c4")
    c5 = FixedIncomeSecurity("c5")

    s = StrategyBase("p", children=[c1, c2, c3, c4, c5])

    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]
    c4 = s["c4"]
    c5 = s["c5"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4", "c5"], data=100.0)
    coupons = pd.DataFrame(index=dts, columns=["c2", "c4"], data=0.0)
    s.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i])

    c1.transact(1000)
    c2.transact(1000)
    c3.transact(1000)
    c4.transact(1000)
    c5.transact(1000)
    for c in [c1, c2, c3, c4, c5]:
        assert c.position == 1000
        assert c.price == 100
    assert c1.notional_value == pytest.approx(1000 * 100.0)
    assert c2.notional_value == 1000
    assert c3.notional_value == 0
    assert c4.notional_value == 0
    assert c5.notional_value == 1000
    for c in [c1, c2, c3, c4, c5]:
        assert len(c.notional_values) == 1
        assert c.notional_values[dts[i]] == c.notional_value
    assert (
        s.notional_value == 2000 + 1000 * 100
    )  # Strategy notional value always positive

    i = 1
    s.update(dts[i])

    c1.transact(-3000)
    c2.transact(-3000)
    c3.transact(-3000)
    c4.transact(-3000)
    c5.transact(-3000)
    for c in [c1, c2, c3, c4, c5]:
        assert c.position == -2000
        assert c.price == 100
    assert c1.notional_value == pytest.approx(-2000 * 100.0)
    assert c2.notional_value == -2000
    assert c3.notional_value == 0
    assert c4.notional_value == 0
    assert c5.notional_value == -2000
    for c in [c1, c2, c3, c4, c5]:
        assert len(c.notional_values) == 2
        assert c.notional_values[dts[i]] == c.notional_value
    assert (
        s.notional_value == 2000 * 100 + 4000
    )  # Strategy notional value always positive


# FixedIncomeStrategy Tests


def test_fi_strategy_flag():
    s1 = SecurityBase("s1")
    s2 = SecurityBase("s2")
    s = StrategyBase("p", children=[s1, s2])
    assert s.fixed_income == False

    s = FixedIncomeStrategy("p", [s1, s2])
    assert s.fixed_income == True


def test_fi_strategy_no_bankruptcy():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = FixedIncomeStrategy("p", children=[c1, c2])

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])
    s.transact(10, "c2")
    assert s.value == pytest.approx(0.0)
    assert s.capital == -10 * 100

    i = 1
    s.update(dts[i], data.loc[dts[i]])
    assert s.value == -5 * 10
    assert s.capital == -10 * 100
    assert s.bankrupt == False


def test_fi_strategy_tree_adjust():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = FixedIncomeStrategy("p", children=[c1, c2])

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    # Basic setup works with no adjustment
    assert s.value == 0
    assert c1.value == 0
    assert c2.value == 0
    assert c1.weight == 0
    assert c2.weight == 0
    assert c1.notional_value == 0
    assert c2.notional_value == 0

    # Positive or negative capital adjustments are fine
    s.adjust(1000)
    assert s.capital == 1000
    assert s.value == 1000

    s.adjust(-2000)
    assert s.capital == -1000
    assert s.value == -1000


def test_fi_strategy_tree_update():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = FixedIncomeStrategy("p", children=[c1, c2])
    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = -5  # Test negative prices
    data.loc[dts[2], "c2"] = 0  # Test zero price

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    assert c1.price == 100
    assert c2.price == 100

    i = 1
    s.update(dts[i], data.loc[dts[i]])

    assert c1.price == 105
    assert c2.price == -5

    i = 2
    s.update(dts[i], data.loc[dts[i]])

    assert c1.price == 100
    assert c2.price == 0


def test_fi_strategy_tree_allocate():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = FixedIncomeStrategy("p", children=[c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)
    # since children have w == 0 this should stay in s
    s.allocate(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now allocate directly to child
    c1.allocate(500)

    assert c1.position == 5
    assert c1.value == 500
    assert c1.notional_value == 500
    assert s.capital == 1000 - 500
    assert s.value == 1000
    assert s.notional_value == 500  # Capital does not count towards notl
    assert c1.weight == pytest.approx(1.0)
    assert c2.weight == 0


def test_fi_strategy_tree_allocate_child_from_strategy():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = FixedIncomeStrategy("p", children=[c1, c2])
    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    s.adjust(1000)
    # since children have w == 0 this should stay in s
    s.allocate(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now allocate to c1
    s.allocate(500, "c1")

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 500
    assert s.value == 1000
    assert c1.weight == pytest.approx(1.0)
    assert c2.weight == 0


def test_fi_strategy_close():
    c1 = SecurityBase("c1")
    c2 = CouponPayingSecurity("c2")
    c3 = HedgeSecurity("c3")
    c4 = CouponPayingHedgeSecurity("c4")

    s = FixedIncomeStrategy("p", children=[c1, c2, c3, c4])

    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]
    c4 = s["c4"]

    dts = pd.date_range("2010-01-01", periods=3)
    # Price
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    coupons = pd.DataFrame(index=dts, columns=["c2", "c4"], data=0.0)
    s.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i])

    for c in [c1, c2, c3, c4]:
        s.transact(10, c.name)

        assert c.position == 10
        assert c.value == 1000
        assert s.capital == -1000
        assert s.value == 0

        s.close(c.name)

        assert c.position == 0
        assert c.value == 0
        assert s.capital == 0
        assert s.value == 0

        s.transact(-10, c.name)
        assert c.position == -10
        assert c.value == -1000
        assert s.capital == 1000
        assert s.value == 0

        s.close(c.name)

        assert c.position == 0
        assert c.value == 0
        assert s.capital == 0
        assert s.value == 0


def test_fi_strategy_close_zero_price():
    c1 = SecurityBase("c1")
    c2 = CouponPayingSecurity("c2")
    c3 = HedgeSecurity("c3")
    c4 = CouponPayingHedgeSecurity("c4")

    s = FixedIncomeStrategy("p", children=[c1, c2, c3, c4])

    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]
    c4 = s["c4"]

    dts = pd.date_range("2010-01-01", periods=3)
    # Zero prices are OK in fixed income space (i.e. swaps)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=0.0)
    coupons = pd.DataFrame(index=dts, columns=["c2", "c4"], data=0.0)
    s.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i])

    for c in [c1, c2, c3, c4]:
        s.transact(10, c.name)
        assert c.position == 10
        assert c.value == 0
        s.close(c.name)
        assert c.position == 0
        assert c.value == 0

        s.transact(-10, c.name)
        assert c.position == -10
        assert c.value == 0
        s.close(c.name)
        assert c.position == 0
        assert c.value == 0


def test_fi_strategy_flatten():
    c1 = SecurityBase("c1")
    c2 = CouponPayingSecurity("c2")
    c3 = HedgeSecurity("c3")
    c4 = CouponPayingHedgeSecurity("c4")

    s = FixedIncomeStrategy("p", children=[c1, c2, c3, c4])

    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]
    c4 = s["c4"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    coupons = pd.DataFrame(index=dts, columns=["c2", "c4"], data=0.0)
    s.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i])

    for c in [c1, c2, c3, c4]:
        s.transact(10, c.name)

    for c in [c1, c2, c3, c4]:
        assert c.position == 10
        assert c.value == 1000

    s.flatten()

    for c in [c1, c2, c3, c4]:
        assert c.position == 0
        assert c.value == 0


def test_fi_strategy_prices():
    c1 = CouponPayingSecurity("c1")
    s = FixedIncomeStrategy("s", children=[c1])
    c1 = s["c1"]

    dts = pd.date_range("2010-01-01", periods=4)
    rawd = [2, -3, 0, 1]
    data = pd.DataFrame(index=dts, data=rawd, columns=["c1"])

    coupons = pd.DataFrame(index=dts, columns=["c1"], data=[1, 2, 3, 4])
    s.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i])

    s.transact(10, "c1")
    assert c1.coupon == 10 * 1
    assert s.capital == -10 * 2
    assert s.value == 0
    assert len(s.children) == 1
    assert s.price == 100
    assert s.notional_value == 10
    last_coupon = c1.coupon
    last_value = s.value
    last_notional_value = s.notional_value
    last_price = 100.0

    i = 1
    s.update(dts[i])
    cpn = last_coupon
    assert c1.coupon == 10 * 2
    assert s.capital == -10 * 2 + cpn
    assert s.value == -5 * 10 + cpn  # MTM + coupon
    assert s.notional_value == 10
    assert s.price == last_price + 100 * (s.value - last_value) / last_notional_value
    last_value = s.value
    last_notional_value = s.notional_value
    last_price = s.price
    last_coupon = c1.coupon

    i = 2
    s.update(dts[i])
    cpn += last_coupon
    assert c1.coupon == 10 * 3
    assert s.capital == -10 * 2 + cpn
    assert s.value == -2 * 10 + cpn  # MTM + coupon
    assert s.notional_value == 10
    assert s.price == last_price + 100 * (s.value - last_value) / last_notional_value
    last_value = s.value
    last_notional_value = s.notional_value
    last_price = s.price
    last_coupon = c1.coupon

    i = 3
    s.update(dts[i])
    s.transact(10, "c1")
    # Coupon still from previous period - not affected by new transaction
    cpn += last_coupon
    assert c1.coupon == 20 * 4
    assert s.capital == -10 * 2 - 10 * 1 + cpn
    assert s.value == -1 * 10 + 0 + cpn  # MTM + coupon
    assert s.notional_value == 20
    assert s.price == last_price + 100 * (s.value - last_value) / last_notional_value


def test_fi_fail_if_0_base_in_return_calc():
    c1 = HedgeSecurity("c1")
    s = FixedIncomeStrategy("s", children=[c1])
    c1 = s["c1"]

    dts = pd.date_range("2010-01-01", periods=4)
    rawd = [2, -3, 0, 1]
    data = pd.DataFrame(index=dts, data=rawd, columns=["c1"])

    s.setup(data)

    i = 0
    s.update(dts[i])

    assert s.notional_value == 0
    # Hedge security has no notional value, so strategy doesn't either
    # and thus shouldn't be making PNL.

    i = 1
    try:
        s.update(dts[i])
    except ZeroDivisionError as e:
        if "Could not update" not in str(e):
            assert False


def test_fi_strategy_tree_rebalance():
    c1 = SecurityBase("c1")
    c2 = CouponPayingSecurity("c2")
    c3 = HedgeSecurity("c3")
    c4 = CouponPayingHedgeSecurity("c4")

    s = FixedIncomeStrategy("p", children=[c1, c2, c3, c4])

    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]
    c4 = s["c4"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=50.0)
    coupons = pd.DataFrame(index=dts, columns=["c2", "c4"], data=0.0)
    s.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i], data.loc[dts[i]])

    assert s.value == 0
    assert s.capital == 0
    assert c1.value == 0
    assert c2.value == 0

    # now rebalance c1
    s.rebalance(0.5, "c1", base=1000)

    assert c1.position == 10
    assert c1.value == 500
    assert c1.notional_value == 500
    assert s.capital == -500
    assert s.value == 0
    assert s.notional_value == 500
    assert c1.weight == pytest.approx(1.0)
    assert c2.weight == 0

    assert c2.notional_value == 0

    # Now rebalance to s2, with no base weight.
    # It takes base weight from strategy weight (500)
    s.rebalance(0.5, "c2")
    assert c1.position == 10
    assert c1.notional_value == 500
    assert c2.position == 250
    assert c2.notional_value == 250

    assert s.notional_value == c1.notional_value + c2.notional_value

    assert c1.weight == pytest.approx(2.0 / 3.0)
    assert c2.weight == pytest.approx(1.0 / 3.0)

    assert s.value == 0

    i = 1
    s.update(dts[i], data.loc[dts[i]])
    # Now rebalance to a new, higher base with given target weights (including negative)
    s.rebalance(0.5, "c1", 1000, update=False)
    s.rebalance(-0.5, "c2", 1000)

    assert c1.weight == pytest.approx(0.5)
    assert c2.weight == pytest.approx(-0.5)
    assert c1.position == 10
    assert c1.notional_value == 500
    assert c2.position == -500
    assert c2.notional_value == -500


def test_fi_strategy_tree_rebalance_nested():
    c1 = CouponPayingSecurity("c1")
    c2 = CouponPayingSecurity("c2")

    s1 = FixedIncomeStrategy("s1", children=[c1, c2])
    s2 = FixedIncomeStrategy("s2", children=[c1, c2])
    s = FixedIncomeStrategy("s", children=[s1, s2])
    p = FixedIncomeStrategy("p", children=[c1, c2])

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=50.0)
    coupons = pd.DataFrame(index=dts, columns=["c1", "c2"], data=0.0)
    s.setup(data, coupons=coupons)
    p.setup(data, coupons=coupons)

    i = 0
    s.update(dts[i])
    p.update(dts[i])

    s["s1"].transact(100, "c1")
    s["s2"].transact(100, "c2")
    p.transact(100, "c1")
    p.transact(100, "c2")

    assert s["s1"]["c1"].position == 100
    assert s["s2"]["c2"].position == 100
    assert p["c1"].position == 100
    assert p["c2"].position == 100
    s.update(dts[i])  # Force update to be safe

    base = s.notional_value
    s.rebalance(0.5, "s1", base * 10, update=False)
    s.rebalance(-0.5, "s2", base * 10)
    p.rebalance(5, "c1", update=False)
    p.rebalance(-5, "c2")

    s.update(dts[i])  # Force update to be safe
    assert s["s1"]["c1"].position == 1000
    assert s["s2"]["c2"].position == -1000
    assert s["s1"]["c1"].weight == pytest.approx(1.0)
    assert s["s2"]["c2"].weight == -1
    assert p["c1"].position == 1000
    assert p["c2"].position == -1000

    # Note that even though the security weights are signed,
    # the strategy weights are all positive (and hence not equal)
    # to the weight passed in to the rebalance call
    assert s["s1"].weight == pytest.approx(0.5)
    assert s["s2"].weight == pytest.approx(0.5)

    assert s.value == pytest.approx(0.0)
    assert p.value == pytest.approx(0.0)
    assert s.capital == 0
    assert p.capital == 0


def test_fi_strategy_precision():
    N = 100
    children = [SecurityBase("c%i" % i) for i in range(N)]
    s = FixedIncomeStrategy("p", children=children)
    children = [s[c.name] for c in children]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=[c.name for c in children], data=1.0)

    s.setup(data)

    i = 0
    s.update(dts[i])

    for c in children:
        c.transact(0.1)

    # Even within tolerance, value is nonzero
    assert s.value == pytest.approx(0, 14)
    assert not is_zero(s.value)
    # Notional value not quite equal to N * 0.1
    assert s.notional_value == pytest.approx(sum(0.1 for _ in range(N)))
    assert s.notional_value != N * 0.1
    assert s.price == pytest.approx(100.0)

    old_value = s.value
    old_notional_value = s.notional_value

    # Still make sure we can update - PNL nonzero, and last notional value is zero
    i = 1
    s.update(dts[i])
    assert s.price == pytest.approx(100.0)
    # Even within tolerance, value is nonzero
    assert s.value == old_value
    assert s.notional_value == old_notional_value

    # The weights also have numerical precision issues
    assert children[0].weight == pytest.approx(1 / float(N), 16)
    assert children[0].weight != 1 / float(N)

    # Now rebalance "out" of an asset with the almost zero weight
    new_weight = children[0].weight - 1 / float(N)
    s.rebalance(new_weight, children[0].name)

    # Check that the position is still closed completely
    assert children[0].position == 0


def test_fi_strategy_bidoffer():
    c1 = SecurityBase("c1")
    c2 = SecurityBase("c2")
    s = FixedIncomeStrategy("p", children=[c1, c2])

    c1 = s["c1"]
    c2 = s["c2"]

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[0], "c1"] = 105
    data.loc[dts[0], "c2"] = 95

    bidoffer = pd.DataFrame(index=dts, columns=["c1", "c2"], data=1.0)
    bidoffer.loc[dts[0], "c1"] = 2
    bidoffer.loc[dts[0], "c2"] = 1.5

    s.setup(data, bidoffer=bidoffer)
    i = 0
    s.update(dts[i])
    assert s.value == pytest.approx(0.0)
    assert s.price == pytest.approx(100.0)

    # Do some transactions, and check that bidoffer_paid is updated
    c1.transact(100)
    assert c1.bidoffer_paid == 100 * 1
    assert c1.bidoffers_paid.iloc[i] == c1.bidoffer_paid
    c1.transact(100)
    assert c1.bidoffer_paid == 200 * 1
    assert c1.bidoffers_paid.iloc[i] == c1.bidoffer_paid

    c2.transact(-100)
    assert c2.bidoffer_paid == pytest.approx(100 * 0.75)
    assert c2.bidoffers_paid.iloc[i] == c2.bidoffer_paid

    s.update(dts[i])
    assert s.bidoffer_paid == pytest.approx(275.0)
    assert s.bidoffers_paid.iloc[i] == s.bidoffer_paid
    assert s.value == pytest.approx(-275.0)
    assert s.notional_value == 105 * 200 + 95 * 100
    assert s.price == pytest.approx(100 * (1.0 - 275.0 / (105 * 200 + 95 * 100)))

    old_notional = s.notional_value
    old_value = s.value
    old_price = s.price

    i = 1
    s.update(dts[i])
    assert s.bidoffer_paid == pytest.approx(0.0)
    assert s.bidoffers_paid.iloc[i] == s.bidoffer_paid
    assert s.value == pytest.approx(-275.0 - 200 * 5 - 100 * 5)  # Bid-offer paid
    assert s.notional_value == 100 * 200 + 100 * 100
    new_value = s.value
    assert s.price == old_price + 100 * (new_value - old_value) / old_notional


def test_strategy_combined_universe_regression():
    """This test checks for regressions with children strategies of a parent strategy,
    and how those related to underlying securities"""
    child_strategy1 = Strategy(
        "child_strategy1",
        [
            bt.algos.RunOnce(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )

    child_strategy2 = Strategy(
        "child_strategy2",
        [
            bt.algos.RunOnce(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    tests = [
        bt.Backtest(child_strategy1, data),
        bt.Backtest(child_strategy2, data),
    ]

    parent_strategy = bt.Strategy(
        "parent_strategy",
        [
            bt.algos.RunOnce(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
        children=[x.strategy for x in tests],
    )
    test = bt.Backtest(parent_strategy, data)
    result = bt.run(test)
    weights = result.get_security_weights()

    assert result["parent_strategy"]
    assert list(parent_strategy.children.keys()) == [
        "child_strategy1",
        "child_strategy2",
    ]


def test_strategy_combined_universe_regression_backtest_run_first():
    """This test checks for regressions with children strategies of a parent strategy,
    and how those related to underlying securities"""
    child_strategy1 = Strategy(
        "child_strategy1",
        [
            bt.algos.RunOnce(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )

    child_strategy2 = Strategy(
        "child_strategy2",
        [
            bt.algos.RunOnce(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    tests = [
        bt.Backtest(child_strategy1, data),
        bt.Backtest(child_strategy2, data),
    ]

    results = []
    for test in tests:
        results.append(bt.run(test))

    merged_prices_df = bt.merge(results[0].prices, results[1].prices)

    parent_strategy = bt.Strategy(
        "parent_strategy",
        [
            bt.algos.RunOnce(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
        children=[x.strategy for x in tests],
    )
    test = bt.Backtest(parent_strategy, merged_prices_df)
    result = bt.run(test)
    weights = result.get_security_weights()

    assert result["parent_strategy"]
    assert list(parent_strategy.children.keys()) == [
        "child_strategy1",
        "child_strategy2",
    ]
    result_obj = result["parent_strategy"]
    assert len(weights.columns) == 2
