from __future__ import division
import copy

import bt
from bt.core import Node, StrategyBase, SecurityBase, AlgoStack, Strategy
import pandas as pd
from nose.tools import assert_almost_equal as aae
import mock


def test_node_tree():
    c1 = Node('c1')
    c2 = Node('c2')
    p = Node('p', children=[c1, c2])

    c1 = p['c1']
    c2 = p['c2']

    assert len(p.children) == 2
    assert 'c1' in p.children
    assert 'c2' in p.children
    assert p == c1.parent
    assert p == c2.parent

    m = Node('m', children=[p])
    p = m['p']
    c1 = p['c1']
    c2 = p['c2']

    assert len(m.children) == 1
    assert 'p' in m.children
    assert p.parent == m
    assert len(p.children) == 2
    assert 'c1' in p.children
    assert 'c2' in p.children
    assert p == c1.parent
    assert p == c2.parent


def test_strategybase_tree():
    s1 = SecurityBase('s1')
    s2 = SecurityBase('s2')
    s = StrategyBase('p', [s1, s2])

    s1 = s['s1']
    s2 = s['s2']

    assert len(s.children) == 2
    assert 's1' in s.children
    assert 's2' in s.children
    assert s == s1.parent
    assert s == s2.parent


def test_node_members():
    s1 = SecurityBase('s1')
    s2 = SecurityBase('s2')
    s = StrategyBase('p', [s1, s2])

    s1 = s['s1']
    s2 = s['s2']

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
    s1 = SecurityBase('s1')
    s2 = SecurityBase('s2')
    s = StrategyBase('p', [s1, s2])

    # we cannot access s1 and s2 directly since they are copied
    # we must therefore access through s
    assert s.full_name == 'p'
    assert s['s1'].full_name == 'p>s1'
    assert s['s2'].full_name == 'p>s2'


def test_security_setup_prices():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[0]] = 105
    data['c2'][dts[0]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])

    assert c1.price == 105
    assert len(c1.prices) == 1
    assert c1.prices[0] == 105

    assert c2.price == 95
    assert len(c2.prices) == 1
    assert c2.prices[0] == 95

    # now with setup
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])
    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[0]] = 105
    data['c2'][dts[0]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])

    assert c1.price == 105
    assert len(c1.prices) == 1
    assert c1.prices[0] == 105

    assert c2.price == 95
    assert len(c2.prices) == 1
    assert c2.prices[0] == 95


def test_strategybase_tree_setup():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    assert len(s.data) == 3
    assert len(c1.data) == 3
    assert len(c2.data) == 3

    assert len(s.prices) == 0
    assert len(c1.prices) == 0
    assert len(c2.prices) == 0

    assert len(s.values) == 0
    assert len(c1.values) == 0
    assert len(c2.values) == 0


def test_strategybase_tree_adjust():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    s.adjust(1000)

    assert s.capital == 1000
    assert s.value == 1000
    assert c1.value == 0
    assert c2.value == 0
    assert c1.weight == 0
    assert c2.weight == 0


def test_strategybase_tree_update():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])

    c1.price == 100
    c2.price == 100

    i = 1
    s.update(dts[i], data.ix[dts[i]])

    c1.price == 105
    c2.price == 95

    i = 2
    s.update(dts[i], data.ix[dts[i]])

    c1.price == 100
    c2.price == 100


def test_strategybase_tree_allocate():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])

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
    assert s.capital == 1000 - 501
    assert s.value == 999
    assert c1.weight == 500.0 / 999
    assert c2.weight == 0


def test_strategybase_tree_allocate_child_from_strategy():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])
    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])

    s.adjust(1000)
    # since children have w == 0 this should stay in s
    s.allocate(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now allocate to c1
    s.allocate(500, 'c1')

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 501
    assert s.value == 999
    assert c1.weight == 500.0 / 999
    assert c2.weight == 0


def test_strategybase_tree_allocate_level2():
    c1 = SecurityBase('c1')
    c12 = copy.deepcopy(c1)
    c2 = SecurityBase('c2')
    c22 = copy.deepcopy(c2)
    s1 = StrategyBase('s1', [c1, c2])
    s2 = StrategyBase('s2', [c12, c22])
    m = StrategyBase('m', [s1, s2])

    s1 = m['s1']
    s2 = m['s2']

    c1 = s1['c1']
    c2 = s1['c2']
    c12 = s2['c1']
    c22 = s2['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    m.setup(data)

    i = 0
    m.update(dts[i], data.ix[dts[i]])

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
    assert s1.weight == 500.0 / 1000
    assert s2.weight == 0

    # now allocate directly to child of child
    c1.allocate(200)

    assert s1.value == 499
    assert s1.capital == 500 - 201
    assert c1.value == 200
    assert c1.weight == 200.0 / 499
    assert c1.position == 2

    assert m.capital == 1000 - 500
    assert m.value == 999
    assert s1.weight == 499.0 / 999
    assert s2.weight == 0

    assert c12.value == 0


def test_strategybase_tree_allocate_long_short():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])

    s.adjust(1000)
    c1.allocate(500)

    assert c1.position == 5
    assert c1.value == 500
    assert c1.weight == 500.0 / 999
    assert s.capital == 1000 - 501
    assert s.value == 999

    c1.allocate(-200)

    assert c1.position == 3
    assert c1.value == 300
    assert c1.weight == 300.0 / 998
    assert s.capital == 1000 - 501 + 199
    assert s.value == 998

    c1.allocate(-400)

    assert c1.position == -1
    assert c1.value == -100
    assert c1.weight == -100.0 / 997
    assert s.capital == 1000 - 501 + 199 + 399
    assert s.value == 997

    # close up
    c1.allocate(-c1.value)

    assert c1.position == 0
    assert c1.value == 0
    assert c1.weight == 0
    assert s.capital == 1000 - 501 + 199 + 399 - 101
    assert s.value == 996


def test_strategybase_tree_allocate_update():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])
    assert s.price == 100

    s.adjust(1000)

    assert s.price == 100
    assert s.value == 1000
    assert s._value == 1000

    c1.allocate(500)

    assert c1.position == 5
    assert c1.value == 500
    assert c1.weight == 500.0 / 999
    assert s.capital == 1000 - 501
    assert s.value == 999
    assert s.price == 99.9

    i = 1
    s.update(dts[i], data.ix[dts[i]])

    assert c1.position == 5
    assert c1.value == 525
    assert c1.weight == 525.0 / 1024
    assert s.capital == 1000 - 501
    assert s.value == 1024
    assert s.price == 102.4


def test_strategybase_universe():
    s = StrategyBase('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[0]] = 105
    data['c2'][dts[0]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i])

    assert len(s.universe) == 1
    assert 'c1' in s.universe
    assert 'c2' in s.universe
    assert s.universe['c1'][dts[i]] == 105
    assert s.universe['c2'][dts[i]] == 95

    # should not have children unless allocated
    assert len(s.children) == 0


def test_strategybase_allocate():
    s = StrategyBase('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[0]] = 100
    data['c2'][dts[0]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i])

    s.adjust(1000)
    s.allocate(100, 'c1')
    c1 = s['c1']

    assert c1.position == 1
    assert c1.value == 100
    assert s.value == 999


def test_strategybase_close():
    s = StrategyBase('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    s.setup(data)

    i = 0
    s.update(dts[i])

    s.adjust(1000)
    s.allocate(100, 'c1')
    c1 = s['c1']

    assert c1.position == 1
    assert c1.value == 100
    assert s.value == 999

    s.close('c1')

    assert c1.position == 0
    assert c1.value == 0
    assert s.value == 998


def test_strategybase_flatten():
    s = StrategyBase('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    s.setup(data)

    i = 0
    s.update(dts[i])

    s.adjust(1000)
    s.allocate(100, 'c1')
    c1 = s['c1']
    s.allocate(100, 'c2')
    c2 = s['c2']

    assert c1.position == 1
    assert c1.value == 100
    assert c2.position == 1
    assert c2.value == 100
    assert s.value == 998

    s.flatten()

    assert c1.position == 0
    assert c1.value == 0
    assert s.value == 996


def test_strategybase_multiple_calls():
    s = StrategyBase('s')

    dts = pd.date_range('2010-01-01', periods=5)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    data.c2[dts[0]] = 95
    data.c1[dts[1]] = 95
    data.c2[dts[2]] = 95
    data.c2[dts[3]] = 95
    data.c2[dts[4]] = 95
    data.c1[dts[4]] = 105

    s.setup(data)

    # define strategy logic
    def algo(target):
        # close out any open positions
        target.flatten()

        # get stock w/ lowest price
        c = target.universe.ix[target.now].idxmin()

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

    c2 = s['c2']
    assert c2.value == 950
    assert c2.weight == 950.0 / 999
    assert c2.price == 95

    # update out t0
    s.update(dts[i])

    c2 == s['c2']
    assert len(s.children) == 1
    assert s.value == 999
    assert s.capital == 49

    assert c2.value == 950
    assert c2.weight == 950.0 / 999
    assert c2.price == 95

    # update t1
    i = 1
    s.update(dts[i])

    assert s.value == 1049
    assert s.capital == 49
    assert len(s.children) == 1

    assert 'c2' in s.children
    c2 == s['c2']
    assert c2.value == 1000
    assert c2.weight == 1000.0 / 1049.0
    assert c2.price == 100

    # run t1 - close out c2, open c1
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1047
    assert s.capital == 2

    c1 = s['c1']
    assert c1.value == 1045
    assert c1.weight == 1045.0 / 1047
    assert c1.price == 95

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 100

    # update out t1
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1047
    assert s.capital == 2

    assert c1 == s['c1']
    assert c1.value == 1045
    assert c1.weight == 1045.0 / 1047
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
    assert c1.weight == 1100.0 / 1102
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
    assert c2.weight == 1045.0 / 1100
    assert c2.price == 95

    # update out t2
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1100
    assert s.capital == 55

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == 1045.0 / 1100
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
    assert c2.weight == 1045.0 / 1100
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
    assert c2.weight == 1045.0 / 1098
    assert c2.price == 95

    # update out t3
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1098
    assert s.capital == 53

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == 1045.0 / 1098
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
    assert c2.weight == 1045.0 / 1098
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
    assert c2.weight == 1045.0 / 1096
    assert c2.price == 95

    # update out t4
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1096
    assert s.capital == 51

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 105

    assert c2.value == 1045
    assert c2.weight == 1045.0 / 1096
    assert c2.price == 95


def test_strategybase_multiple_calls_preset_secs():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('s', [c1, c2])

    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=5)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    data.c2[dts[0]] = 95
    data.c1[dts[1]] = 95
    data.c2[dts[2]] = 95
    data.c2[dts[3]] = 95
    data.c2[dts[4]] = 95
    data.c1[dts[4]] = 105

    s.setup(data)

    # define strategy logic
    def algo(target):
        # close out any open positions
        target.flatten()

        # get stock w/ lowest price
        c = target.universe.ix[target.now].idxmin()

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
    assert s.value == 999
    assert s.capital == 49

    assert c2.value == 950
    assert c2.weight == 950.0 / 999
    assert c2.price == 95

    # update out t0
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 999
    assert s.capital == 49

    assert c2.value == 950
    assert c2.weight == 950.0 / 999
    assert c2.price == 95

    # update t1
    i = 1
    s.update(dts[i])

    assert s.value == 1049
    assert s.capital == 49
    assert len(s.children) == 2

    assert c2.value == 1000
    assert c2.weight == 1000.0 / 1049.0
    assert c2.price == 100

    # run t1 - close out c2, open c1
    s.run(s)

    assert c1.value == 1045
    assert c1.weight == 1045.0 / 1047
    assert c1.price == 95

    assert c2.value == 0
    assert c2.weight == 0
    assert c2.price == 100

    assert len(s.children) == 2
    assert s.value == 1047
    assert s.capital == 2

    # update out t1
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1047
    assert s.capital == 2

    assert c1.value == 1045
    assert c1.weight == 1045.0 / 1047
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
    assert c1.weight == 1100.0 / 1102
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
    assert c2.weight == 1045.0 / 1100
    assert c2.price == 95

    # update out t2
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1100
    assert s.capital == 55

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == 1045.0 / 1100
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
    assert c2.weight == 1045.0 / 1100
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
    assert c2.weight == 1045.0 / 1098
    assert c2.price == 95

    # update out t3
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1098
    assert s.capital == 53

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 100

    assert c2.value == 1045
    assert c2.weight == 1045.0 / 1098
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
    assert c2.weight == 1045.0 / 1098
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
    assert c2.weight == 1045.0 / 1096
    assert c2.price == 95

    # update out t4
    s.update(dts[i])

    assert len(s.children) == 2
    assert s.value == 1096
    assert s.capital == 51

    assert c1.value == 0
    assert c1.weight == 0
    assert c1.price == 105

    assert c2.value == 1045
    assert c2.weight == 1045.0 / 1096
    assert c2.price == 95


def test_strategybase_multiple_calls_no_post_update():
    s = StrategyBase('s')

    dts = pd.date_range('2010-01-01', periods=5)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    data.c2[dts[0]] = 95
    data.c1[dts[1]] = 95
    data.c2[dts[2]] = 95
    data.c2[dts[3]] = 95
    data.c2[dts[4]] = 95
    data.c1[dts[4]] = 105

    s.setup(data)

    # define strategy logic
    def algo(target):
        # close out any open positions
        target.flatten()

        # get stock w/ lowest price
        c = target.universe.ix[target.now].idxmin()

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

    c2 = s['c2']
    assert c2.value == 950
    assert c2.weight == 950.0 / 999
    assert c2.price == 95

    # update t1
    i = 1
    s.update(dts[i])

    assert s.value == 1049
    assert s.capital == 49
    assert len(s.children) == 1

    assert 'c2' in s.children
    c2 == s['c2']
    assert c2.value == 1000
    assert c2.weight == 1000.0 / 1049.0
    assert c2.price == 100

    # run t1 - close out c2, open c1
    s.run(s)

    assert len(s.children) == 2
    assert s.value == 1047
    assert s.capital == 2

    c1 = s['c1']
    assert c1.value == 1045
    assert c1.weight == 1045.0 / 1047
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
    assert c1.weight == 1100.0 / 1102
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
    assert c2.weight == 1045.0 / 1100
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
    assert c2.weight == 1045.0 / 1100
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
    assert c2.weight == 1045.0 / 1098
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
    assert c2.weight == 1045.0 / 1098
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
    assert c2.weight == 1045.0 / 1096
    assert c2.price == 95


def test_strategybase_prices():
    dts = pd.date_range('2010-01-01', periods=21)
    rawd = [13.555, 13.75, 14.16, 13.915, 13.655,
            13.765, 14.02, 13.465, 13.32, 14.65,
            14.59, 14.175, 13.865, 13.865, 13.89,
            13.85, 13.565, 13.47, 13.225, 13.385,
            12.89]
    data = pd.DataFrame(index=dts, data=rawd, columns=['a'])

    s = StrategyBase('s')
    s.setup(data)

    # buy 100 shares on day 1 - hold until end
    # just enough to buy 100 shares + 1$ commission
    s.adjust(1356.50)

    s.update(dts[0])
    # allocate all capital to child a
    # a should be dynamically created and should have
    # 100 shares allocated. s.capital should be 0
    s.allocate(s.value, 'a')

    assert s.capital == 0
    assert s.value == 1355.50
    assert len(s.children) == 1
    aae(s.price, 99.92628, 5)

    a = s['a']
    assert a.position == 100
    assert a.value == 1355.50
    assert a.weight == 1
    assert a.price == 13.555
    assert len(a.prices) == 1

    # update through all dates and make sure price is ok
    s.update(dts[1])
    aae(s.price, 101.3638, 4)

    s.update(dts[2])
    aae(s.price, 104.3863, 4)

    s.update(dts[3])
    aae(s.price, 102.5802, 4)

    # finish updates and make sure ok at end
    for i in range(4, 21):
        s.update(dts[i])

    assert len(s.prices) == 21
    aae(s.prices[-1], 95.02396, 5)
    aae(s.prices[-2], 98.67306, 5)


def test_fail_if_root_value_negative():
    s = StrategyBase('s')
    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[0]] = 100
    data['c2'][dts[0]] = 95
    s.setup(data)

    s.adjust(-100)
    # trigger update
    s.update(dts[0])

    assert s.bankrupt

    # make sure only triggered if root negative
    c1 = StrategyBase('c1')
    s = StrategyBase('s', children=[c1])
    c1 = s['c1']

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
    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[0]] = 100
    data['c2'][dts[0]] = 95

    # must setup tree because if not negative root error pops up first
    c1 = StrategyBase('c1')
    s = StrategyBase('s', children=[c1])
    c1 = s['c1']
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
        if 'Could not update' not in str(e):
            assert False


def test_strategybase_tree_rebalance():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])

    s.adjust(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now rebalance c1
    s.rebalance(0.5, 'c1')

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 501
    assert s.value == 999
    assert c1.weight == 500.0 / 999
    assert c2.weight == 0


def test_rebalance_child_not_in_tree():
    s = StrategyBase('p')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i])
    s.adjust(1000)

    # rebalance to 0 w/ child that is not present - should ignore
    s.rebalance(0, 'c2')

    assert s.value == 1000
    assert s.capital == 1000
    assert len(s.children) == 0


def test_strategybase_tree_rebalance_to_0():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])

    s.adjust(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # now rebalance c1
    s.rebalance(0.5, 'c1')

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 501
    assert s.value == 999
    assert c1.weight == 500.0 / 999
    assert c2.weight == 0

    # now rebalance c1
    s.rebalance(0, 'c1')

    assert c1.position == 0
    assert c1.value == 0
    assert s.capital == 1000 - 501 + 499
    assert s.value == 998
    assert c1.weight == 0
    assert c2.weight == 0


def test_strategybase_tree_rebalance_level2():
    c1 = SecurityBase('c1')
    c12 = copy.deepcopy(c1)
    c2 = SecurityBase('c2')
    c22 = copy.deepcopy(c2)
    s1 = StrategyBase('s1', [c1, c2])
    s2 = StrategyBase('s2', [c12, c22])
    m = StrategyBase('m', [s1, s2])

    s1 = m['s1']
    s2 = m['s2']

    c1 = s1['c1']
    c2 = s1['c2']

    c12 = s2['c1']
    c22 = s2['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    m.setup(data)

    i = 0
    m.update(dts[i], data.ix[dts[i]])

    m.adjust(1000)

    assert m.value == 1000
    assert m.capital == 1000
    assert s1.value == 0
    assert s2.value == 0
    assert c1.value == 0
    assert c2.value == 0

    # now rebalance child s1 - since its children are 0, no waterfall alloc
    m.rebalance(0.5, 's1')

    assert s1.value == 500
    assert m.capital == 1000 - 500
    assert m.value == 1000
    assert s1.weight == 500.0 / 1000
    assert s2.weight == 0

    # now allocate directly to child of child
    s1.rebalance(0.4, 'c1')

    assert s1.value == 499
    assert s1.capital == 500 - 201
    assert c1.value == 200
    assert c1.weight == 200.0 / 499
    assert c1.position == 2

    assert m.capital == 1000 - 500
    assert m.value == 999
    assert s1.weight == 499.0 / 999
    assert s2.weight == 0

    assert c12.value == 0

    # now rebalance child s1 again and make sure c1 also gets proportional
    # increase
    m.rebalance(0.8, 's1')
    assert s1.value == 798.2
    aae(m.capital, 199.8, 1)
    assert m.value == 998
    assert s1.weight == 798.2 / 998
    assert s2.weight == 0
    assert c1.value == 300.0
    assert c1.weight == 300.0 / 798.2
    assert c1.position == 3

    # now rebalance child s1 to 0 - should close out s1 and c1 as well
    m.rebalance(0, 's1')

    assert s1.value == 0
    assert m.capital == 997
    assert m.value == 997
    assert s1.weight == 0
    assert s2.weight == 0
    assert c1.weight == 0


def test_strategybase_tree_rebalance_base():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(data)

    i = 0
    s.update(dts[i], data.ix[dts[i]])

    s.adjust(1000)

    assert s.value == 1000
    assert s.capital == 1000
    assert c1.value == 0
    assert c2.value == 0

    # check that 2 rebalances of equal weight lead to two different allocs
    # since value changes after first call
    s.rebalance(0.5, 'c1')

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 501
    assert s.value == 999
    assert c1.weight == 500.0 / 999
    assert c2.weight == 0

    s.rebalance(0.5, 'c2')

    assert c2.position == 4
    assert c2.value == 400
    assert s.capital == 1000 - 501 - 401
    assert s.value == 998
    assert c2.weight == 400.0 / 998
    assert c1.weight == 500.0 / 998

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
    s.rebalance(0.5, 'c1', base=base)

    assert c1.position == 5
    assert c1.value == 500
    assert s.capital == 1000 - 501
    assert s.value == 999
    assert c1.weight == 500.0 / 999
    assert c2.weight == 0

    s.rebalance(0.5, 'c2', base=base)

    assert c2.position == 5
    assert c2.value == 500
    assert s.capital == 1000 - 501 - 501
    assert s.value == 998
    assert c2.weight == 500.0 / 998
    assert c1.weight == 500.0 / 998


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
    s = StrategyBase('s')

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)

    s.setup(data)
    s.update(dts[0])
    s.adjust(1000)

    s.allocate(500, 'c1')
    assert s.capital == 499

    s.set_commissions(lambda x, y: 0.0)
    s.allocate(-500, 'c1')
    assert s.capital == 999


def test_strategy_tree_proper_return_calcs():
    s1 = StrategyBase('s1')
    s2 = StrategyBase('s2')
    m = StrategyBase('m', [s1, s2])

    s1 = m['s1']
    s2 = m['s2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data.loc['c1', dts[1]] = 105
    data.loc['c2', dts[1]] = 95

    m.setup(data)

    i = 0
    m.update(dts[i], data.ix[dts[i]])

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
    assert s1.weight == 500.0 / 1000
    assert s1.price == 100
    assert s2.weight == 0

    # allocate to child2 via master method
    m.allocate(500, 's2')

    assert m.capital == 0
    assert m.value == 1000
    assert m.price == 100
    assert s1.value == 500
    assert s1.weight == 500.0 / 1000
    assert s1.price == 100
    assert s2.value == 500
    assert s2.weight == 500.0 / 1000
    assert s2.price == 100

    # now allocate and incur commission fee
    s1.allocate(500, 'c1')

    assert m.capital == 0
    assert m.value == 999
    assert m.price == 99.9
    assert s1.value == 499
    assert s1.weight == 499.0 / 999.0
    assert s1.price == 99.8
    assert s2.value == 500
    assert s2.weight == 500.0 / 999.0
    assert s2.price == 100


def test_strategy_tree_proper_universes():
    def do_nothing(x):
        return True

    child1 = Strategy('c1', [do_nothing], ['b', 'c'])
    master = Strategy('m', [do_nothing], [child1, 'a'])

    child1 = master['c1']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(
        {'a': pd.TimeSeries(data=1, index=dts, name='a'),
         'b': pd.TimeSeries(data=2, index=dts, name='b'),
         'c': pd.TimeSeries(data=3, index=dts, name='c')})

    master.setup(data)

    assert len(master.children) == 2
    assert 'c1' in master.children
    assert 'a' in master.children
    assert len(master.universe.columns) == 2
    assert 'c1' in master.universe.columns
    assert 'a' in master.universe.columns

    assert len(child1.universe.columns) == 2
    assert 'b' in child1.universe.columns
    assert 'c' in child1.universe.columns


def test_strategy_tree_paper():
    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['a'], data=100.)
    data['a'].ix[dts[1]] = 101
    data['a'].ix[dts[2]] = 102

    s = Strategy('s',
                 [bt.algos.SelectWhere(data > 100),
                  bt.algos.WeighEqually(),
                  bt.algos.Rebalance()])

    m = Strategy('m', [], [s])
    s = m['s']

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
    assert s.price == 99.9901

    s.update(dts[2])
    m.run()

    assert m.price == 100
    assert m.value == 0
    assert s.value == 0
    aae(s.price, 100.9801, 4)
