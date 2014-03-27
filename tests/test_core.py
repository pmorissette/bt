from bt.core import Node, StrategyBase, SecurityBase
import pandas as pd


def test_node_tree():
    c1 = Node('c1')
    c2 = Node('c2')
    p = Node('p', children=[c1, c2])

    assert len(p.children) == 2
    assert 'c1' in p.children
    assert 'c2' in p.children
    assert p == p['c1'].parent
    assert p == p['c2'].parent

    m = Node('m', children=[p])

    assert len(m.children) == 1
    assert 'p' in m.children
    assert m['p'].parent == m
    assert len(p.children) == 2
    assert 'c1' in p.children
    assert 'c2' in p.children
    assert p == p['c1'].parent
    assert p == p['c2'].parent


def test_strategy_tree():
    s1 = SecurityBase('s1')
    s2 = SecurityBase('s2')
    s = StrategyBase('p', [s1, s2])

    assert len(s.children) == 2
    assert 's1' in s.children
    assert 's2' in s.children
    assert s == s['s1'].parent
    assert s == s['s2'].parent


def test_strategy_tree_setup():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(dts)

    c1 = s['c1']
    c2 = s['c2']

    assert len(s.data) == 3
    assert len(c1.data) == 3
    assert len(c2.data) == 3

    assert len(s.prices) == 0
    assert len(c1.prices) == 0
    assert len(c2.prices) == 0

    assert len(s.values) == 0
    assert len(c1.values) == 0
    assert len(c2.values) == 0


def test_strategy_tree_adjust():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(dts)

    s.adjust(1000)

    assert s.capital == 1000
    assert s.value == 1000
    assert c1.value == 0
    assert c2.value == 0
    assert c1.weight == 0
    assert c2.weight == 0


def test_strategy_tree_update():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(dts)

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


def test_strategy_tree_allocate():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])
    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(dts)

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


def test_strategy_tree_allocate_level2():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s1 = StrategyBase('s1', [c1, c2])
    s2 = StrategyBase('s2', [c1, c2])
    m = StrategyBase('m', [s1, s2])
    # re-reference
    s1 = m['s1']
    s2 = m['s2']
    s1c1 = s1['c1']
    s2c1 = s2['c1']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    m.setup(dts)

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
    s1c1.allocate(200)

    assert s1.value == 499
    assert s1.capital == 500 - 201
    assert s1.children['c1'].value == 200
    assert s1.children['c1'].weight == 200.0 / 499
    assert s1.children['c1'].position == 2

    assert m.capital == 1000 - 500
    assert m.value == 999
    assert s1.weight == 499.0 / 999
    assert s2.weight == 0

    assert s2c1.value == 0


def test_strategy_tree_allocate_long_short():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])
    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(dts)

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


def test_strategy_tree_allocate_update():
    c1 = SecurityBase('c1')
    c2 = SecurityBase('c2')
    s = StrategyBase('p', [c1, c2])
    c1 = s['c1']
    c2 = s['c2']

    dts = pd.date_range('2010-01-01', periods=3)
    data = pd.DataFrame(index=dts, columns=['c1', 'c2'], data=100)
    data['c1'][dts[1]] = 105
    data['c2'][dts[1]] = 95

    s.setup(dts)

    i = 0
    s.update(dts[i], data.ix[dts[i]])
    assert s.price == 100

    s.adjust(1000)

    assert s.price == 100

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
