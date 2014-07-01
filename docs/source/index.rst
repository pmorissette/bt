bt - A Flexible Backtesting Framework for Python
================================================

What is bt?
-----------

**bt** is a flexible backtesting framework for Python used to test quantitative
trading strategies. Backtesting is the process of testing a strategy over a given 
data set. This framework allows you to easily create strategies that mix and match 
different :class:`Algos <bt.core.Algo>`. It aims to foster the creation of easily testable, re-usable and 
flexible blocks of strategy logic to facilitate the rapid development of complex 
trading strategies. 

The goal: to save **quants** from re-inventing the wheel and let them focus on the 
important part of the job - strategy development.

**bt** is coded in **Python** and joins a vibrant and rich ecosystem for data analysis. 
Numerous libraries exist for machine learning, signal processing and statistics and can be leveraged to avoid
re-inventing the wheel - something that happens all too often when using other
languages that don't have the same wealth of high-quality, open-source projects.

A Quick Example
---------------

Here is a quick taste of bt:

.. include:: intro.rst

As you can see, the strategy logic is easy to understand and more importantly, 
easy to modify.  The idea of using simple, composable Algos to create strategies is one of the
core building blocks of bt. 

The Tree Structure
------------------

In addition to the concept of :class:`Algos <bt.core.Algo>` and :class:`AlgoStacks <bt.core.AlgoStack>`, a tree structure lies 
at the heart of the framework.  It allows you to mix and match securities and strategies in order to express 
your sophisticated trading ideas.  Here is a very simple diagram to help explain this concept:

.. image:: _static/tree1.png
    :align: center
    :alt: simple tree structure

This diagram represents the strategy we tested above. A simple :class:`strategy <bt.core.Strategy>` 
with two children that happen to be :class:`securities <bt.core.SecurityBase>`. However, children nodes don't have to be
securities. They can also be strategies. This convept is very powerful as it
allows you to combine strategies together and allocate capital dynamically
between different strategies as time progresses using sophisticated allocation
logic. This is similar to what hedge funds do - they have a portfolio of strategies and dynamically allocate capital
according to a set of rules. 

For example, say we didn't mind having a passive bond allocation (AGG in the
above graph), but we wanted to swap out the equity portion (SPY) for something a
little more sophisticated. In this case, we will swap out the SPY node for another strategy.
This strategy could be a momentum strategy that attempts to pick the best
performing ETF every month (to keep it simple, let's say it picks either the SPY
or the EEM based on total return over the past 3 months).

Here is the updated graph:

.. image:: _static/tree2.png
    :align: center
    :alt: advanced tree structure

This approach allows you to build complex systems even though all of the building
blocks may be relatively simple. Hopefully you can see how powerful this can be
when desingning and testing quantitative strategies.

Oh and here's the code for the second example - not much more complex:

.. code:: python

    import bt

    # create the momentum strategy - we will specify the children (3rd argument)
    # to limit the universe the strategy can choose from
    mom_s = bt.Strategy('mom_s', [bt.algos.RunMonthly(),
                                  bt.algos.SelectAll(),
                                  bt.algos.SelectMomentum(1),
                                  bt.algos.WeighEqually(),
                                  bt.algos.Rebalance()],
                        ['spy', 'eem'])

    # create the master strategy - this is the top-most node in the tree
    # Once again, we are also specifying  the children. In this case, one of the
    # children is a Security and the other is a Strategy.
    master = bt.Strategy('master', [bt.algos.RunMonthly(),
                                    bt.algos.SelectAll(),
                                    bt.algos.WeighEqually(),
                                    bt.algos.Rebalance()],
                        [mom_s, 'agg'])

    # create the backtest and run it
    t = bt.Backtest(master, data)
    r = bt.run(t)

So there you have it. Please read the rest of the docs to have a better idea of
all the features packed into bt.

Features
---------

* **Tree Structure**
    The tree structure facilitates the construction and composition of complex algorithmic trading 
    strategies that are modular and re-usable. Furthermore, each tree :class:`Node <bt.core.Node>`
    has its own :func:`price index <bt.core.Node.prices>` that can be
    used by Algos to determine a Node's allocation. 

* **Algorithm Stacks**
    :class:`Algos <bt.core.Algo>` and :class:`AlgoStacks <bt.core.AlgoStack>` are
    another core feature that facilitate the creation of modular and re-usable strategy
    logic. Due to their modularity, these logic blocks are also easier to test -
    an important step in building robust financial solutions.

* **Charting and Reporting**
    bt also provides many useful charting functions that help visualize backtest
    results. We also plan to add more charts, tables and report formats in the future, 
    such as automatically generated PDF reports.

* **Detailed Statistics**
    Furthermore, bt calculates a bunch of stats relating to a backtest and offers a quick way to compare
    these various statistics across many different backtests via :class:`Results'
    <bt.backtests.Results>` display methods.


Roadmap
--------

Future development efforts will focus on:

* **Speed**
    Due to the flexible nature of bt, a tradeoff had to be made between
    usability and performance. Usability will always be the priority, but we do
    wish to enhance the performance as much as possible.

* **Algos**
    We will also be developping more algorithms as time goes on. We also
    encourage anyone to contribute their own algos as well.

* **Charting and Reporting**
    This is another area we wish to constantly improve on
    as reporting is an important aspect of the job. Charting and reporting also
    facilitate finding bugs in strategy logic.


.. toctree::
   :maxdepth: 2
   :hidden:

    Overview <index>
    Installation Guide <install>
    Getting Started <quickstart>
    API <bt>
