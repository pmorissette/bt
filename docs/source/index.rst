bt - A Flexible Backtesting Framework for Python
================================================

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

.. notebook:: intro.ipynb

As you can see, the strategy logic is easy to understand and more importantly, 
easy to change.  The idea of using simple, composable Algos to create strategies is one of the
core building blocks of bt. 

The Tree Structure
------------------

The tree structure lies at the core of the framework. It allows you to mix and
match securities and strategies in order to create sophisticated trading strategies. 
Here is a very simple diagram to help explain this concept:

.. image:: _static/tree1.png
    :align: center
    :alt: simple tree structure

This diagram represents the strategy we tested above. A simple :class:`strategy <bt.core.Strategy>` 
with two children that happen to be :class:`securities <bt.core.SecurityBase>`. However, children don't have to be
securities. They can also be strategies. This convept is very powerful as it
allows you to combine strategies together and allocate capital dynamically
between different strategies as time progresses. This is similar to what a hedge
fund does - it has a portfolio of strategies and dynamically allocates capital
according to a set of rules. 

For example, say we didn't mind having a passive bond allocation (AGG in the
above graph), but we wanted to swap out the equity portion (SPY) for something a
little more sophisticated. We will swap out the SPY node for another strategy.
This strategy could be a momentum strategy that attempts to pick the best
performing ETF every month. 

Here is the updated graph:

.. image:: _static/tree2.png
    :align: center
    :alt: advanced tree structure

This framework allows you to build complex systems even though all the building
blocks may be relatively simple. Hopefully you can see how powerful this can be
when desingning and testing quantitative strategies.

Features
---------

* **Tree Structure**
    The tree structure allows the construction and composition of complex algorithmic trading strategies that
    are modular and re-usable. Furthermore, each tree :class:`Node <bt.core.Node>`
    has its own :func:`price index <bt.core.Node.prices>` that can be
    used by Algos to determine a Node's allocation. 

* **Algorithm Stacks**
    :class:`Algos <bt.core.Algo>` and :class:`AlgoStacks <bt.core.AlgoStack>` are
    another core feature that facilitate the creation of modular and re-usable strategy
    logic. 

* **Charting and Reporting**
    bt also provides many useful charting functions that help visualize backtest
    results. We also plan to add more charts, tables and report formats in the future, 
    such as automatically generated PDF reports.

* **Detailed Statistics**
    Furthermore, bt calculates a bunch of stats relating to a backtest and offers a quick way to compare
    these various statistics across many different backtests via :class:`Results'
    <bt.backtests.Results>` display methods.


Contents
--------

.. toctree::
   :maxdepth: 2

    API <bt>
