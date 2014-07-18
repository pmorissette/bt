bt - Flexible Backtesting for Python
====================================

What is bt?
-----------

**bt** is a flexible backtesting framework for Python used to test quantitative
trading strategies. **Backtesting** is the process of testing a strategy over a given 
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

bt is built atop `ffn <https://github.com/pmorissette/ffn>`_ - a financial function library for Python. Check it out!

A Quick Example
---------------

Here is a quick taste of bt:

.. include:: intro.rst

As you can see, the strategy logic is easy to understand and more importantly, 
easy to modify.  The idea of using simple, composable Algos to create strategies is one of the
core building blocks of bt. 

Features
---------

* **Tree Structure**
    :doc:`The tree structure <tree>` facilitates the construction and composition of complex algorithmic trading 
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
    <bt.backtests.Result>` display methods.


Roadmap
--------

Future development efforts will focus on:

* **Speed**
    Due to the flexible nature of bt, a trade-off had to be made between
    usability and performance. Usability will always be the priority, but we do
    wish to enhance the performance as much as possible.

* **Algos**
    We will also be developing more algorithms as time goes on. We also
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
    All About Algos <algos>
    The Tree Structure <tree>
    Examples <examples>
    API <bt>
