.. image:: http://pmorissette.github.io/bt/_static/logo.png

bt - Flexible Backtesting for Python
====================================

bt is currently in alpha stage - if you find a bug, please submit an issue.

Read the docs here: http://pmorissette.github.io/bt.

What is bt?
-----------

**bt** is a flexible backtesting framework for Python used to test quantitative
trading strategies. **Backtesting** is the process of testing a strategy over a given 
data set. This framework allows you to easily create strategies that mix and match 
different `Algos <http://pmorissette.github.io/bt/bt.html#bt.core.Algo>`_. It aims to foster the creation of easily testable, re-usable and 
flexible blocks of strategy logic to facilitate the rapid development of complex 
trading strategies. 

The goal: to save **quants** from re-inventing the wheel and let them focus on the 
important part of the job - strategy development.

**bt** is coded in **Python** and joins a vibrant and rich ecosystem for data analysis. 
Numerous libraries exist for machine learning, signal processing and statistics and can be leveraged to avoid
re-inventing the wheel - something that happens all too often when using other
languages that don't have the same wealth of high-quality, open-source projects.

bt is built atop `ffn <https://github.com/pmorissette/ffn>`_ - a financial function library for Python. Check it out!

Features
---------

* **Tree Structure**
    `The tree structure <http://pmorissette.github.io/bt/tree.html>`_ facilitates the construction and composition of complex algorithmic trading 
    strategies that are modular and re-usable. Furthermore, each tree `Node
    <http://pmorissette.github.io/bt/bt.html#bt.core.Node>`_
    has its own price index that can be
    used by Algos to determine a Node's allocation. 

* **Algorithm Stacks**
    `Algos <http://pmorissette.github.io/bt/bt.html#bt.core.Algo>`_ and `AlgoStacks <http://pmorissette.github.io/bt/bt.html#bt.core.AlgoStack>`_ are
    another core feature that facilitate the creation of modular and re-usable strategy
    logic. Due to their modularity, these logic blocks are also easier to test -
    an important step in building robust financial solutions.

* **Charting and Reporting**
    bt also provides many useful charting functions that help visualize backtest
    results. We also plan to add more charts, tables and report formats in the future, 
    such as automatically generated PDF reports.

* **Detailed Statistics**
    Furthermore, bt calculates a bunch of stats relating to a backtest and offers a quick way to compare
    these various statistics across many different backtests via `Results'
    <http://pmorissette.github.io/bt/bt.html#bt.backtest.Result>`_ display methods.


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

Installing bt
-------------

The easiest way to install ``bt`` is from the `Python Package Index <https://pypi.python.org/pypi/bt/>`_
using ``pip`` or ``easy_insatll``:

.. code-block:: bash

    $ pip install bt 

Since bt has many dependencies, we strongly recommend installing the `Anaconda Scientific Python
Distribution <https://store.continuum.io/cshop/anaconda/>`_, especially on Windows. This distribution 
comes with many of the required packages pre-installed, including pip. Once Anaconda is installed, the above 
command should complete the installation. 

bt should be compatible with Python 2.7 and Python 3 thanks to the contributions
made by fellow users.

Recommended Setup
-----------------

We believe the best environment to develop with bt is the `IPython Notebook
<http://ipython.org/notebook.html>`__. From their homepage, the IPython Notebook
is:

    "[...] a web-based interactive computational environment
    where you can combine code execution, text, mathematics, plots and rich
    media into a single document [...]"

This environment allows you to plot your charts in-line and also allows you to
easily add surrounding text with Markdown. You can easily create Notebooks that
you can share with colleagues and you can also save them as PDFs. If you are not
yet convinced, head over to their website.

Special Thanks
--------------

A special thanks to the following contributors for their involvement with the project:

* Vladimir Filimonov `@vfilimonov <https://github.com/vfilimonov>`_ 


License
-------

MIT
