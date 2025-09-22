![](http://pmorissette.github.io/bt/_static/logo.png)

[![Build Status](https://github.com/pmorissette/bt/workflows/Build%20Status/badge.svg)](https://github.com/pmorissette/bt/actions/)
[![PyPI Version](https://img.shields.io/pypi/v/bt)](https://pypi.org/project/bt/)
[![PyPI License](https://img.shields.io/pypi/l/bt)](https://github.com/pmorissette/bt/blob/master/LICENSE)

# bt - Flexible Backtesting for Python

bt is currently in alpha stage - if you find a bug, please submit an issue.

Read the docs here: http://pmorissette.github.io/bt.

## What is bt?

**bt** is a flexible backtesting framework for Python used to test quantitative
trading strategies. **Backtesting** is the process of testing a strategy over a given
data set. This framework allows you to easily create strategies that mix and match
different [Algos](http://pmorissette.github.io/bt/bt.html#bt.core.Algo). It aims to foster the creation of easily testable, re-usable and
flexible blocks of strategy logic to facilitate the rapid development of complex
trading strategies.

The goal: to save **quants** from re-inventing the wheel and let them focus on the
important part of the job - strategy development.

**bt** is coded in **Python** and joins a vibrant and rich ecosystem for data analysis.
Numerous libraries exist for machine learning, signal processing and statistics and can be leveraged to avoid
re-inventing the wheel - something that happens all too often when using other
languages that don't have the same wealth of high-quality, open-source projects.

bt is built atop [ffn](https://github.com/pmorissette/ffn) - a financial function library for Python. Check it out!

## Features

* **Tree Structure**
    [The tree structure](http://pmorissette.github.io/bt/tree.html) facilitates the construction and composition of complex algorithmic trading
    strategies that are modular and re-usable. Furthermore, each tree [Node](http://pmorissette.github.io/bt/bt.html#bt.core.Node) has its own
    price index that can be used by Algos to determine a Node's allocation.

* **Algorithm Stacks**
    [Algos](http://pmorissette.github.io/bt/bt.html#bt.core.Algo) and [AlgoStacks](http://pmorissette.github.io/bt/bt.html#bt.core.AlgoStack) are
    another core feature that facilitate the creation of modular and re-usable strategy
    logic. Due to their modularity, these logic blocks are also easier to test -
    an important step in building robust financial solutions.

* **Charting and Reporting**
    bt also provides many useful charting functions that help visualize backtest
    results. We also plan to add more charts, tables and report formats in the future,
    such as automatically generated PDF reports.

* **Detailed Statistics**
    Furthermore, bt calculates a bunch of stats relating to a backtest and offers a quick way to compare
    these various statistics across many different backtests via [Results](http://pmorissette.github.io/bt/bt.html#bt.backtest.Result) display methods.


## Roadmap

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

## Installing bt

The easiest way to install `bt` is from the [Python Package Index](https://pypi.python.org/pypi/bt/)
using `pip`:

```bash
pip install bt
```


Since bt has many dependencies, we strongly recommend installing the [Anaconda Scientific Python
Distribution](https://store.continuum.io/cshop/anaconda/), especially on Windows. This distribution
comes with many of the required packages pre-installed, including pip. Once Anaconda is installed, the above
command should complete the installation.

## Recommended Setup

We believe the best environment to develop with bt is the [IPython Notebook](http://ipython.org/notebook.html).
From their homepage, the IPython Notebook is:

    "[...] a web-based interactive computational environment
    where you can combine code execution, text, mathematics, plots and rich
    media into a single document [...]"

This environment allows you to plot your charts in-line and also allows you to
easily add surrounding text with Markdown. You can easily create Notebooks that
you can share with colleagues and you can also save them as PDFs. If you are not
yet convinced, head over to their website.

## Contributing to bt

A Makefile is available to simplify local development.
[GNU Make](https://www.gnu.org/software/make/) is required to run the `make` targets directly, and it is not often preinstalled [on Windows systems](https://gnuwin32.sourceforge.net/packages/make.htm).

When developing in Python, it's advisable to [create and activate a virtual environment](https://docs.python.org/3/library/venv.html) to keep the project's dependencies isolated from the system.

After the usual preparation steps for [contributing to a GitHub project](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) (forking, cloning, creating a feature branch), run `make develop` to install dependencies in the environment.

While making changes and adding tests, run `make lint` and `make test` often to check for mistakes.

After [commiting and pushing changes](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project?tool=webui#making-and-pushing-changes), [create a Pull Request](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project?tool=webui#making-a-pull-request) to discuss and get feedback on the proposed feature or fix.
